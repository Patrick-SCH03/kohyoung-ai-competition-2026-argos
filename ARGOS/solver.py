"""ICCAS 2026 KohYoung AI Contest - ARGOS: solve(data_folder) writes data_folder/output_fov.csv.

Per board:
  1. read the board (geometry.Problem); when the worker pool is ready, cover variants 1-3 (and the extra variant) go
     to the worker processes at once (_entry_dispatch);
  2. write an always-valid trivial plan, so an output exists from the first moment (anytime);
  3. plan(): build the cover variant(s) the main process owns, write a quick plan of the first, keep the cover with
     the lowest capture proxy, then
       - lattice boards (most ordinary parts on lattice lines): finish that cover, then re-cover the ordinary parts
         with the set-cover local search (setcover.py) and keep the alternative only if it verifies and its exact
         score is strictly lower (_plan_ls);
       - camera-bound boards: the same alternative search, entered only when the finished plan is camera-bound;
       - otherwise: visit order and centre refinement (_order_and_refine), then the finishing stages (_finish);
  4. polish the finished plan (order moves, member moves; _polish), keeping only verified, strictly better plans;
  5. write the plan, then take a worker's finished plan if its exact score is strictly lower (_portfolio_pick);
  6. on a camera-bound final plan, re-split neighbouring FOV pairs (recover.py, main process only), keeping the result
     only if it verifies and its exact score is strictly lower (_recover).
The import compiles the kernels and starts the worker pool, then collects its garbage, so that collection does not
land in the first solve() (a process per board would pay it on every board).
Every write is atomic, every later stage is isolated (an exception keeps the plan already written), and every time
limit is a fraction of time_budget. The objective is the checker's min over k of CT(k) + 10 k (ct_sim).
Config: DEFAULT_CONFIG, overridable only through solve(..., config=dict) (analysis tooling; the grader passes none).
"""
import os
import gc
import json
import math
import time
import numpy as np
import pandas as pd

try:
    from .geometry import Problem, FOV
    from .cover import cover_group, remove_redundant, recenter, eliminate, GridCover, TOL
    from . import setcover
    from .order import build_sequence, critical_relocate
    from .verify import verify, repair
    from .refine import refine_positions
    from .assign import reassign_heavy
    from .anneal import anneal_order, polish_order, polish_members
    from . import polish_nb
    from .release import improve_release_sequence
    from .atomic_io import replace_output
    from . import ct_sim
    from . import parallel
except ImportError:  # loaded as a plain module rather than as a package
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from geometry import Problem, FOV
    from cover import cover_group, remove_redundant, recenter, eliminate, GridCover, TOL
    import setcover
    from order import build_sequence, critical_relocate
    from verify import verify, repair
    from refine import refine_positions
    from assign import reassign_heavy
    from anneal import anneal_order, polish_order, polish_members
    import polish_nb
    from release import improve_release_sequence
    from atomic_io import replace_output
    import ct_sim
    import parallel

try:  # the pair re-cover stage runs in the main process only; the workers neither import nor compile it
    import multiprocessing as _mp
    if _mp.current_process().name.startswith('ARGOS-worker'):
        recover = None
    else:
        try:
            from . import recover
        except ImportError:
            import recover
except Exception:  # any failure: the stage is skipped and the plan is the one before it
    recover = None

LAST_STATS = {}  # what the last solve() did (stages, fallbacks, the portfolio record), for analysis tools and tests
_CTX = {}  # the board solve() is on (dataset folder, config, the portfolio request); empty outside solve()
PORTFOLIO_WAIT = 0.3  # seconds the main waits for the workers' plans once its own plan is written
PORTFOLIO_CAP = 0.95  # ... never past this fraction of the time budget
PORTFOLIO_WAIT_TARGET = 0.6  # early-dispatched boards: one longer wait (from the main plan's end) for the variant the
#                               main would have kept by its cover proxy, when that variant has not replied yet

DEFAULT_CONFIG = dict(
    eliminate=True,             # cover: dissolve FOVs whose members fit their neighbours
    elim_chains=True,           # ... with depth-2 ejection chains
    cover_portfolio=True,       # build all COVER_VARIANTS, not only the first
    portfolio_slack=0.5,        # covers within this capture proxy of the best are kept as candidates ...
    portfolio_max=1,            # ... at most this many (1: the lowest proxy only)
    order='portfolio',          # 'portfolio' (structural candidates, scored) | 'tsp' (travel only)
    objective='sweep',          # 'sweep' (min_k CT(k) + 10k, the checker's score) | 'ct' (CT at max_core)
    lpt='auto',                 # member order inside FOVs: 'auto' (best of three) | True (longest first) | False
    refine=True,                # slide free FOV centres to shorten travel
    crit_relocate=True,         # move the critical FOV earlier while the score improves
    crit_rounds=8,
    crit_evals=24,
    reassign=True,              # move heavy parts to earlier FOVs that also cover them
    reassign_top=40,
    anneal=True,                # simulated annealing of the visit order
    anneal_ms=350.0,
    anneal_min_tail=2.0,
    anneal_scale='gap',         # annealing budget x clamp(gap-to-load-bound / 6 s, 0.3, 1)
    polish=True,                # greedy order polish of the finished plan
    polish_stall=1000,
    polish_evals=8000,
    polish_members=True,        # greedy moves of normal parts between covering FOVs after the order polish
    members_stall=250,
    members_evals=2000,
    polish_nb_scale=4,          # compiled polishes with this many times the budget (0: the Python loops)
    time_budget=3.0,            # seconds; every stage limit is a fraction of it
    early_write=True,           # anytime writes (trivial plan, quick plan, finished plans)
    verify=True,                # self-check with the checker's rules and repair
    debug=False,
    ls_cover=True,              # set-cover local search on lattice boards
    ls_camera=True,             # ... and, gated, on camera-bound boards
    portfolio_final_frac=0.45,  # no new local-search stage past this fraction of the budget
    polish_lattice=False,       # the order polish skips lattice boards (little found there at the highest cost)
    block_insert=True,          # order: re-insert small same-step runs inside the other runs
    block_insert_top=1,         # ... on this many best structural candidates
    recover=True,               # pair re-cover of the final plan when it is camera-bound (recover.py)
    recover_parts=48,           # ... pairs with at most this many parts
    recover_sims=1600,          # ... at most this many simulated candidates per board
    recover_pairs=4000,         # ... at most this many visited pairs per board
    recover_ms=150.0,           # ... safety stop (wall clock, ms)
)

COVER_VARIANTS = (  # the main process builds variant 0 and, without the pool, all four
    dict(weights='invarea^1.5', grid_res=0.5, two_phase_side=False),
    dict(weights='invarea', grid_res=0.5, two_phase_side=True),
    dict(weights='invarea', grid_res=1.0, two_phase_side=False),
    dict(weights='uniform', grid_res=1.0, two_phase_side=True),
)
EXTRA_VARIANTS = (dict(weights='uniform', grid_res=0.5, two_phase_side=False),)  # extra workers only


def variant_def(v):
    """Cover variant v: 0-3 the main process's four, 4 and up the extra workers' EXTRA_VARIANTS."""
    return COVER_VARIANTS[v] if v < len(COVER_VARIANTS) else EXTRA_VARIANTS[v - len(COVER_VARIANTS)]


def _extra_members():
    """[(variant, config override)] for the extra workers, in order (the main process never builds these)."""
    return [(len(COVER_VARIANTS) + j, {}) for j in range(len(EXTRA_VARIANTS))]


def get_config(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    return cfg


def place_big_tiles(P, i, gc):
    """Minimal contiguous tiles of big component i, each placed inside its slack where the
    count map says it absorbs the most neighbours."""
    W, H = float(P.W[i]), float(P.H[i])
    w, h = float(P.w[i]), float(P.h[i])
    nx = math.ceil(w / W) if w > W else 1
    ny = math.ceil(h / H) if h > H else 1
    s = int(P.step[i])

    def choose(axis_lo, axis_hi, L, n, other_lo, other_hi, x_axis):
        picks = []
        prev_left, prev_right = None, axis_lo
        for k in range(n):
            a = axis_hi - (n - k) * L
            if prev_left is not None:
                a = max(a, prev_left)
            lo_c, hi_c = a + L / 2.0, prev_right + L / 2.0
            r = gc.best_in(lo_c, hi_c, other_lo, other_hi) if x_axis else gc.best_in(other_lo, other_hi, lo_c, hi_c)
            if r is None or r[2] <= 1e-12:
                c = min(max(hi_c if k < n - 1 else lo_c, lo_c), hi_c)
            else:
                c = r[0] if x_axis else r[1]
            picks.append(c)
            prev_left, prev_right = c - L / 2.0, c + L / 2.0
        return picks

    if nx == 1:
        xs = [float(P.cx[i])]
    else:
        if ny > 1:
            y_lo, y_hi = P.bry[i] - ny * H + H / 2.0, P.tly[i] + H / 2.0
        else:
            y_lo = y_hi = float(P.cy[i])
        xs = choose(P.tlx[i], P.brx[i], W, nx, y_lo, y_hi, True)
    ys = [float(P.cy[i])] if ny == 1 else choose(P.tly[i], P.bry[i], H, ny, xs[0], xs[0], False)
    out = []
    for x in xs:
        for y in ys:
            mem = gc.commit(x, y)
            out.append(FOV(x, y, [int(i)] + [int(m) for m in mem], s, fixed=True))
    return out


def build_fovs(P, cfg, weights, grid_res, two_phase_side):
    """One cover variant: fiducials alone first; then per step the big parts' tiles and a greedy cover of the
    ordinary parts on a grid of grid_res mm with the given weights (two_phase_side: tiles at their default places, the
    side parts covered first and the normal parts absorbed); then redundant FOVs removed, FOVs eliminated and free
    centres recentred. No clock: the same inputs always give the same cover."""
    fovs = []
    assigned = np.zeros(P.n, dtype=bool)
    for i in np.where(P.type == 2)[0]:
        fovs.append(FOV(P.cx[i], P.cy[i], [i], P.step[i], exclusive=True))
        assigned[i] = True
    for s in sorted(set(P.step.tolist())):
        grp = np.where((P.step == s) & (~assigned))[0]
        if len(grp) == 0:
            continue
        bigs = grp[P.big[grp]]
        if not two_phase_side:
            gc = GridCover(P, grp[~P.big[grp]], grid_res, weights)
            for i in sorted(bigs, key=lambda q: (-(P.w[q] * P.h[q]), q)):
                new = place_big_tiles(P, i, gc)
                for f in new:
                    assigned[f.members] = True
                fovs.extend(new)
            for x, y, mem in gc.greedy():
                fovs.append(FOV(x, y, mem, s))
                assigned[mem] = True
            continue
        for i in bigs:
            for (x, y) in P.tiles(i):
                f = FOV(x, y, [i], s, fixed=True)
                free = grp[(~P.big[grp]) & (~assigned[grp])]
                if len(free):
                    extra = free[P.contains(free, x, y)]
                    if len(extra):
                        f.members.extend(int(e) for e in extra)
                        assigned[extra] = True
                fovs.append(f)
            assigned[i] = True
        rest = grp[(~P.big[grp]) & (~assigned[grp])]
        new = cover_group(P, rest, grid_res, weights, True)
        for f in new:
            assigned[f.members] = True
        fovs.extend(new)
    fovs = remove_redundant(P, fovs)
    if cfg.get('eliminate', True):
        fovs = eliminate(P, fovs, chains=bool(cfg.get('elim_chains', True)))
    return recenter(P, fovs)


def _cap_proxy(P, fovs):
    """Capture proxy of a cover: the total imaging time of its FOVs (side parts take the longer side capture)."""
    c = 0.0
    for f in fovs:
        c += max(P.capture, P.side_capture) if any(P.side[m] == 1 for m in f.members) else P.capture
    return c


def _score(P, fovs, seq, cfg):
    if cfg.get('objective', 'ct') == 'sweep':
        return ct_sim.evaluate_sweep(P, fovs, seq)
    return ct_sim.evaluate(P, fovs, seq)


def _order_and_refine(P, fovs, cfg):
    """Visit order, then centre refinement and a second order on the refined centres; the refined result is kept
    only if it scores strictly lower (the centres are restored otherwise)."""
    seq, cm = build_sequence(P, fovs, cfg)
    if not cfg.get('refine', True):
        return seq, cm
    ct0 = _score(P, fovs, seq, cfg)
    saved = [(f.x, f.y) for f in fovs]
    refine_positions(P, fovs, seq)
    seq1, cm1 = build_sequence(P, fovs, cfg)
    refine_positions(P, fovs, seq1)
    if _score(P, fovs, seq1, cfg) < ct0 - 1e-9:
        return seq1, cm1
    for f, (x, y) in zip(fovs, saved):
        f.x, f.y = x, y
    return seq, cm


def quick_plan(P, fovs, deadline=None):
    """A fast valid plan of a cover (travel-only order, longest jobs first, verified and repaired), for an early
    write."""
    fv = [FOV(f.x, f.y, list(f.members), f.step, fixed=f.fixed, exclusive=f.exclusive) for f in fovs]
    seq, _ = build_sequence(P, fv, dict(order='tsp', deadline=deadline))
    for f in fv:
        f.members.sort(key=lambda m: (-P.time[m], m))
    for _ in range(3):
        issues = verify(P, fv, seq)
        if not issues:
            break
        fv, seq = repair(P, fv, seq, issues)
    return fv, seq


# Set-cover local search of the ordinary parts (setcover.py): candidate grid, budgets, gates.
LS_SNAP = 0.5               # mm: hx / hy values closer than this share a candidate coordinate
LS_ALIGNED = 0.5            # lattice boards: at least this share of ordinary parts on lattice lines (_aligned_frac)
LS_ITERS = 40000            # lattice boards: iteration budget of the second stage (deterministic, no clock)
LS_ITERS_CP = 10000         # lattice boards: the first stage, searched and finished first
LS_MAX_NNZ = 4000000        # work / memory bound: part-centre incidences (~0.1 s per 1e6 on the dev lattices)
LS_MAX_COLUMNS = 200000     # memory bound on the candidate grid
LS_ITERS_CAMERA = 2000      # camera-bound boards: iteration budget
LS_PREFINISH_MARGIN = 0.05  # camera-bound boards: the alternative must lead by this much before it is finished
LS_ITERS_CAMERA_LONG = 10000  # camera-bound boards: a second stage on large problems (continuing the first run)
LS_CAMERA_LONG_PARTS = 700  # ... when the local-search groups hold at least this many ordinary parts
LS_MODES = (False, True)    # camera-bound boards: the full incidence first (keeps ties), then the dominance-reduced one
LS_SECOND_STRICT = True     # a later mode's cover is finished only if its capture proxy is strictly lower
PY_ITERS_SCALE = 0.05       # without numba (setcover.PY: plain-Python kernels, ~0.1 ms per iteration) every budget
PY_CAMERA = False           # above is scaled by this, and the camera-bound search stays off (no gain found there)
PY_MAX_NNZ = 1000000        # ... and lattice boards are searched only up to this incidence size (the tested range;
                            # the Python search time grows ~1 s per 1e6 incidences)
if setcover.PY:  # never with numba: the compiled budgets above stay as they are
    LS_ITERS, LS_ITERS_CP, LS_ITERS_CAMERA, LS_ITERS_CAMERA_LONG = (
        max(1, int(round(v * PY_ITERS_SCALE))) for v in (LS_ITERS, LS_ITERS_CP, LS_ITERS_CAMERA, LS_ITERS_CAMERA_LONG))


def _aligned_frac(P, idxs):
    """Share of ordinary parts whose centre lies on a lattice line. Per axis (cx, then cy) and step: sort the values; a
    gap > LS_SNAP starts a new cluster; a part is aligned when its cluster has >= 3 members and spans <= LS_SNAP.
    Larger axis share. Centres, not the feasible-box edges hx / hy: a side part's box edge is shifted by a quarter FOV,
    which can merge two lattice columns of hx into one wide cluster."""
    n = sum(len(idx) for _, idx in idxs)
    best = 0.0
    for vals in (P.cx, P.cy):
        tight = 0
        for _, idx in idxs:
            v = np.sort(vals[idx])
            cuts = np.concatenate(([0], np.flatnonzero(np.diff(v) > LS_SNAP) + 1, [len(v)]))
            size, span = np.diff(cuts), v[cuts[1:] - 1] - v[cuts[:-1]]
            tight += int(size[(size >= 3) & (span <= LS_SNAP)].sum())
        best = max(best, tight / n)
    return best


def _ls_groups(P, fixed_members, lattice=True):
    """Ordinary parts per step (not fiducial, not big, not already inside a kept tile) and their snapped, undominated
    axes, with the candidate-grid size and incidence count; ([], 0, 0) when the parts do not sit on a lattice."""
    assigned = np.zeros(P.n, dtype=bool)
    assigned[fixed_members] = True
    idxs = []
    for s in sorted(set(P.step.tolist())):
        idx = np.where((P.step == s) & (~assigned) & (~P.big))[0]
        if len(idx):
            idxs.append((int(s), idx))
    if not idxs or (lattice and _aligned_frac(P, idxs) < LS_ALIGNED):
        return [], 0, 0
    groups, columns, nnz = [], 0, 0
    for s, idx in idxs:
        xs = setcover.undominated(setcover.reps(P.hx[idx], LS_SNAP), P.lx[idx], P.hx[idx], TOL)
        ys = setcover.undominated(setcover.reps(P.hy[idx], LS_SNAP), P.ly[idx], P.hy[idx], TOL)
        nx = np.searchsorted(xs, P.hx[idx] + TOL, 'right') - np.searchsorted(xs, P.lx[idx] - TOL, 'left')
        ny = np.searchsorted(ys, P.hy[idx] + TOL, 'right') - np.searchsorted(ys, P.ly[idx] - TOL, 'left')
        groups.append((s, idx, xs, ys))
        columns += len(xs) * len(ys)
        nnz += int((np.maximum(nx, 0).astype(np.int64) * np.maximum(ny, 0)).sum())
    return groups, columns, nnz


def _same_group(a, b):
    """Cache key equality: (step, part indices, x axis, y axis, start-cover centres) all equal, so the incidence and
    the start state are the same (the start cover alone does not pin the incidence)."""
    return a[0] == b[0] and all(np.array_equal(p, q) for p, q in zip(a[1:], b[1:]))


def _rwls_resumed(runs, gi, key, inc, init, iters):
    """The rwls answer for this group and budget, continuing the run kept in `runs` from an earlier (smaller or
    equal) budget on the same group. Any failure drops the kept run and answers with a fresh rwls (same result)."""
    kept = runs.get(gi)
    try:
        if kept is None or not _same_group(kept[0], key) or kept[1].done > iters:
            kept = (key, setcover.RWLSRun(*inc, init, 0))
            runs[gi] = kept
        return kept[1].run(iters)
    except Exception:
        runs.pop(gi, None)
        return setcover.rwls(*inc, init, iters, 0)


def _ls_cover(P, base, cfg, groups, iters=None, runs=None, dominance=True, inc_cache=None):
    """Local-search cover: keep the fiducial and big-tile FOVs of `base`, re-cover the ordinary parts of each step with
    the row-weighting local search started from base's own ordinary FOV centres; one FOV per selected centre.
    dominance: search the dominance-reduced incidence (True) or the full one (False). With `runs` (a dict kept across
    calls on the same base), a larger budget continues the earlier search; `inc_cache` keeps one incidence per group."""
    out = [FOV(f.x, f.y, list(f.members), f.step, fixed=f.fixed, exclusive=f.exclusive)
           for f in base if f.fixed or f.exclusive]
    for gi, (s, idx, xs, ys) in enumerate(groups):
        own = np.array([(f.x, f.y) for f in base if not (f.fixed or f.exclusive) and f.step == s],
                       dtype=np.float64).reshape(-1, 2)
        key = (s, idx, xs, ys, own)
        store = inc_cache if inc_cache is not None else runs
        cached = store.get(('inc', gi)) if store is not None else None
        if cached is not None and _same_group(cached[0], key):
            e_ptr, e_col, c_ptr, c_el, nc, red = cached[1]  # same parts, axes and start cover: same incidence
        else:
            e_ptr, e_col, c_ptr, c_el, nc = setcover.build_incidence(P.lx[idx], P.hx[idx], P.ly[idx], P.hy[idx],
                                                                     xs, ys, own, TOL)
            red = None
        use_dom = bool(dominance)
        if use_dom and red is None:
            try:  # any failure keeps the full incidence
                red = setcover.reduce_columns(e_ptr, e_col, c_ptr, c_el, P.lx[idx], P.hx[idx], P.ly[idx],
                                              P.hy[idx], xs, ys, TOL)
            except Exception:
                red = None
        if store is not None:
            store[('inc', gi)] = (key, (e_ptr, e_col, c_ptr, c_el, nc, red))
        init = np.arange(nc - len(own), nc, dtype=np.int64)
        budget = LS_ITERS if iters is None else iters
        inc, init_s, rkey = (e_ptr, e_col, c_ptr, c_el), init, gi
        reduced = use_dom and red is not None
        if reduced:
            inc, init_s, rkey = red[:4], red[5][init], ('red', gi)  # the reduced search keeps its own resumed run
        if runs is not None:
            sel = _rwls_resumed(runs, rkey, key, inc, init_s, budget)
        else:
            sel = setcover.rwls(*inc, init_s, budget, 0)
        if reduced and len(sel):
            sel = red[4][sel]  # back to full column ids (kept order is increasing, so still sorted)
        ng = len(xs) * len(ys)
        pts = np.empty((len(sel), 2), dtype=np.float64)
        rank = np.full(nc, -1, dtype=np.int64)
        for k, j in enumerate(sel):
            j = int(j)
            rank[j] = k
            pts[k] = (xs[j // len(ys)], ys[j % len(ys)]) if j < ng else own[j - ng]
        col = setcover.assign(e_ptr, e_col, rank, pts, np.ascontiguousarray(P.cx[idx], dtype=np.float64),
                              np.ascontiguousarray(P.cy[idx], dtype=np.float64))
        if len(col) and col.min() < 0:
            raise ValueError('local-search cover left a part uncovered')  # isolated in _plan_ls: the first plan stays
        members = [[] for _ in range(len(sel))]
        for e in range(len(idx)):
            members[int(rank[col[e]])].append(int(idx[e]))
        pts = [(float(a), float(b)) for a, b in pts]
        for (x, y), m in zip(pts, members):
            if m:
                out.append(FOV(x, y, sorted(m), s))
    return recenter(P, out)


def _finish(P, cfg, fovs, seq, t_start, budget, over):
    """The finishing stages after the cover and order choice: critical-FOV relocation, heavy-part reassignment, order
    annealing, member order inside FOVs (LPT), release relocation, then verification and repair. Each stage is skipped
    past its share of the time budget. Returns (fovs, seq, number of FOVs the repair added)."""
    if cfg.get('crit_relocate', True) and not over(0.8):
        cm_, t_img, t_recon, t_comps, steps = ct_sim.plan_times(P, fovs, list(range(len(fovs))))
        seq = critical_relocate(P, fovs, seq, cm_, t_img, t_comps, steps,
                                max_rounds=int(cfg.get('crit_rounds', 8)), max_evals=int(cfg.get('crit_evals', 24)),
                                objective=cfg.get('objective', 'ct'))

    if cfg.get('reassign', True) and not over(0.85):
        fovs, seq = reassign_heavy(P, fovs, seq, sweep_obj=cfg.get('objective', 'ct') == 'sweep',
                                   deadline=t_start + 0.9 * budget, top=int(cfg.get('reassign_top', 40)))

    if cfg.get('anneal', True) and not over(0.85):
        for f in fovs:
            f.members.sort(key=lambda m: (-P.time[m], m))
        cm_, t_img, t_recon, t_comps, steps = ct_sim.plan_times(P, fovs, list(range(len(fovs))))
        seq = anneal_order(P, fovs, seq, cm_, t_img, t_comps, steps, budget_ms=float(cfg.get('anneal_ms', 350.0)),
                           min_tail=float(cfg.get('anneal_min_tail', 2.0)), scale=cfg.get('anneal_scale', 'gap'),
                           deadline=t_start + 0.9 * budget)

    lpt = cfg.get('lpt', 'auto')
    if lpt == 'auto' and over(0.9):
        lpt = True
    if lpt == 'auto':
        best_key, best_ct = None, None
        for key in (lambda m: (-P.time[m], m), lambda m: (P.time[m], m), lambda m: m):
            for f in fovs:
                f.members.sort(key=key)
            ct = _score(P, fovs, seq, cfg)
            if best_ct is None or ct < best_ct - 1e-9:
                best_key, best_ct = key, ct
        for f in fovs:
            f.members.sort(key=best_key)
    elif lpt:
        for f in fovs:
            f.members.sort(key=lambda m: (-P.time[m], m))

    if cfg.get('objective', 'ct') == 'sweep' and not over(0.95):
        cm_, img_, rec_, jobs_, steps_ = ct_sim.plan_times(P, fovs, seq)
        types_ = [fovs[i].type_of(P) for i in seq]
        released = improve_release_sequence(cm_, img_, rec_, jobs_, types_, steps_, P.max_core,
                    deadline=t_start + 0.98 * budget, max_rounds=8)
        seq = [seq[i] for i in released['sequence']]
        LAST_STATS['release_rounds'] = len(released['prefixes'])
        LAST_STATS['release_evaluations'] = released['fixed_evaluations']
        LAST_STATS['release_stop'] = released['stop_reason']

    repaired = 0
    if cfg.get('verify', True):
        for _ in range(3):
            issues = verify(P, fovs, seq)
            if not issues:
                break
            n_before = len(seq)
            fovs, seq = repair(P, fovs, seq, issues)
            repaired += len(seq) - n_before
    return fovs, seq, repaired

def _camera_bound(P, fovs, seq):
    """Is the finished plan camera-bound at the checker's k*? Camera finish time (core 0) of the plan vs the inspection
    floor total load / (k* - 3), both from the same core sweep the score uses."""
    cm_, t_img_, t_recon_, t_comps_, steps_ = ct_sim.plan_times(P, fovs, seq)
    _, k_star, _, core0_end = ct_sim.sweep(cm_, t_img_, t_recon_, t_comps_, P.max_core, steps_)
    load = sum(sum(c) for c in t_comps_)
    return core0_end >= load / max(k_star - 3, 1)


def _plan_ls(P, cfg, fv0, groups, t_start, budget, over, early_write=None, camera_gate=False):
    """Finish the greedy cover first (as plan() does) and write it, then build the local-search cover and finish it
    too; the second plan replaces the first only if it verifies clean and its exact score is strictly lower. Everything
    after the first plan is finished is isolated: any exception keeps the first plan.
    Lattice boards: two stages (LS_ITERS_CP, then LS_ITERS continuing the same run), the full incidence only.
    camera_gate (non-lattice boards): the alternative is searched only if the finished plan is camera-bound (decided
    before the candidate grid is built); full then dominance-reduced incidence per stage; an alternative is finished
    only if, ordered and refined, it leads the greedy plan before its finishing stages by LS_PREFINISH_MARGIN."""
    snapshot = [FOV(f.x, f.y, list(f.members), f.step, fixed=f.fixed, exclusive=f.exclusive) for f in fv0]
    for f in fv0:
        f.members.sort(key=lambda m: (-P.time[m], m))
    sq, _ = _order_and_refine(P, fv0, cfg if not over(0.5) else dict(cfg, refine=False))
    pre = None
    if camera_gate:
        try:  # the greedy plan before its finishing stages, for a like-with-like comparison
            pre = ([FOV(f.x, f.y, list(f.members), f.step, fixed=f.fixed, exclusive=f.exclusive) for f in fv0],
                   list(sq))
        except Exception:
            pre = None  # no copy: the camera branch is skipped below (the finished greedy plan stays)
    fovs, seq, repaired = _finish(P, cfg, fv0, sq, t_start, budget, over)
    stats = lambda: {k: LAST_STATS.get(k) for k in ('release_rounds', 'release_evaluations', 'release_stop')}
    best = (None, fovs, seq, repaired, stats())
    LAST_STATS['ls_used'] = False
    try:
        go = True
        if camera_gate:
            go = pre is not None and _camera_bound(P, fovs, seq)
            if go:
                groups, columns, nnz = _ls_groups(P, [m for f in snapshot if f.fixed or f.exclusive for m in f.members],
                                                  lattice=False)
                go = bool(groups) and columns <= LS_MAX_COLUMNS and nnz <= LS_MAX_NNZ
            LAST_STATS['ls_camera_bound'] = go
        if go and early_write is not None:
            early_write(fovs, seq)  # anytime: the finished greedy plan is on disk while the alternative is searched
        if go:
            best = (_score(P, fovs, seq, cfg),) + best[1:]
        stages = [LS_ITERS_CAMERA] if camera_gate else [LS_ITERS_CP, LS_ITERS]
        if camera_gate and go and sum(len(g[1]) for g in groups) >= LS_CAMERA_LONG_PARTS:
            stages.append(LS_ITERS_CAMERA_LONG)  # a longer search pays on large problems only
        firsts = {}
        # A later stage continues the earlier stage's search. A single stage runs plain rwls (the same answer and the
        # same code path, so small camera-bound boards keep the single-run timing).
        runs = {} if len(stages) > 1 else None
        inc_cache = {}  # one incidence per group, shared by both modes
        LAST_STATS['ls_pick'] = None
        for si, iters_s in enumerate(stages):
            if not (go and not over(float(cfg.get('portfolio_final_frac', 0.45)))):
                break
            tried = []  # covers met in this stage (finished now, or unchanged from the first stage)
            stage_cap = None
            fresh = False
            modes = LS_MODES if camera_gate else LS_MODES[:1]  # the full-incidence cover first (it keeps ties), then
            # the reduced-incidence cover; lattice boards keep the single full-incidence search
            for mi, mode in enumerate(modes):
                if over(float(cfg.get('portfolio_final_frac', 0.45))):
                    break
                alt = _ls_cover(P, snapshot, cfg, groups, iters_s, runs, dominance=mode, inc_cache=inc_cache)
                key = [(f.x, f.y, tuple(f.members), f.step) for f in alt]
                if key in tried:
                    continue  # both modes found the same cover this stage
                cap = _cap_proxy(P, alt)
                later, smaller = bool(tried), stage_cap is None or cap < stage_cap - 1e-9
                stage_cap = cap if stage_cap is None else min(stage_cap, cap)
                tried.append(key)
                if si > 0 and key == firsts.get(mode):
                    continue  # this mode's longer search ended on its first-stage cover (finished then)
                if si == 0:
                    firsts[mode] = key
                if LS_SECOND_STRICT and later and not smaller:
                    continue  # not a smaller cover than one already met in this stage
                fresh = True
                LAST_STATS['ls_stages'] = si + 1
                if _cap_proxy(P, alt) < _cap_proxy(P, snapshot) - 1e-9:
                    for f in alt:
                        f.members.sort(key=lambda m: (-P.time[m], m))
                    sq2, _ = _order_and_refine(P, alt, cfg if not over(0.5) else dict(cfg, refine=False))
                    ahead = pre is None or _score(P, alt, sq2, cfg) < _score(P, pre[0], pre[1], cfg) - LS_PREFINISH_MARGIN
                    if camera_gate:
                        LAST_STATS['ls_prefinish_ahead'] = ahead
                    if ahead:
                        fovs2, seq2, repaired2 = _finish(P, cfg, alt, sq2, t_start, budget, over)
                        if not verify(P, fovs2, seq2):
                            score2 = _score(P, fovs2, seq2, cfg)
                            if score2 < best[0] - 1e-9:
                                best = (score2, fovs2, seq2, repaired2, stats())
                                LAST_STATS['ls_used'] = True
                                LAST_STATS['ls_pick'] = 'reduced' if mode else 'full'
                                if early_write is not None and (si + 1 < len(stages) or mi + 1 < len(modes)):
                                    early_write(fovs2, seq2)  # on disk before any further search
            if not fresh:
                break  # no mode brought a new cover
    except Exception:
        pass
    _, fovs, seq, repaired, st = best
    LAST_STATS.update(st)
    LAST_STATS['repaired'] = repaired
    return fovs, seq


def _early_ok(cfg):
    """The early-dispatch rule: spawned pool active, board folder known, portfolio_max 1, and every base worker idle
    (a partial send would change the candidates)."""
    variants = COVER_VARIANTS if cfg.get('cover_portfolio', True) else COVER_VARIANTS[:1]
    return bool(len(variants) > 1 and int(cfg.get('portfolio_max', 1)) == 1
                and _CTX.get('folder') is not None and parallel.STATE.get('active') and parallel.STATE.get('spawn')
                and sum(1 for w in parallel.STATE.get('workers', ()) if not w.get('extra') and not w.get('busy'))
                >= len(variants) - 1)


def _entry_dispatch(cfg):
    """The early dispatch, done in solve() right after the main has read the board (a board the main cannot read
    never reaches the workers) and before the trivial plan and its write, so the workers start as soon as possible.
    Spawned workers get only the variant number and the folder and build the cover themselves: no Problem is sent.
    plan() then builds variant 0 (and any variant not taken)."""
    if not _early_ok(cfg):
        return
    variants = COVER_VARIANTS if cfg.get('cover_portfolio', True) else COVER_VARIANTS[:1]
    kw = {'extras': _extra_members()} if parallel.STATE.get('extras') else {}
    _CTX['gen'], _CTX['sent'] = parallel.dispatch(None, [(v, None) for v in range(1, len(variants))], _CTX['config'],
                                                  folder=_CTX['folder'], **kw)
    _CTX['t_dispatch'] = time.perf_counter()
    _CTX['entry'] = True


def plan(P, cfg, t_start=None, early_write=None, prebuilt=None):
    """The main plan of one board (see the module docstring, step 3). prebuilt: a worker finishes a cover it was
    given or built itself, with no dispatch of its own. Returns (fovs, seq)."""
    t_start = time.perf_counter() if t_start is None else t_start
    budget = float(cfg.get('time_budget', 3.0))
    cfg = dict(cfg, deadline=t_start + 0.7 * budget)

    def over(frac):
        return time.perf_counter() - t_start > frac * budget

    variants = COVER_VARIANTS if cfg.get('cover_portfolio', True) else COVER_VARIANTS[:1]
    built = []
    order = list(range(len(variants)))  # the variant numbers the main process builds, in this order
    early = False
    if prebuilt is not None:  # a worker finishes a cover (no rebuild)
        order = []
        built.append((_cap_proxy(P, prebuilt), 0, prebuilt))
    elif _CTX.get('entry'):  # the variants already went out at solve() entry (_entry_dispatch)
        early = _CTX['gen'] is not None  # a stopped pool returned (None, []): the main builds all four
        if early:
            taken = {v for v, _ in _CTX['sent']}
            order = [0] + [v for v in range(1, len(variants)) if v not in taken]
    _CTX['early'] = early
    for vi in order:
        v = variants[vi]
        if built and over(0.25):
            break
        fv = build_fovs(P, cfg, v['weights'], v['grid_res'], v['two_phase_side'])
        built.append((_cap_proxy(P, fv), vi, fv))
        if early_write is not None and len(built) == 1:
            early_write(*quick_plan(P, fv, deadline=t_start + 0.3 * budget))
    built.sort(key=lambda t: (t[0], t[1]))
    if early:  # the cover proxies of what the main built (variant 0, and any not taken), for the targeted wait
        _CTX['proxies_main'] = {b[1]: b[0] for b in built}
    if not early and _CTX.get('folder') is not None and len(built) > 1:  # the other built covers to the workers
        # the extra members only while extra workers are up
        kw = {'extras': _extra_members()} if parallel.STATE.get('extras') else {}
        _CTX['gen'], _CTX['sent'] = parallel.dispatch(P, [(b[1], b[2]) for b in built[1:]], _CTX['config'],
                                                      folder=_CTX['folder'], **kw)  # spawned workers read the board
        _CTX['t_dispatch'] = time.perf_counter()
    keep = [b for b in built if b[0] <= built[0][0] + cfg.get('portfolio_slack', 0.5) + 1e-9]
    keep = keep[:int(cfg.get('portfolio_max', 1))]
    if cfg.get('ls_cover', True) and setcover.ENABLED and len(keep) == 1:
        try:
            groups, columns, nnz = _ls_groups(P, [m for f in keep[0][2] if f.fixed or f.exclusive for m in f.members])
        except Exception:
            groups, columns, nnz = [], 0, 0
        if groups and columns <= LS_MAX_COLUMNS and nnz <= LS_MAX_NNZ and (not setcover.PY or nnz <= PY_MAX_NNZ):
            LAST_STATS['lattice'] = True  # the order polish skips this route
            return _plan_ls(P, cfg, keep[0][2], groups, t_start, budget, over, early_write)
        if not groups and cfg.get('ls_camera', True) and (PY_CAMERA or not setcover.PY):  # camera-gated alternative
            return _plan_ls(P, cfg, keep[0][2], None, t_start, budget, over, early_write, camera_gate=True)
    best = None
    for _, _, fv in keep:
        if best is not None and over(0.6):
            break
        for f in fv:
            f.members.sort(key=lambda m: (-P.time[m], m))
        sq, cmm = _order_and_refine(P, fv, cfg if not over(0.5) else dict(cfg, refine=False))
        ct = _score(P, fv, sq, cfg)
        if best is None or ct < best[0] - 1e-9:
            best = (ct, fv, sq, cmm)
    _, fovs, seq, _ = best
    fovs, seq, repaired = _finish(P, cfg, fovs, seq, t_start, budget, over)
    LAST_STATS['repaired'] = repaired
    return fovs, seq


def _csv_bytes(fovs, seq):
    """The bytes pandas' DataFrame(rows).to_csv(index=True) writes, without building a DataFrame. Only for finite float
    coordinates (a float64 column, formatted by ndarray.astype(str) as pandas does; pandas writes NaN as an empty
    field); None otherwise, and the caller uses pandas."""
    xs = [fovs[k].x for k in seq]
    ys = [fovs[k].y for k in seq]
    if not xs or not all(isinstance(v, float) and math.isfinite(v) for v in xs) \
            or not all(isinstance(v, float) and math.isfinite(v) for v in ys):
        return None
    sx = np.asarray(xs, dtype=np.float64).astype(str)
    sy = np.asarray(ys, dtype=np.float64).astype(str)
    lines = [',x,y,comp_idx']
    for i, k in enumerate(seq):
        c = json.dumps([int(m) for m in fovs[k].members])
        if ',' in c:  # csv.QUOTE_MINIMAL: only a field holding the delimiter is quoted here
            c = '"' + c + '"'
        lines.append('%d,%s,%s,%s' % (i, sx[i], sy[i], c))
    return (os.linesep.join(lines) + os.linesep).encode('utf-8')


def write_output(data_folder, fovs, seq):
    """Atomic: the CSV is complete on disk at every instant (temp file + os.replace). pandas writes the file only when
    the direct writer cannot (non-finite coordinates, or any error)."""
    final = os.path.join(data_folder, 'output_fov.csv')
    tmp = final + '.tmp'
    try:
        data = _csv_bytes(fovs, seq)
    except Exception:
        data = None
    if data is not None:
        with open(tmp, 'wb') as fh:
            fh.write(data)
    else:
        rows = [{'x': fovs[k].x, 'y': fovs[k].y, 'comp_idx': json.dumps([int(m) for m in fovs[k].members])}
                for k in seq]
        pd.DataFrame(rows, columns=['x', 'y', 'comp_idx']).to_csv(tmp, index=True)
    replace_output(tmp, final, LAST_STATS)
    return len(seq)


def trivial_plan(P):
    """Always-valid fallback: one FOV per component (tiles for big ones), fiducials first."""
    fovs = []
    for i in sorted(range(P.n), key=lambda q: (-P.type[q], q)):
        if P.big[i]:
            fovs.extend(FOV(x, y, [i], P.step[i], fixed=True) for (x, y) in P.tiles(i))
        else:
            fovs.append(FOV(P.cx[i], P.cy[i], [i], P.step[i], exclusive=P.type[i] == 2))
    return fovs, list(range(len(fovs)))


def _polish(P, cfg, fovs, seq, t_start, early=None):
    """The polishes of the finished plan: the order polish (not on lattice boards, where it found little at the
    highest cost per evaluation), then the member polish. Each result is kept only if verify() finds no issue (the
    member polish also needs a strictly lower score); past 95% of the time budget, without the compiled evaluator, or on
    any exception, plan()'s plan stays. The finished plan is written first (early), so a stop during the polish leaves it
    on disk. Returns (fovs, seq)."""
    budget = float(cfg.get('time_budget', 3.0))
    if not cfg.get('polish', True) or time.perf_counter() - t_start > 0.95 * budget:
        return fovs, seq
    if not (ct_sim._acc._NUMBA and ct_sim._acc._SEQ):  # no compiled evaluator: both polishes would return as is
        return fovs, seq
    order_on = not (LAST_STATS.get('lattice') and not cfg.get('polish_lattice', False))
    members_on = bool(cfg.get('polish_members', True))
    if not (order_on or members_on):
        return fovs, seq
    try:
        if early is not None:
            early(fovs, seq)  # the finished plan is on disk before the polish
        cm_, t_img, _, t_comps, steps = ct_sim.plan_times(P, fovs, list(range(len(fovs))))
        if order_on:
            new = polish_order(P, fovs, seq, cm_, t_img, t_comps, steps, stall=int(cfg.get('polish_stall', 1000)),
                               cap=int(cfg.get('polish_evals', 8000)), deadline=t_start + 0.98 * budget,
                               nb_scale=int(cfg.get('polish_nb_scale', 4)))
            if new != seq and not verify(P, fovs, new):
                LAST_STATS['polished'] = True
                seq = new
        if members_on and time.perf_counter() - t_start <= 0.95 * budget:
            got = polish_members(P, fovs, seq, cm_, t_img, t_comps, steps, stall=int(cfg.get('members_stall', 250)),
                                 cap=int(cfg.get('members_evals', 2000)), deadline=t_start + 0.98 * budget,
                                 nb_scale=int(cfg.get('polish_nb_scale', 4)))
            if got is not None:
                mem, order = got
                keep = [q for q in range(len(fovs)) if mem[q]]
                remap = {old: new for new, old in enumerate(keep)}
                fv2 = [FOV(fovs[q].x, fovs[q].y, mem[q], fovs[q].step, fixed=fovs[q].fixed, exclusive=fovs[q].exclusive)
                       for q in keep]
                sq2 = [remap[q] for q in order]
                # tables re-derived from scratch: the new plan must verify and be strictly better
                if not verify(P, fv2, sq2) and _score(P, fv2, sq2, cfg) < _score(P, fovs, seq, cfg) - 1e-9:
                    LAST_STATS['members_polished'] = True
                    fovs, seq = fv2, sq2
    except Exception:
        pass
    LAST_STATS['polish_nb'] = bool(polish_nb.OK)
    return fovs, seq


def _portfolio_pick(P, cfg, fovs, seq, t0):
    """Collect the workers' finished plans for this board (until the main plan's end + PORTFOLIO_WAIT, never past
    PORTFOLIO_CAP of the budget; on an early-dispatched board the variant the main would have kept gets one longer wait),
    and return the plan with the lowest exact score; the main plan wins ties. A reply is re-checked (verify, exact
    score) only if its worker's score could still win; the worker scored it with the same evaluate_sweep on the same
    board, so an unchecked plan is never picked. Records what happened in LAST_STATS['portfolio']."""
    sent = _CTX.get('sent') or []
    if not sent:
        return fovs, seq
    budget = float(cfg.get('time_budget', 3.0))
    t_main = time.perf_counter()
    main_score = float(ct_sim.evaluate_sweep(P, fovs, seq))  # before the wait
    proxies, ext, rec_ext = {}, None, {}
    if _CTX.get('early'):
        def ext(pending, prox):
            # the base variant the main would have kept (lowest cover proxy, then lowest index, among the variants the
            # main and the workers built) gets until the main plan's end + PORTFOLIO_WAIT_TARGET when it has not replied
            # by the usual deadline (missing it cost P on slow workers); no other wait changes
            known = {v: p for v, p in prox.items() if v < len(COVER_VARIANTS)}
            known.update(_CTX.get('proxies_main') or {})
            if not known:
                return None
            target = min(known, key=lambda v: (known[v], v))
            rec_ext['target'] = target
            if target in pending:
                rec_ext['extended'] = True
                return min(t_main + PORTFOLIO_WAIT_TARGET, t0 + PORTFOLIO_CAP * budget)
            return None
    kw = {'proxies': proxies, 'extend': ext} if ext is not None else {}  # not early: the plain wait
    got = parallel.collect(_CTX['gen'], sent, min(t_main + PORTFOLIO_WAIT, t0 + PORTFOLIO_CAP * budget), **kw)
    t_disp = _CTX.get('t_dispatch', t0)
    missing = len(got) < len(sent)
    cpu = sorted(g[6] for g in got)[len(got) // 2] if got else None
    parallel.note_board(cpu, missing)  # the runtime CPU guard: pauses the pool when the workers get too little CPU
    best = (main_score, None, fovs, seq)
    rec = {'sent': [v for v, _ in sent], 'main_score': best[0], 'main_done': t_main - t0,
           't_dispatch': t_disp - t0, 'worker_cpu': cpu, 'replies': [], 'early': bool(_CTX.get('early'))}
    rec.update(target=rec_ext.get('target'), extended=bool(rec_ext.get('extended')), proxies=dict(proxies))
    rank = {v: r for r, (v, _) in enumerate(sent)}
    for v, plan_, sq, score, rt, arrival, _cpu in sorted(got, key=lambda g: rank[g[0]]):
        if not score < best[0] - 1e-9:
            # the worker scored its plan with the same evaluate_sweep on the same board, so this reply cannot be
            # picked; not re-checking it never admits an unchecked plan (a reply that could win is re-checked)
            rec['replies'].append({'v': v, 'score': score, 'checked': False, 'worker_score': score, 'worker_rt': rt,
                                   'arrival': arrival - t0, 'cpu': _cpu})
            continue
        fv = [FOV(x, y, list(m), st, fixed=fx, exclusive=ex) for x, y, m, st, fx, ex in plan_]
        ok = not verify(P, fv, sq)
        s = float(ct_sim.evaluate_sweep(P, fv, sq)) if ok else None
        rec['replies'].append({'v': v, 'score': s, 'checked': True, 'worker_score': score, 'worker_rt': rt,
                               'arrival': arrival - t0, 'cpu': _cpu})
        if ok and s < best[0] - 1e-9:
            best = (s, v, fv, sq)
    rec['picked'] = best[1]
    LAST_STATS['portfolio'] = rec
    return best[2], best[3]


def _recover(P, cfg, fovs, seq, t0):
    """The pair re-cover stage (recover.recover()) on the final plan; its plan replaces this one only if verify() finds
    no issue and the exact score is strictly lower. Off when the kernels are missing, outside the main process, or past
    90% of the time budget. Records LAST_STATS['recover']."""
    LAST_STATS.pop('recover', None)
    if not cfg.get('recover', True) or recover is None or not recover.OK:
        return fovs, seq
    budget = float(cfg.get('time_budget', 3.0))
    if time.perf_counter() - t0 > 0.9 * budget:
        return fovs, seq
    fv2, sq2, st = recover.recover(P, fovs, seq, cfg, t0, budget)
    LAST_STATS['recover'] = st
    if fv2 is fovs:
        return fovs, seq
    if verify(P, fv2, sq2):
        st['rejected'] = 'verify'
        return fovs, seq
    new, old = _score(P, fv2, sq2, cfg), _score(P, fovs, seq, cfg)
    st['exact_new'], st['exact_old'] = float(new), float(old)
    if new < old - 1e-9:
        st['kept'] = True
        return fv2, sq2
    st['rejected'] = 'score'
    return fovs, seq


def solve(data_folder: str, config=None) -> int:
    """Solve the board in data_folder and write data_folder/output_fov.csv; returns the number of FOVs written."""
    t0 = time.perf_counter()
    LAST_STATS.update(io_retries=0, io_retry_wait_s=0., io_errors=[], polished=False, lattice=False,
                      members_polished=False, write_errors=0)
    cfg = get_config(config)
    parallel.tick()  # late replies of the last board, worker checks, a paused pool's board count
    LAST_STATS.pop('portfolio', None)  # no record of an earlier board survives into this one
    _CTX.clear()
    _CTX.update(folder=data_folder, config=config, gen=None, sent=[])
    comp = pd.read_csv(os.path.join(data_folder, 'input_component.csv'))
    size = pd.read_csv(os.path.join(data_folder, 'input_size.csv'))
    param = pd.read_csv(os.path.join(data_folder, 'input_parameter.csv'))
    P = Problem(comp, size, param)
    _entry_dispatch(cfg)  # the workers start once the main has read the board, before the trivial plan's write
    on_disk = []  # FOV count of the plan this call wrote last, i.e. of the file on disk (empty: nothing written yet)

    def write(fv, sq, keep=False):
        """write_output(), noting the FOV count on disk. keep (the main plan's write, the one no handler encloses): once
        this call has a plan on disk, a failed write leaves that file (the replace is atomic) and returns its count
        instead of raising, so a write error after a valid output never ends the run. Other writes raise as before."""
        try:
            on_disk[:] = [write_output(data_folder, fv, sq)]
        except Exception:
            if not (keep and on_disk):
                raise
            LAST_STATS['write_errors'] += 1
        return on_disk[0]

    early = write if cfg.get('early_write', True) else None
    if early is not None:
        try:
            early(*trivial_plan(P))
        except Exception:
            _CTX.clear()  # no dispatch state outlives a failed solve()
            raise
    try:
        fovs, seq = plan(P, cfg, t_start=t0, early_write=early)
        fovs, seq = _polish(P, cfg, fovs, seq, t0, early)
    except Exception:
        try:
            fovs, seq = quick_plan(P, build_fovs(P, cfg, **COVER_VARIANTS[0]), deadline=t0 + cfg.get('time_budget', 3.0))
        except Exception:
            fovs, seq = trivial_plan(P)
        LAST_STATS['fallback'] = True
    n = write(fovs, seq, keep=True)
    try:  # a strictly better finished plan from a worker replaces the main plan
        fv2, sq2 = _portfolio_pick(P, cfg, fovs, seq, t0)
        if fv2 is not fovs:
            fovs, seq = fv2, sq2
            n = write(fovs, seq)
    except Exception:
        LAST_STATS['portfolio_error'] = True
    try:  # pair re-cover of a camera-bound final plan, kept only if it verifies and scores strictly lower
        fv3, sq3 = _recover(P, cfg, fovs, seq, t0)
        if fv3 is not fovs:
            fovs, seq = fv3, sq3
            n = write(fovs, seq)
    except Exception:
        LAST_STATS['recover_error'] = True
    _CTX.clear()
    LAST_STATS['n_fov'] = n
    LAST_STATS['t'] = time.perf_counter() - t0
    if cfg.get('debug'):
        print(f'[ARGOS] {os.path.basename(data_folder)}: fovs={n} ct_sim={ct_sim.evaluate(P, fovs, seq):.2f} t={LAST_STATS["t"]:.3f}s')
    return n


try:  # the worker pool starts at import, outside the scored solve() time (official Q&A #4)
    parallel.start()
except Exception:
    pass
gc.collect()  # the import's garbage (kernel compilation, the pool start) is collected here, outside solve()
