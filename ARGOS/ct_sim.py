"""Bit-exact re-implementation of utils.checker.Checker.get_real_cost (pipeline cycle time) and of the checker's core
sweep (score = min over k = 4..max_core of CT(k) + 10 k).

Pipeline: core 0 moves and captures (travel time max(Tx, Ty) on trapezoidal profiles, +5 s between steps); every other
core is bumped to the capture end; reconstruction runs on core 1 or 2 (lower load, lowest index on ties); the FOV's
inspection jobs go, in comp_idx order, to the least-loaded inspection core (3 .. k-1, availability at least the
reconstruction end, first index on ties). CT is the largest core time.
simulate() is the compiled kernel (acceleration.py) with _reference_simulate (the Python loop below) as its fallback;
both give the same floats. SeqEval scores visit orders straight from the full per-FOV tables."""
import numpy as np


def axis_time(d, v, a):
    """Trapezoidal/triangular travel time for distance d (array or scalar)."""
    d = np.asarray(d, dtype=float)
    th = v * v / a
    over = 2.0 * v / a + (d - th) / v
    under = 2.0 * np.sqrt(d / a)
    return np.where(d >= th, over, under)


def cost_map(xs, ys, vx, vy, ax, ay):
    """Travel-time matrix between all pairs of centres: max of the x and y axis times."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    dx = np.abs(xs[:, None] - xs[None, :])
    dy = np.abs(ys[:, None] - ys[None, :])
    return np.maximum(axis_time(dx, vx, ax), axis_time(dy, vy, ay))


def _reference_simulate(cm, t_img, t_recon, t_comps, max_core, fov_steps, z_delay=5.0, want_crit=False):
    """The Python reference of simulate(). Return (cycle_time, core0_end)  [or (cycle_time, core0_end, crit_pos) if
    want_crit: crit_pos = sequence position of the FOV whose task finishes last].

    cm        : NxN travel-time matrix (cost_map) in visit order
    t_img     : imaging time per FOV (already max(capture, side_capture) when side present)
    t_recon   : recon time per FOV
    t_comps   : list (per FOV) of lists of component inspection times, in comp_idx order
    fov_steps : step per FOV
    """
    seqlen = len(t_img)
    if seqlen == 0:
        return 0.0, 0.0
    tl = [float(t_img[0])] * max_core
    last_fov = [0] * max_core          # sequence position of the last task on each core
    insp = range(4, max_core)
    n_insp = max_core - 3

    def dispatch(fi, sel):
        # components of FOV fi: least-loaded inspection core (availability >= sel), first index on ties
        for tt in t_comps[fi]:
            best = 3
            bv = tl[3] if tl[3] > sel else sel
            for k in insp:
                v = tl[k] if tl[k] > sel else sel
                if v < bv:
                    bv = v
                    best = k
            tl[best] = bv + tt
            last_fov[best] = fi

    # first FOV: recon on core 1 (tie -> lowest index), then inspections
    rc = 1 if tl[1] <= tl[2] else 2
    tl[rc] += float(t_recon[0])
    last_fov[rc] = 0
    if n_insp > 0:
        dispatch(0, tl[rc])

    others = range(1, max_core)
    for i in range(seqlen - 1):
        t0 = tl[0] + float(cm[i, i + 1])
        if fov_steps[i] != fov_steps[i + 1]:
            t0 += z_delay
        t0 += float(t_img[i + 1])
        tl[0] = t0
        # checker bumps every core to at least the capture end
        for k in others:
            if tl[k] < t0:
                tl[k] = t0
        rc = 1 if tl[1] <= tl[2] else 2
        tl[rc] += float(t_recon[i + 1])
        last_fov[rc] = i + 1
        if n_insp > 0:
            dispatch(i + 1, tl[rc])

    ct = max(tl)
    if want_crit:
        kmax = max(range(max_core), key=lambda k: (tl[k], -k))
        crit = seqlen - 1 if kmax == 0 else last_fov[kmax]
        return ct, tl[0], crit
    return ct, tl[0]


def plan_times(P, fovs, seq):
    """Helper: build (cm, t_img, t_recon, t_comps, steps) for a sequence of FOV objects."""
    xs = [fovs[k].x for k in seq]
    ys = [fovs[k].y for k in seq]
    cm = cost_map(xs, ys, P.vx, P.vy, P.ax, P.ay)
    t_img, t_comps, steps = [], [], []
    for k in seq:
        f = fovs[k]
        has_side = any(P.side[m] == 1 for m in f.members)
        t_img.append(max(P.capture, P.side_capture) if has_side else P.capture)
        t_comps.append([float(P.time[m]) for m in f.members])
        steps.append(f.step)
    t_recon = [P.recon] * len(seq)
    return cm, t_img, t_recon, t_comps, steps


def evaluate(P, fovs, seq):
    """CT of the plan at max_core cores."""
    cm, t_img, t_recon, t_comps, steps = plan_times(P, fovs, seq)
    return simulate(cm, t_img, t_recon, t_comps, P.max_core, steps)[0]


BETA = 10.0


def sweep(cm, t_img, t_recon, t_comps, max_core, fov_steps, beta=BETA, want_crit=False, k_hint=None):
    """Checker core sweep (rules 2026-09-08): k* = argmin_{k=4..max_core} CT(k) + beta*k.
    With k_hint, descend locally from the hint (CT(k)+beta*k is unimodal in practice) instead of
    evaluating every k.  Return (cost, k_star, ct_at_k_star, core0_end)  [+ crit_pos if want_crit]."""
    # for every k: CT(k) >= core0_end + recon + longest job of the last FOV.  If CT(k) is within
    # beta of that bound, no larger k can beat CT(k) + beta*k, so the sweep can stop at k.
    lb_extra = float(t_recon[-1]) + (max(t_comps[-1]) if len(t_comps) and t_comps[-1] else 0.0)
    return _sweep_core(lambda k: simulate(cm, t_img, t_recon, t_comps, k, fov_steps, want_crit=want_crit), lb_extra,
                       max_core, beta, want_crit, k_hint)


def _sweep_core(sim_k, lb_extra, max_core, beta, want_crit, k_hint):
    """sweep()'s k search on a simulator sim_k(k) -> simulate(...) result (shared with SeqEval)."""
    kmax = max(int(max_core), 4)
    cache = {}

    def at(k):
        if k not in cache:
            r = sim_k(k)
            cache[k] = (r[0] + beta * k, k, r[0], r[1], r[2] if want_crit else None)
        return cache[k]

    def settled(e):
        return e[2] - (e[3] + lb_extra) <= beta + 1e-12

    if k_hint is None:
        best = None
        for k in range(4, kmax + 1):
            e = at(k)
            if best is None or e[0] < best[0]:
                best = e
            if settled(e):
                break
    else:
        k = min(max(int(k_hint), 4), kmax)
        best = at(k)
        for step in (-1, 1):
            j = k + step
            if step == 1 and settled(at(k)):
                continue
            while 4 <= j <= kmax and at(j)[0] < best[0]:
                best = at(j)
                j += step
    return best if want_crit else best[:4]


def evaluate_sweep(P, fovs, seq, beta=BETA, k_hint=None):
    """The checker's score of the plan: min over k of CT(k) + beta k."""
    cm, t_img, t_recon, t_comps, steps = plan_times(P, fovs, seq)
    return sweep(cm, t_img, t_recon, t_comps, P.max_core, steps, beta, k_hint=k_hint)[0]


try:
    from .acceleration import simulate_fast as _simulate_fast
    from . import acceleration as _acc
except ImportError:
    from acceleration import simulate_fast as _simulate_fast
    import acceleration as _acc


def simulate(cm, t_img, t_recon, t_comps, max_core, fov_steps, z_delay=5.0, want_crit=False):
    """The pipeline simulation (see _reference_simulate for the contract): the compiled kernel when available, the
    Python reference otherwise; the same floats either way."""
    return _simulate_fast(cm, t_img, t_recon, t_comps, max_core, fov_steps,
                          z_delay, want_crit, reference=_reference_simulate)


class SeqEval:
    """simulate() / sweep() of a visit order `seq` of FOV indices, from the full per-FOV tables the caller already
    holds (cm over all FOVs; t_img, t_recon, t_comps, steps per FOV), without building the permuted copies
    (cm[np.ix_(seq, seq)] and four lists) on every call. The compiled kernel reads the same values in the same order as
    simulate()'s kernel reads the copies, so every result is the same float. Without the kernel, or after any failure
    of it, each call builds the copies and calls simulate() exactly as the callers did before.
    Contract: the tables are not modified while the evaluator is in use."""

    def __init__(self, cm, t_img, t_recon, t_comps, steps):
        self.cm, self.t_img, self.t_recon, self.t_comps, self.steps = cm, t_img, t_recon, t_comps, steps
        self.fast = None
        try:
            if _acc._NUMBA and _acc._SEQ:
                offs = np.zeros(len(t_comps) + 1, dtype=np.int64)
                acc, flat = 0, []
                for i, c in enumerate(t_comps):
                    acc += len(c)
                    offs[i + 1] = acc
                    flat.extend(c)
                self.fast = (_acc._simulate_seq_nb, np.ascontiguousarray(cm, dtype=np.float64),
                             np.asarray(t_img, dtype=np.float64), np.asarray(t_recon, dtype=np.float64),
                             np.array(flat, dtype=np.float64), offs, np.asarray(steps, dtype=np.int64))
        except Exception:
            self.fast = None

    def simulate(self, seq, max_core, z_delay=5.0):
        if self.fast is not None and _acc._NUMBA and len(seq) and max_core >= 4:
            try:
                kern, c, ti, tr, flat, offs, st = self.fast
                sa = np.array(seq, dtype=np.int64)
                if sa.min() >= 0 and sa.max() < ti.shape[0]:  # other indices: the copies below (raise or wrap as before)
                    return kern(c, sa, ti, tr, flat, offs, int(max_core), st, float(z_delay))
            except Exception:
                self.fast = None
        return simulate(self.cm[np.ix_(seq, seq)], [self.t_img[q] for q in seq], [self.t_recon[q] for q in seq],
                        [self.t_comps[q] for q in seq], max_core, [self.steps[q] for q in seq], z_delay)

    def sweep(self, seq, max_core, beta=BETA, k_hint=None):
        """sweep(...)[:4] of the permuted inputs (want_crit=False)."""
        last = seq[-1]
        lb_extra = float(self.t_recon[last]) + (max(self.t_comps[last]) if len(seq) and self.t_comps[last] else 0.0)
        return _sweep_core(lambda k: self.simulate(seq, k), lb_extra, max_core, beta, False, k_hint)[:4]
