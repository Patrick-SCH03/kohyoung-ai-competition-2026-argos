"""Part -> FOV reassignment after the visit order is fixed.

Heavy inspection jobs released late create the inspection tail at k*. A non-big normal part that also fits an
earlier-visited FOV (same step) can move there so the dispatcher sees it sooner; a FOV emptied by the move
disappears. Every move is kept only if the scoring objective (core sweep, or CT) improves.
"""
import time
import numpy as np
try:
    from . import ct_sim
except ImportError:
    import ct_sim


def _entry(P, f):
    """plan_times' capture time and job list of one FOV."""
    has_side = any(P.side[m] == 1 for m in f.members)
    return max(P.capture, P.side_capture) if has_side else P.capture, [float(P.time[m]) for m in f.members]


def _score(P, fovs, seq, sweep_obj, k_hint, cm_all=None, tab=None):
    if cm_all is None:
        cm, t_img, t_recon, t_comps, steps = ct_sim.plan_times(P, fovs, seq)
    else:  # plan_times(P, fovs, seq) from cm_all (cost_map over every FOV; elementwise, so the same floats) and the
        # per-FOV table tab[k] = _entry(P, fovs[k]), kept current by the caller for every FOV whose members change
        cm = cm_all[np.ix_(seq, seq)]
        t_img, t_comps = [tab[k][0] for k in seq], [tab[k][1] for k in seq]
        t_recon, steps = [P.recon] * len(seq), [fovs[k].step for k in seq]
    if sweep_obj:
        cost, k, ct, c0 = ct_sim.sweep(cm, t_img, t_recon, t_comps, P.max_core, steps, k_hint=k_hint)
        return cost, k, ct - c0
    ct, c0 = ct_sim.simulate(cm, t_img, t_recon, t_comps, P.max_core, steps)
    return ct, P.max_core, ct - c0


def reassign_heavy(P, fovs, seq, sweep_obj=True, deadline=None, top=40, min_tail=0.5):
    """Return (fovs, seq) with heavy components moved to earlier FOVs when it lowers the score."""
    for f in fovs:
        f.members.sort(key=lambda m: (-P.time[m], m))
    xs, ys = np.asarray([f.x for f in fovs], dtype=float), np.asarray([f.y for f in fovs], dtype=float)
    cm_all = ct_sim.cost_map(xs, ys, P.vx, P.vy, P.ax, P.ay)
    tab = [_entry(P, f) for f in fovs]
    cur, k_hint, tail = _score(P, fovs, seq, sweep_obj, None, cm_all, tab)
    if tail < min_tail:
        return fovs, seq
    pos = {k: i for i, k in enumerate(seq)}
    cands = [(float(P.time[m]), int(m), k) for k in seq for m in fovs[k].members
             if not P.big[m] and P.type[m] == 0 and not fovs[k].exclusive]
    cands.sort(key=lambda t: (-t[0], t[1]))
    idx_all = np.arange(P.n)
    for t, m, k in cands[:top]:
        if deadline is not None and time.perf_counter() > deadline:
            break
        if m not in fovs[k].members:
            continue
        src = fovs[k].members
        inside = P.contains(idx_all[m:m + 1], xs, ys)  # every centre at once (FOVs do not move here)
        targets = [g for g in seq if g != k and pos[g] < pos[k] and fovs[g].step == P.step[m]
                   and not fovs[g].exclusive and inside[g]]
        if not targets:
            continue
        best = None
        src_idx = src.index(m)
        src.pop(src_idx)
        tab_k = tab[k]
        tab[k] = _entry(P, fovs[k])
        seq_wo = seq if src else [q for q in seq if q != k]
        for g in targets:
            dst = fovs[g].members
            ins = next((i for i, q in enumerate(dst) if (-P.time[q], q) > (-t, m)), len(dst))
            dst.insert(ins, m)
            tab_g = tab[g]
            tab[g] = _entry(P, fovs[g])
            s, kk, _ = _score(P, fovs, seq_wo, sweep_obj, k_hint, cm_all, tab)
            dst.pop(ins)
            tab[g] = tab_g
            if s < (best[0] if best else cur) - 1e-9:
                best = (s, g, ins, kk)
        if best is None:
            src.insert(src_idx, m)
            tab[k] = tab_k
            continue
        s, g, ins, kk = best
        fovs[g].members.insert(ins, m)
        tab[g] = _entry(P, fovs[g])
        cur, k_hint, seq = s, kk, seq_wo
        pos = {q: i for i, q in enumerate(seq)}
    if len(seq) < len(fovs):
        keep = sorted(seq)
        remap = {old: new for new, old in enumerate(keep)}
        fovs = [fovs[q] for q in keep]
        seq = [remap[q] for q in seq]
    return fovs, seq
