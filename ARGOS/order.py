"""Visit order: rule blocks (fiducial -> barcode -> normal, grouped by step to keep the +5 s step changes few),
nearest neighbour + 2-opt + Or-opt paths on the exact travel-time matrix, a scored portfolio of structural candidates
(heavy FOVs first, light FOV last, heavy prelude, reversed step order, deadline repair, small runs re-inserted), and a
critical-FOV relocation local search on the finished plan. The compiled scans make the same moves in the same order as
the numpy scans they replace."""
import time
import numpy as np
try:
    from .ct_sim import cost_map, simulate, sweep, BETA, SeqEval
except ImportError:
    from ct_sim import cost_map, simulate, sweep, BETA, SeqEval


try:  # compiled sequential scans with the same move choice, thresholds and float operation order
    import numpy as _np
    from numba import njit as _njit

    @_njit('int64[::1](float64[:, ::1], int64[::1], boolean)', cache=False, nogil=True)
    def _two_opt_nb(cm, p, fixed_end):
        m = p.shape[0]
        jmax = m - 2 if fixed_end else m - 1
        improved, it = True, 0
        while improved and it < 300:
            improved = False
            it += 1
            i_cur = 1
            while True:
                found = False
                for i in range(i_cur, jmax):
                    a, b = p[i - 1], p[i]
                    bestv, bestj = np.inf, -1
                    for j in range(2, jmax + 1):
                        if j <= i:
                            continue
                        c = p[j]
                        if j < m - 1:
                            d = p[j + 1]
                            v = cm[a, c] - cm[a, b] + (cm[b, d] - cm[c, d])
                        else:
                            v = cm[a, c] - cm[a, b] + 0.0
                        if v < bestv:
                            bestv, bestj = v, j
                    if bestv < -1e-12:
                        lo, hi = i, bestj
                        while lo < hi:
                            tmp = p[lo]
                            p[lo] = p[hi]
                            p[hi] = tmp
                            lo += 1
                            hi -= 1
                        improved = True
                        i_cur = i + 1
                        found = True
                        break
                if not found:
                    break
        return p

    @_njit('int64[::1](float64[:, ::1], int64[::1], boolean)', cache=False, nogil=True)
    def _or_opt_nb(cm, p, fixed_end):
        m = p.shape[0]
        improved, it = True, 0
        while improved and it < 100:
            improved = False
            it += 1
            last_mov = m - 2 if fixed_end else m - 1
            for L in (1, 2, 3):
                n_r = m - L
                pmax = n_r if not fixed_end else n_r - 1
                i_hi = last_mov - L + 1          # ii = 1 .. i_hi
                if pmax < 1 or i_hi < 1:
                    continue
                i_cur = 1
                R = np.empty(n_r, dtype=np.int64)
                while True:
                    found = False
                    for i in range(i_cur, i_hi + 1):
                        jj = i + L - 1
                        s0, s1, prev_n = p[i], p[jj], p[i - 1]
                        if jj + 1 < m:
                            nxt = p[jj + 1]
                            remove_gain = cm[prev_n, s0] + cm[s1, nxt] - cm[prev_n, nxt]
                        else:
                            remove_gain = cm[prev_n, s0]
                        for t in range(n_r):
                            R[t] = p[t + L] if t >= i else p[t]
                        bf, kf, br, kr = np.inf, 0, np.inf, 0
                        for t in range(pmax):
                            u = R[t]
                            if t + 1 < n_r:
                                v = R[t + 1]
                                base = cm[u, v]
                                af = cm[u, s0] + cm[s1, v] - base
                                ar = cm[u, s1] + cm[s0, v] - base
                            else:
                                af = cm[u, s0] + 0.0 - 0.0
                                ar = cm[u, s1] + 0.0 - 0.0
                            if af < bf:
                                bf, kf = af, t
                            if ar < br:
                                br, kr = ar, t
                        best_f = bf - remove_gain
                        best_r = br - remove_gain
                        if min(best_f, best_r) < -1e-12:
                            q = np.empty(m, dtype=np.int64)
                            forward = best_f <= best_r
                            pins = (kf if forward else kr) + 1
                            w = 0
                            for t in range(pins):
                                q[w] = R[t]
                                w += 1
                            for t in range(L):
                                q[w] = p[i + t] if forward else p[jj - t]
                                w += 1
                            for t in range(pins, n_r):
                                q[w] = R[t]
                                w += 1
                            p = q
                            improved = True
                            i_cur = i
                            found = True
                            break
                    if not found:
                        break
        return p

    @_njit('int64[::1](float64[:, ::1], int64[::1], int64)', cache=False, nogil=True)
    def _nn_nb(cm, rem, prev):
        """nn_path's loop: from prev, repeatedly the first remaining id with the smallest cm[cur, id] (np.argmin's
        rules: the first occurrence of the minimum, and the first NaN if the row has one), removed keeping the order of
        the rest."""
        m = rem.shape[0]
        r = rem.copy()
        path = _np.empty(m + 1, dtype=_np.int64)
        path[0] = prev
        cur = prev
        for t in range(m):
            left = m - t
            best = 0
            bv = cm[cur, r[0]]
            if bv == bv:
                for j in range(1, left):
                    v = cm[cur, r[j]]
                    if v != v:
                        best = j
                        break
                    if v < bv:
                        bv = v
                        best = j
            cur = r[best]
            path[t + 1] = cur
            for j in range(best, left - 1):
                r[j] = r[j + 1]
        return path

    _NUMBA_ORDER = True
except Exception:  # numba unavailable: pure numpy path below
    _NUMBA_ORDER = False
_NUMBA_NN = _NUMBA_ORDER

_NUMBA_MARGIN = 0.5  # use the compiled scan only when the deadline (if any) is comfortably far


def _disable_order():
    global _NUMBA_ORDER
    _NUMBA_ORDER = False


def _use_numba(deadline):
    return _NUMBA_ORDER and (deadline is None or deadline - time.perf_counter() > _NUMBA_MARGIN)


def two_opt_open(cm, path, fixed_end=False, deadline=None):
    if _use_numba(deadline) and len(path) >= 4:
        try:
            return [int(q) for q in _two_opt_nb(np.ascontiguousarray(cm, dtype=np.float64),
                                              np.array(path, dtype=np.int64), bool(fixed_end))]
        except Exception:  # compiled path failed (dispatch before execution): original numpy scan from now on
            _disable_order()
    return _two_opt_open_py(cm, path, fixed_end, deadline)


def or_opt_open(cm, path, fixed_end=False, deadline=None):
    if _use_numba(deadline) and len(path) >= 4:
        try:
            return [int(q) for q in _or_opt_nb(np.ascontiguousarray(cm, dtype=np.float64),
                                             np.array(path, dtype=np.int64), bool(fixed_end))]
        except Exception:  # compiled path failed (dispatch before execution): original numpy scan from now on
            _disable_order()
    return _or_opt_open_py(cm, path, fixed_end, deadline)


def _two_opt_open_py(cm, path, fixed_end=False, deadline=None):
    """First-improvement 2-opt on an open path (position 0 fixed; last fixed if fixed_end).
    All (i, j) deltas are evaluated at once; the earliest improvable i is applied, exactly as
    a sequential scan would.  Stops early (anytime) once the deadline passes."""
    p = np.asarray(path, dtype=int)
    m = len(p)
    if m < 4:
        return list(p)
    jmax = m - 2 if fixed_end else m - 1
    ii = np.arange(1, jmax)
    js = np.arange(2, jmax + 1)
    valid = js[None, :] > ii[:, None]
    has_next = (js < m - 1)[None, :]
    jn = np.minimum(js + 1, m - 1)
    improved, it = True, 0
    while improved and it < 300:
        improved = False
        it += 1
        i_cur = 1
        while True:
            if deadline is not None and time.perf_counter() > deadline:
                return [int(q) for q in p]
            a, b = p[ii - 1][:, None], p[ii][:, None]
            c, d = p[js][None, :], p[jn][None, :]
            delta = cm[a, c] - cm[a, b] + np.where(has_next, cm[b, d] - cm[c, d], 0.0)
            delta = np.where(valid, delta, np.inf)
            k = np.argmin(delta, axis=1)
            best = delta[np.arange(len(ii)), k]
            rows = np.where((best < -1e-12) & (ii >= i_cur))[0]
            if len(rows) == 0:
                break
            r = int(rows[0])
            i, j = int(ii[r]), int(js[k[r]])
            p[i:j + 1] = p[i:j + 1][::-1].copy()
            improved = True
            i_cur = i + 1
    return [int(q) for q in p]


def _or_opt_open_py(cm, path, fixed_end=False, deadline=None):
    """First-improvement Or-opt (segments of 1-3) on an open path, vectorised over the segment
    start i; identical move choice to a sequential scan.  Anytime under a deadline."""
    p = np.asarray(path, dtype=int)
    m = len(p)
    if m < 4:
        return list(p)
    improved, it = True, 0
    while improved and it < 100:
        improved = False
        it += 1
        last_mov = m - 2 if fixed_end else m - 1
        for L in (1, 2, 3):
            n_r = m - L
            pmax = n_r if not fixed_end else n_r - 1
            ii = np.arange(1, last_mov - L + 2)
            if pmax < 1 or len(ii) == 0:
                continue
            pos = np.arange(1, pmax + 1)
            has_v = (pos < n_r)[None, :]
            vpos = np.minimum(pos, n_r - 1)
            base_idx = np.arange(n_r)[None, :]
            ridx = base_idx + (base_idx >= ii[:, None]) * L
            i_cur = 1
            while True:
                if deadline is not None and time.perf_counter() > deadline:
                    return [int(q) for q in p]
                jj = ii + L - 1
                s0, s1, prev_n = p[ii], p[jj], p[ii - 1]
                nxt = p[np.minimum(jj + 1, m - 1)]
                remove_gain = np.where(jj + 1 < m, cm[prev_n, s0] + cm[s1, nxt] - cm[prev_n, nxt], cm[prev_n, s0])
                R = p[ridx]
                u = R[:, :pmax]
                v = R[:, vpos]
                base = np.where(has_v, cm[u, v], 0.0)
                add_f = cm[u, s0[:, None]] + np.where(has_v, cm[s1[:, None], v], 0.0) - base
                add_r = cm[u, s1[:, None]] + np.where(has_v, cm[s0[:, None], v], 0.0) - base
                kf = np.argmin(add_f, axis=1)
                kr = np.argmin(add_r, axis=1)
                ar = np.arange(len(ii))
                best_f = add_f[ar, kf] - remove_gain
                best_r = add_r[ar, kr] - remove_gain
                rows = np.where((np.minimum(best_f, best_r) < -1e-12) & (ii >= i_cur))[0]
                if len(rows) == 0:
                    break
                r = int(rows[0])
                i, j = int(ii[r]), int(jj[r])
                Rr = R[r]
                if best_f[r] <= best_r[r]:
                    pins, sseg = int(kf[r]) + 1, p[i:j + 1]
                else:
                    pins, sseg = int(kr[r]) + 1, p[i:j + 1][::-1]
                p = np.concatenate((Rr[:pins], sseg, Rr[pins:]))
                improved = True
                i_cur = i
    return [int(q) for q in p]


def improve(cm, path, fixed_end=False, deadline=None):
    """2-opt, Or-opt, 2-opt on an open path (position 0 fixed; the last too if fixed_end)."""
    path = two_opt_open(cm, path, fixed_end, deadline)
    path = or_opt_open(cm, path, fixed_end, deadline)
    return two_opt_open(cm, path, fixed_end, deadline)


def nn_path(cm, ids, prev):
    """Nearest-neighbour path from prev through ids (first index on ties); [prev] + the visited ids."""
    global _NUMBA_NN
    if _NUMBA_NN:
        ids = list(ids)
        if (len(ids) > 1 and isinstance(cm, np.ndarray) and cm.dtype == np.float64 and cm.ndim == 2
                and isinstance(prev, (int, np.integer)) and all(isinstance(q, (int, np.integer)) for q in ids)):
            try:  # compiled loop, same choices
                out = _nn_nb(np.ascontiguousarray(cm, dtype=np.float64), np.array(ids, dtype=np.int64), int(prev))
                return [prev] + [int(q) for q in out[1:]]
            except Exception:  # dispatch failed: the loop below from now on
                _NUMBA_NN = False
    path = [prev]
    cur = prev
    rem = np.asarray(list(ids), dtype=int)
    while len(rem):
        k = int(np.argmin(cm[cur, rem]))
        cur = int(rem[k])
        path.append(cur)
        rem = np.delete(rem, k)
    return path


def path_plain(cm, ids, prev, deadline=None):
    return improve(cm, nn_path(cm, ids, prev), deadline=deadline)[1:]


def path_fixed_end(cm, ids, prev, end, deadline=None):
    rest = [k for k in ids if k != end]
    return improve(cm, nn_path(cm, rest, prev) + [end], fixed_end=True, deadline=deadline)[1:]


def path_fixed_start(cm, ids, prev, start, deadline=None):
    rest = [k for k in ids if k != start]
    return [start] + improve(cm, nn_path(cm, rest, start), deadline=deadline)[1:]


def deadline_repair(P, cm, seq, fid_n, block_of, t_img, peak, steps, max_iter=30):
    """Move FOVs whose largest task cannot finish before Core0 ends to an earlier position in
    their block (cheapest travel insertion among feasible positions)."""
    seq = list(seq)
    n = len(seq)
    if n < 3:
        return seq

    def times(sq):
        t = np.zeros(n)
        acc = float(t_img[sq[0]])
        t[0] = acc
        for p in range(1, n):
            acc += float(cm[sq[p - 1], sq[p]]) + (5.0 if steps[sq[p - 1]] != steps[sq[p]] else 0.0) + float(t_img[sq[p]])
            t[p] = acc
        return t

    for _ in range(max_iter):
        t = times(seq)
        T_end = t[-1]
        viol = [(peak[seq[p]] - (T_end - t[p]), p) for p in range(fid_n, n)]
        viol = [v for v in viol if v[0] > 1e-9]
        if not viol:
            break
        viol.sort(reverse=True)
        p = viol[0][1]
        k = seq[p]
        b = block_of[k]
        lim = T_end - peak[k]
        cands = [q for q in range(fid_n, p) if block_of[seq[q]] == b and t[q] - t_img[seq[q]] <= lim + 1e-9]
        if not cands:
            cands = [q for q in range(fid_n, p) if block_of[seq[q]] == b][:1]
            if not cands:
                break
        best_q, best_d = None, None
        for q in cands:
            a = seq[q - 1] if q > 0 else None
            d = (cm[a, k] if a is not None else 0.0) + cm[k, seq[q]] - (cm[a, seq[q]] if a is not None else 0.0)
            if best_d is None or d < best_d - 1e-12:
                best_q, best_d = q, d
        seq.pop(p)
        seq.insert(best_q, k)
    return seq


def _blocks(P, fovs, reverse_normal_steps=False):
    N = len(fovs)
    types = [f.type_of(P) for f in fovs]
    fid = sorted([k for k in range(N) if types[k] == 2], key=lambda k: min(fovs[k].members))
    blocks = []
    prev_step = fovs[fid[-1]].step if fid else None
    for t in (1, 0):
        group = [k for k in range(N) if types[k] == t]
        if not group:
            continue
        steps_present = sorted(set(fovs[k].step for k in group))
        if prev_step in steps_present:
            sub_order = [prev_step] + [s for s in steps_present if s != prev_step]
        else:
            sub_order = steps_present
        if t == 0 and reverse_normal_steps:
            sub_order = sub_order[::-1]
        for s in sub_order:
            blocks.append([k for k in group if fovs[k].step == s])
            prev_step = s
    return fid, blocks


def build_sequence(P, fovs, cfg):
    """Visit order of the FOVs: fiducials first (index order), then the barcode and normal blocks by step.
    cfg['order'] == 'tsp': each block as a travel-only path. Otherwise a portfolio of structural candidates, each scored
    by the exact objective (candidates whose lower bound cannot beat the best are skipped; a camera-bound first
    candidate keeps only the first three), then the best block_insert_top candidates are tried with small same-step runs
    re-inserted inside the other runs. Returns (seq, travel-time matrix over all FOVs)."""
    xs = np.asarray([f.x for f in fovs])
    ys = np.asarray([f.y for f in fovs])
    cm = cost_map(xs, ys, P.vx, P.vy, P.ax, P.ay)
    fid, blocks = _blocks(P, fovs)
    types_ = [f.type_of(P) for f in fovs]

    def start_of(seq, ids):
        if seq:
            return seq[-1], ids
        start = min(ids, key=lambda k: (xs[k] ** 2 + ys[k] ** 2, k))
        return start, [k for k in ids if k != start]

    deadline = cfg.get('deadline')
    if cfg.get('order', 'portfolio') == 'tsp' or not blocks:
        seq = list(fid)
        for ids in blocks:
            prev, rest = start_of(seq, ids)
            if not seq:
                seq.append(prev)
            seq.extend(path_plain(cm, rest, prev, deadline))
        return seq, cm

    t_img, t_comps, steps = [], [], []
    for f in fovs:
        has_side = any(P.side[m] == 1 for m in f.members)
        t_img.append(max(P.capture, P.side_capture) if has_side else P.capture)
        t_comps.append([float(P.time[m]) for m in f.members])
        steps.append(f.step)
    load = np.asarray([sum(t) for t in t_comps])
    peak = np.asarray([max(t) if t else 0.0 for t in t_comps])
    ev = SeqEval(cm, t_img, [P.recon] * len(fovs), t_comps, steps)  # scores candidates without permuted copies
    cache = {}

    def plain(prev, rest):
        key = (prev, tuple(rest))
        if key not in cache:
            cache[key] = path_plain(cm, rest, prev, deadline)
        return cache[key]

    def assemble(heavy_first, light_end_k, heavy_front_k, bl):
        seq = list(fid)
        for bi, ids in enumerate(bl):
            prev, rest = start_of(seq, ids)
            if not seq:
                seq.append(prev)
            last = (bi == len(bl) - 1)
            if heavy_front_k and len(rest) > 2:
                k = min(heavy_front_k, len(rest) - 1)
                front = sorted(rest, key=lambda q: (-peak[q], -load[q], q))[:k]
                front.sort(key=lambda q: (-peak[q], q))
                others = [q for q in rest if q not in front]
                seq.extend(front)
                if others:
                    seq.extend(plain(seq[-1], others))
            elif heavy_first and len(rest) > 2:
                st = max(rest, key=lambda k: (load[k], k))
                seq.extend(path_fixed_start(cm, rest, prev, st, deadline))
            elif last and light_end_k is not None and len(rest) > 2:
                cands = sorted(rest, key=lambda k: (peak[k], load[k], k))
                end = cands[min(light_end_k, len(cands) - 1)]
                seq.extend(path_fixed_end(cm, rest, prev, end, deadline))
            else:
                seq.extend(plain(prev, rest))
        return seq

    def assemble_prelude(k):
        seq = list(fid)
        normal_blocks = [ids for ids in blocks if ids and types_[ids[0]] == 0]
        for ids in (ids for ids in blocks if ids and types_[ids[0]] != 0):
            prev, rest = start_of(seq, ids)
            if not seq:
                seq.append(prev)
            seq.extend(plain(prev, rest))
        pool = [q for ids in normal_blocks for q in ids]
        if not pool:
            return seq
        pre = sorted(pool, key=lambda q: (-peak[q], -load[q], q))[:min(k, len(pool))]
        first_step = fovs[pre[0]].step
        pre.sort(key=lambda q: (fovs[q].step != first_step, -peak[q], q))
        seq.extend(pre)
        last_step = fovs[pre[-1]].step
        rem = [[q for q in ids if q not in pre] for ids in normal_blocks]
        rem = sorted([b for b in rem if b], key=lambda b: (fovs[b[0]].step != last_step, fovs[b[0]].step))
        for ids in rem:
            seq.extend(plain(seq[-1], ids))
        return seq

    base_specs = [(False, None, 0), (False, 0, 0), (False, 1, 0), (True, None, 0),
                  (False, None, 1), (False, None, 2), (False, None, 4), (False, None, 8),
                  (False, None, 10 ** 6), ('deadline', None, 0)]
    specs = [sp + (False,) for sp in base_specs]
    if sum(1 for ids in blocks if ids and types_[ids[0]] == 0) >= 2:
        specs += [sp + (True,) for sp in base_specs]
        specs += [('prelude', None, 1, False), ('prelude', None, 2, False), ('prelude', None, 4, False)]
    blocks_r = None
    best, best_ct = None, None
    cands = []
    sweep_obj = cfg.get('objective', 'ct') == 'sweep'
    load_total = float(load.sum())
    kmax = max(int(P.max_core), 4)
    k_hint = None
    for si, sp in enumerate(specs):
        if best is not None and deadline is not None and time.perf_counter() > deadline:
            break
        hf, le, hk, rev = sp
        bl = blocks
        if rev:
            if blocks_r is None:
                _, blocks_r = _blocks(P, fovs, reverse_normal_steps=True)
            bl = blocks_r
        if hf == 'prelude':
            sq = assemble_prelude(hk)
        elif hf == 'deadline':
            sq = assemble(False, None, 0, bl)
            bo = {k: bi for bi, ids in enumerate(bl) for k in ids}
            sq = deadline_repair(P, cm, sq, len(fid), bo, t_img, peak, steps)
        else:
            sq = assemble(hf, le, hk, bl)
        sa = np.asarray(sq)
        if best_ct is not None:
            st = np.asarray([steps[k] for k in sq])
            core0 = (float(t_img[sq[0]]) + float(cm[sa[:-1], sa[1:]].sum())
                     + 5.0 * int((st[:-1] != st[1:]).sum()) + float(sum(t_img[k] for k in sq[1:])) + P.recon)
            if sweep_obj:
                # lower bound of min_k CT(k)+beta*k: camera chain and total inspection load
                rel0 = float(t_img[sq[0]]) + P.recon
                lb = min(max(core0, rel0 + load_total / (kk - 3)) + BETA * kk for kk in range(4, kmax + 1))
            else:
                lb = core0
            if lb >= best_ct - 1e-9:
                continue
        if sweep_obj:
            score, k_hint, ct, c0 = ev.sweep(sq, P.max_core, k_hint=k_hint)
        else:
            ct, c0 = ev.simulate(sq, P.max_core)
            score = ct
        cands.append((score, si, sq))
        if best_ct is None or score < best_ct - 1e-9:
            best, best_ct = sq, score
        if si == 0 and ct - c0 <= 2.0 * P.recon + 1e-9:
            specs = specs[:3]

    def score_of(sq):
        return ev.sweep(sq, P.max_core, k_hint=k_hint)[0] if sweep_obj else ev.simulate(sq, P.max_core)[0]

    def insert_blocks(seq0, sc0):
        """A small same-step run re-inserted inside the other runs (extra Z transition) smooths
        inspection load on inspection-bound boards; 7 positions per run, up to 2 rounds."""
        cur, cur_sc = seq0, sc0
        fn = next((i for i, k in enumerate(cur) if types_[k] == 0), len(cur))
        improved, rounds = True, 0
        while improved and rounds < 2:
            improved, rounds = False, rounds + 1
            runs, a = [], fn
            while a < len(cur):
                b = a
                while b + 1 < len(cur) and steps[cur[b + 1]] == steps[cur[a]]:
                    b += 1
                runs.append((a, b))
                a = b + 1
            if len(runs) < 2:
                break
            for (a, b) in runs:
                if b - a + 1 > (len(cur) - fn) // 2:
                    continue
                blk = cur[a:b + 1]
                rest = cur[:a] + cur[b + 1:]
                m = len(rest) - fn
                for j in range(1, 8):
                    p = fn + (m * j) // 8
                    if p == a:
                        continue
                    sq = rest[:p] + blk + rest[p:]
                    sc = score_of(sq)
                    if sc < cur_sc - 1e-9:
                        cur, cur_sc, improved = sq, sc, True
                if improved:
                    break
        return cur, cur_sc

    if cfg.get('block_insert', True) and len(specs) > 3:
        cands.sort(key=lambda t: (t[0], t[1]))
        for sc0, _, sq0 in cands[:int(cfg.get('block_insert_top', 1))]:
            sq, sc = insert_blocks(sq0, sc0)
            if sc < best_ct - 1e-9:
                best, best_ct = sq, sc
    return best, cm


def critical_relocate(P, fovs, seq, cm, t_img, t_comps, steps, max_rounds=8, max_evals=24, objective='ct'):
    """Move the FOV whose inspection task finishes last to the best earlier position (same type
    block, any step); repeat while the simulated score improves.  Under the sweep objective the
    schedules at k*, k*+1 and max_core each nominate a critical FOV."""
    seq = list(seq)
    n = len(seq)
    if n < 4:
        return seq
    types = [f.type_of(P) for f in fovs]
    sweep_obj = objective == 'sweep'
    khint = [None]

    def args_for(sq):
        sub = cm[np.ix_(sq, sq)]
        return (sub, [t_img[k] for k in sq], [P.recon] * len(sq), [t_comps[k] for k in sq], P.max_core,
                [steps[k] for k in sq])

    def sim(sq):
        args = args_for(sq)
        if sweep_obj:
            score, kk, ct_, c0_, cr = sweep(*args, want_crit=True, k_hint=khint[0])
            khint[0] = kk
            return score, ct_, c0_, cr
        ct_, c0_, cr = simulate(*args, want_crit=True)
        return ct_, ct_, c0_, cr

    def crit_positions(sq, crit):
        if not sweep_obj:
            return [crit]
        args = args_for(sq)
        out = [crit]
        for kk in sorted({min(khint[0] + 1, P.max_core), P.max_core}):
            if kk > khint[0]:
                cr = (simulate(*args, want_crit=True)[2] if kk == P.max_core
                      else simulate(args[0], args[1], args[2], args[3], kk, args[5], want_crit=True)[2])
                if cr not in out:
                    out.append(cr)
        return out

    ct, raw, c0, crit = sim(seq)
    for _ in range(max_rounds):
        if raw - c0 <= 2.0 * P.recon + 1e-9:
            break
        best = None
        for cpos in crit_positions(seq, crit):
            k = seq[cpos]
            lo = 0
            for p in range(n):
                if types[seq[p]] > types[k]:
                    lo = p + 1
            if cpos <= lo:
                continue
            cands = list(range(lo, cpos))
            if len(cands) > max_evals:
                idxs = np.unique(np.round(np.linspace(0, len(cands) - 1, max_evals)).astype(int))
                cands = [cands[i] for i in idxs]
            for p in cands:
                sq = seq[:cpos] + seq[cpos + 1:]
                sq.insert(p, k)
                c, r_, c0_, cr = sim(sq)
                if c < ct - 1e-9 and (best is None or c < best[0] - 1e-9):
                    best = (c, sq, cr, r_, c0_)
        if best is None:
            break
        ct, seq, crit, raw, c0 = best
    return seq
