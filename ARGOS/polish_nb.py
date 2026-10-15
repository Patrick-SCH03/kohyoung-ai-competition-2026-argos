"""The greedy polishes of a finished plan (anneal.polish_order and anneal.polish_members) as compiled loops.

The moves, the acceptance rule (T = 1e-3, equal scores accepted), the full core sweep on each new best and the stopping
rules (stall, cap, deadline) are those of the Python polishes. Two things differ:
- the random stream is a splitmix64 generator seeded from `seed` (not Python's random), so other moves are drawn;
- a candidate is simulated only from its first changed visit position on, starting from the core state stored for the
  current order at that position (the same operations in the same order as the kernel run from the start, so the same
  floats). The stored states are replaced when a candidate is accepted.
One iteration then costs a fraction of the Python iteration (list copies, array conversion and the unchanged prefix are
gone), so the callers can afford a larger budget in the same time.

All kernels are compiled at import (explicit signatures). OK is False when numba is missing or a compile fails; any
failure later sets OK to False and the callers run the Python polishes instead."""
import time

OK = False
try:
    import numpy as np
    from numba import njit

    _G = np.uint64(0x9E3779B97F4A7C15)
    _M1 = np.uint64(0xBF58476D1CE4E5B9)
    _M2 = np.uint64(0x94D049BB133111EB)
    _S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)

    @njit('float64(uint64[::1])', cache=False, nogil=True)
    def _u01(st):
        """splitmix64 step; a double in [0, 1) from the top 53 bits."""
        st[0] += _G
        z = st[0]
        z = (z ^ (z >> _S30)) * _M1
        z = (z ^ (z >> _S27)) * _M2
        z = z ^ (z >> _S31)
        return np.float64(z >> _S11) * (1.0 / 9007199254740992.0)

    @njit('int64(uint64[::1], int64, int64)', cache=False, nogil=True)
    def _rr(st, a, b):
        """an integer in [a, b) (b > a)."""
        j = a + np.int64(_u01(st) * (b - a))
        return j if j < b else b - 1

    @njit('UniTuple(float64, 2)(float64[:, ::1], int64[::1], int64, float64[::1], float64[::1], float64[::1], int64[::1], '
          'int64[::1], int64, float64, float64[:, ::1], float64[:, ::1], int64)', cache=False, nogil=True)
    def _sim_from(cm, seq, n, t_img, t_recon, flat, offs, steps, k, z_delay, snap_in, snap_out, start):
        """acceleration._simulate_seq_nb on seq[:n] with k cores, resumed at position `start` from snap_in[start] (the
        core state before that position; start = 0 starts from the beginning). snap_out[i] receives the state before
        position i for every i >= start. Returns (CT, core-0 end)."""
        if k < 4 or n < 1 or n > seq.shape[0] or snap_out.shape[0] < n or snap_out.shape[1] < k or (
                start > 0 and (start >= snap_in.shape[0] or snap_in.shape[1] < k)):
            raise ValueError('polish_nb: table shapes')  # no bounds checks in compiled code: refuse, the caller falls back
        tl = np.empty(k)
        if start == 0:
            for c in range(k):
                tl[c] = t_img[seq[0]]
        else:
            for c in range(k):
                tl[c] = snap_in[start, c]
        for i in range(start, n):
            for c in range(k):
                snap_out[i, c] = tl[c]
            f = seq[i]
            if i:
                g = seq[i - 1]
                t0 = tl[0] + cm[g, f]
                if steps[g] != steps[f]:
                    t0 += z_delay
                t0 += t_img[f]
                tl[0] = t0
                for c in range(1, k):
                    if tl[c] < t0:
                        tl[c] = t0
            rc = 1 if tl[1] <= tl[2] else 2
            tl[rc] += t_recon[f]
            ready = tl[rc]
            for q in range(offs[f], offs[f + 1]):
                duration = flat[q]
                best = 3
                value = tl[3] if tl[3] > ready else ready
                for c in range(4, k):
                    candidate = tl[c] if tl[c] > ready else ready
                    if candidate < value:
                        value = candidate
                        best = c
                tl[best] = value + duration
        m = tl[0]
        for c in range(1, k):
            if tl[c] > m:
                m = tl[c]
        return m, tl[0]

    @njit('Tuple((float64, int64))(float64[:, ::1], int64[::1], int64, float64[::1], float64[::1], float64[::1], int64[::1], '
          'int64[::1], int64, float64[:, ::1])', cache=False, nogil=True)
    def _sweep(cm, seq, n, t_img, t_recon, flat, offs, steps, kmax, scratch):
        """ct_sim._sweep_core without a hint (beta 10): k = 4, 5, ... keeping the lowest CT + 10k, stopping once CT is
        within beta of core-0 end + recon + longest job of the last FOV. Returns (cost, k)."""
        last = seq[n - 1]
        lb = t_recon[last]
        if offs[last + 1] > offs[last]:
            mx = flat[offs[last]]
            for q in range(offs[last] + 1, offs[last + 1]):
                if flat[q] > mx:
                    mx = flat[q]
            lb = lb + mx
        else:
            lb = lb + 0.0
        best_c = np.inf
        best_k = 4
        top = kmax if kmax > 4 else 4
        for k in range(4, top + 1):
            m, t0 = _sim_from(cm, seq, n, t_img, t_recon, flat, offs, steps, k, 5.0, scratch, scratch, 0)
            c = m + 10.0 * k
            if c < best_c:
                best_c = c
                best_k = k
            if m - (t0 + lb) <= 10.0 + 1e-12:
                break
        return best_c, best_k

    @njit('void(int64[::1], int64, int64, int64, int64[::1], int64[::1])', cache=False, nogil=True)
    def _relocate(x, lo, hi, j, tmp, blk):
        """Remove x[lo:hi] and insert it at index j of the remaining list (the list operations del x[lo:hi];
        x[j:j] = blk)."""
        n = x.shape[0]
        L = hi - lo
        for q in range(L):
            blk[q] = x[lo + q]
        w = 0
        for q in range(n):
            if q < lo or q >= hi:
                tmp[w] = x[q]
                w += 1
        for q in range(j):
            x[q] = tmp[q]
        for q in range(L):
            x[j + q] = blk[q]
        for q in range(j, n - L):
            x[q + L] = tmp[q]

    @njit('int64(float64[:, ::1], float64[::1], float64[::1], float64[::1], int64[::1], int64[::1], int64[::1], int64[::1], '
          'int64[::1], int64[::1], int64[::1], float64[::1], int64, float64, int64, int64, uint64[::1], float64[::1], '
          'int64[::1], float64[:, ::1], float64[:, ::1], int64[::1], int64[::1], int64)', cache=False, nogil=True)
    def _order_chunk(cm, t_img, t_recon, flat, offs, steps, cur, cand, best, ba, bb, wcum, kmax, T, stall, cap, rng, fs,
                     ist, snap_a, snap_b, tmp, blk, n_iter):
        """Up to n_iter iterations of polish_order. fs = [cur_s, best_s]; ist = [k, last, it]. Returns 1 when the
        stall or cap rule stopped the polish, else 0."""
        n = cur.shape[0]
        nb = ba.shape[0]
        k, last, it = ist[0], ist[1], ist[2]
        cur_s, best_s = fs[0], fs[1]
        stop = 0
        end = it + n_iter
        if end > cap:
            end = cap
        while it < end:
            if it - last > stall:
                stop = 1
                break
            r = _u01(rng) * wcum[nb - 1]  # rng.choices(blocks, weights): the first cumulative weight above r
            bi = 0
            while bi < nb - 1 and wcum[bi] <= r:
                bi += 1
            a, b = ba[bi], bb[bi]
            for q in range(n):
                cand[q] = cur[q]
            i = _rr(rng, a, b)
            mv = _u01(rng)
            ok = True
            if mv < 0.35:  # insert
                j = _rr(rng, a, b)
                x = cand[i]
                if j > i:
                    for q in range(i, j):
                        cand[q] = cand[q + 1]
                else:
                    for q in range(i, j, -1):
                        cand[q] = cand[q - 1]
                cand[j] = x
            elif mv < 0.60:  # reverse
                j = _rr(rng, a, b)
                lo, hi = (i, j) if i < j else (j, i)
                while lo < hi:
                    x = cand[lo]
                    cand[lo] = cand[hi]
                    cand[hi] = x
                    lo += 1
                    hi -= 1
            elif mv < 0.72:  # swap
                j = _rr(rng, a, b)
                x = cand[i]
                cand[i] = cand[j]
                cand[j] = x
            elif mv < 0.87:  # block of 2..4
                L = 2 + _rr(rng, 0, 3)
                if i + L > b:
                    ok = False
                else:
                    j = _rr(rng, a, b - L + 1)
                    _relocate(cand, i, i + L, j, tmp, blk)
            else:  # the run of same-step FOVs around i (at most 12 each side)
                s = steps[cand[i]]
                lo = i
                while lo > a and steps[cand[lo - 1]] == s and i - lo < 12:
                    lo -= 1
                hi = i
                while hi + 1 < b and steps[cand[hi + 1]] == s and hi - i < 12:
                    hi += 1
                j = _rr(rng, a, b - (hi + 1 - lo) + 1)
                _relocate(cand, lo, hi + 1, j, tmp, blk)
            if ok:
                start = 0
                while start < n and cand[start] == cur[start]:
                    start += 1
                if start == n:  # the same order: its score is the current one
                    s_c = cur_s
                else:
                    m, _t0 = _sim_from(cm, cand, n, t_img, t_recon, flat, offs, steps, k, 5.0, snap_a, snap_b, start)
                    s_c = m + 10.0 * k
                d = s_c - cur_s
                if d < 0 or _u01(rng) < np.exp(-d / T):
                    if start < n:
                        for q in range(n):
                            cur[q] = cand[q]
                        for q in range(start, n):
                            for c in range(k):
                                snap_a[q, c] = snap_b[q, c]
                    cur_s = s_c
                    if s_c < best_s - 1e-9:
                        full, k2 = _sweep(cm, cur, n, t_img, t_recon, flat, offs, steps, kmax, snap_b)
                        for q in range(n):
                            best[q] = cur[q]
                        best_s = full
                        last = it
                        if k2 != k:
                            k = k2
                            cur_s = full
                            _sim_from(cm, cur, n, t_img, t_recon, flat, offs, steps, k, 5.0, snap_a, snap_a, 0)
            it += 1
        if it >= cap:
            stop = 1
        ist[0], ist[1], ist[2] = k, last, it
        fs[0], fs[1] = cur_s, best_s
        return stop

    @njit('Tuple((int64, int64, int64))(int64, int64, int64, float64[::1], int64[::1], int64[::1], int64[::1], '
          'float64[::1], float64[::1], int64[::1], int64[::1], int64[::1], float64, float64)', cache=False, nogil=True)
    def _mv(m, g, j, flat, pid, offs, side_cnt, ti, ptime, pside, where, ordst, img_plain, img_side):
        """polish_members' move(m, g, j): part m leaves its FOV a (index i there) for FOV g at slot j (j < 0: the
        longest-first slot, ties by part index), shifting the job arrays as the Python move does; an emptied FOV
        leaves the visit order. ordst = [nord, order...]. Returns (a, i, dropped position or -1)."""
        a = where[m]
        i = 0
        while pid[offs[a] + i] != m:
            i += 1
        if j < 0:
            j = 0
            n_g = offs[g + 1] - offs[g]
            while j < n_g:
                x = pid[offs[g] + j]
                if ptime[x] < ptime[m] or (ptime[x] == ptime[m] and x > m):
                    break
                j += 1
        ps = offs[a] + i
        val, vid = flat[ps], pid[ps]
        if a < g:
            ins = offs[g] - 1 + j
            for q in range(ps, ins):
                flat[q] = flat[q + 1]
                pid[q] = pid[q + 1]
            flat[ins] = val
            pid[ins] = vid
            for q in range(a + 1, g + 1):
                offs[q] -= 1
        else:
            ins = offs[g] + j
            for q in range(ps, ins, -1):
                flat[q] = flat[q - 1]
                pid[q] = pid[q - 1]
            flat[ins] = val
            pid[ins] = vid
            for q in range(g + 1, a + 1):
                offs[q] += 1
        if pside[m] == 1:
            side_cnt[a] -= 1
            side_cnt[g] += 1
        ti[a] = img_side if side_cnt[a] > 0 else img_plain
        ti[g] = img_side if side_cnt[g] > 0 else img_plain
        where[m] = g
        dropped = -1
        if offs[a + 1] == offs[a]:
            nord = ordst[0]
            p = 0
            while ordst[1 + p] != a:
                p += 1
            for q in range(p, nord - 1):
                ordst[1 + q] = ordst[2 + q]
            ordst[0] = nord - 1
            dropped = p
        return a, i, dropped

    @njit('int64(float64[:, ::1], float64[::1], float64[::1], float64[::1], int64[::1], int64[::1], int64[::1], int64[::1], '
          'float64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], float64, float64, int64, '
          'float64, int64, int64, uint64[::1], float64[::1], int64[::1], float64[:, ::1], float64[:, ::1], int64[::1], '
          'int64[::1], int64[::1], int64[::1], int64)', cache=False, nogil=True)
    def _members_chunk(cm, ti, t_recon, flat, pid, offs, steps, side_cnt, ptime, pside, where, ordst, cps, cst, cfv,
                       img_plain, img_side, kmax, T, stall, cap, rng, fs, ist, snap_a, snap_b, gbuf, best_pid, best_offs,
                       best_ord, n_iter):
        """Up to n_iter iterations of polish_members. fs = [cur_s, best_s]; ist = [k, last, it, found]; ordst =
        [nord, order...] (the live visit order); best_ord = [nord, order...] of the best plan. Returns 1 when the stall
        or cap rule stopped the polish, else 0."""
        k, last, it = ist[0], ist[1], ist[2]
        cur_s, best_s = fs[0], fs[1]
        order = ordst[1:]
        ncand = cps.shape[0]
        stop = 0
        end = it + n_iter
        if end > cap:
            end = cap
        while it < end:
            if it - last > stall:
                stop = 1
                break
            ci = _rr(rng, 0, ncand)
            m = cps[ci]
            a = where[m]
            ng = 0
            for q in range(cst[ci], cst[ci + 1]):
                g = cfv[q]
                if g != a and offs[g + 1] > offs[g]:
                    gbuf[ng] = g
                    ng += 1
            if ng == 0:
                it += 1
                continue
            g = gbuf[_rr(rng, 0, ng)]
            nord = ordst[0]
            pa, pg = -1, -1
            for q in range(nord):
                if order[q] == a:
                    pa = q
                if order[q] == g:
                    pg = q
            start = pa if pa < pg else pg
            a, i, dropped = _mv(m, g, -1, flat, pid, offs, side_cnt, ti, ptime, pside, where, ordst, img_plain, img_side)
            nord = ordst[0]
            m_ct, _t0 = _sim_from(cm, order, nord, ti, t_recon, flat, offs, steps, k, 5.0, snap_a, snap_b, start)
            s_c = m_ct + 10.0 * k
            d = s_c - cur_s
            if d < 0 or _u01(rng) < np.exp(-d / T):
                for q in range(start, nord):
                    for c in range(k):
                        snap_a[q, c] = snap_b[q, c]
                cur_s = s_c
                if s_c < best_s - 1e-9:
                    full, k2 = _sweep(cm, order, nord, ti, t_recon, flat, offs, steps, kmax, snap_b)
                    best_s = full
                    last = it
                    for q in range(pid.shape[0]):
                        best_pid[q] = pid[q]
                    for q in range(offs.shape[0]):
                        best_offs[q] = offs[q]
                    for q in range(nord + 1):
                        best_ord[q] = ordst[q]
                    ist[3] = 1
                    if k2 != k:
                        k = k2
                        cur_s = full
                        _sim_from(cm, order, nord, ti, t_recon, flat, offs, steps, k, 5.0, snap_a, snap_a, 0)
            else:
                if dropped >= 0:
                    for q in range(nord, dropped, -1):
                        ordst[1 + q] = ordst[q]
                    ordst[1 + dropped] = a
                    ordst[0] = nord + 1
                _mv(m, a, i, flat, pid, offs, side_cnt, ti, ptime, pside, where, ordst, img_plain, img_side)
            it += 1
        if it >= cap:
            stop = 1
        ist[0], ist[1], ist[2] = k, last, it
        fs[0], fs[1] = cur_s, best_s
        return stop

    OK = True
except Exception:  # no numba, or a compile failed: the callers keep the Python polishes
    OK = False

CHUNK = 256  # iterations between deadline checks (a compiled iteration takes a few microseconds)
LAST = {}  # the last run's best full score and iteration count per polish (for the checks)
T = 1e-3


def _seed(seed):
    return np.array([(int(seed) * 0x2545F4914F6CDD1D + 0x5851F42D4C957F2D) % (1 << 64)], dtype=np.uint64)


def _run(chunk, args, cap, deadline):
    """Call a chunk kernel until it reports a stop, the cap is reached or the deadline passes."""
    ist = args[-1]
    while True:
        if deadline is not None and time.perf_counter() > deadline:
            return
        if chunk(*args[:-1], CHUNK) or ist[2] >= cap:
            return


def order(fast, seq, blocks, weights, k, cur_s, kmax, stall, cap, deadline, seed):
    """polish_order's loop from the current order `seq` (list), its score cur_s at k. Returns (best order, its full
    score); the caller keeps its order unless that score is strictly lower."""
    _kern, cm, t_img, t_recon, flat, offs, steps = fast
    n = len(seq)
    cur = np.array(seq, dtype=np.int64)
    best = cur.copy()
    km = max(int(kmax), 4)
    snap_a, snap_b = np.empty((n, km)), np.empty((n, km))
    _sim_from(cm, cur, n, t_img, t_recon, flat, offs, steps, int(k), 5.0, snap_a, snap_a, 0)
    ba = np.array([a for a, _ in blocks], dtype=np.int64)
    bb = np.array([b for _, b in blocks], dtype=np.int64)
    wcum = np.cumsum(np.array(weights, dtype=np.float64))
    fs = np.array([cur_s, cur_s], dtype=np.float64)
    ist = np.array([int(k), 0, 0], dtype=np.int64)
    rng = _seed(seed)
    args = (cm, t_img, t_recon, flat, offs, steps, cur, np.empty(n, dtype=np.int64), best, ba, bb, wcum, km, T,
            int(stall), int(cap), rng, fs, ist, snap_a, snap_b, np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64), ist)
    _run(_order_chunk, args, int(cap), deadline)
    LAST['order'], LAST['order_it'] = float(fs[1]), int(ist[2])
    return best.tolist(), float(fs[1])


FALLBACK = object()


def members(fast, P, mem, seq, cand, cover, k, cur_s, kmax, stall, cap, deadline, seed):
    """polish_members' loop. mem: member lists per FOV index (aligned with the evaluator's job arrays), seq: the
    visit order, cand / cover: the movable parts and their covering FOVs (as polish_members builds them). Returns
    (members per FOV, visit order) when the full score is strictly lower than cur_s, None when not, FALLBACK when the
    job arrays do not match mem (the caller then runs the Python loop)."""
    _kern, cm, t_img, t_recon, flat0, offs0, steps = fast
    nf = len(mem)
    if offs0.shape[0] != nf + 1:
        return FALLBACK
    pid = np.empty(flat0.shape[0], dtype=np.int64)
    ptime = np.array(P.time, dtype=np.float64)  # a writable copy: the compiled signature takes plain arrays
    for q, ms in enumerate(mem):
        o = int(offs0[q])
        if int(offs0[q + 1]) - o != len(ms):
            return FALLBACK
        for c, x in enumerate(ms):
            if flat0[o + c] != float(P.time[x]):
                return FALLBACK
            pid[o + c] = x
    flat, offs, ti = flat0.copy(), offs0.copy(), np.array(t_img, dtype=np.float64)
    pside = (np.asarray(P.side) == 1).astype(np.int64)
    side_cnt = np.array([sum(1 for x in ms if P.side[x] == 1) for ms in mem], dtype=np.int64)
    where = np.full(len(ptime), -1, dtype=np.int64)
    for q, ms in enumerate(mem):
        for x in ms:
            where[x] = q
    n = len(seq)
    ordst = np.zeros(n + 1, dtype=np.int64)
    ordst[0] = n
    ordst[1:] = seq
    cps = np.array(cand, dtype=np.int64)
    cst = np.zeros(len(cand) + 1, dtype=np.int64)
    cfv_l = []
    for c, m in enumerate(cand):
        cfv_l.extend(cover[m])
        cst[c + 1] = len(cfv_l)
    cfv = np.array(cfv_l, dtype=np.int64)
    gbuf = np.empty(max(int(np.diff(cst).max()), 1), dtype=np.int64)
    km = max(int(kmax), 4)
    snap_a, snap_b = np.empty((n, km)), np.empty((n, km))
    _sim_from(cm, ordst[1:], n, ti, t_recon, flat, offs, steps, int(k), 5.0, snap_a, snap_a, 0)
    img_plain = float(P.capture)
    img_side = float(max(P.capture, P.side_capture))
    fs = np.array([cur_s, cur_s], dtype=np.float64)
    ist = np.array([int(k), 0, 0, 0], dtype=np.int64)
    best_pid, best_offs, best_ord = pid.copy(), offs.copy(), ordst.copy()
    args = (cm, ti, t_recon, flat, pid, offs, steps, side_cnt, ptime, pside, where, ordst, cps, cst, cfv, img_plain,
            img_side, km, T, int(stall), int(cap), _seed(seed), fs, ist, snap_a, snap_b, gbuf, best_pid, best_offs,
            best_ord, ist)
    _run(_members_chunk, args, int(cap), deadline)
    LAST['members'], LAST['members_it'] = float(fs[1]), int(ist[2])
    if not ist[3] or not fs[1] < cur_s - 1e-9:
        return None
    mem2 = [best_pid[best_offs[q]:best_offs[q + 1]].tolist() for q in range(nf)]
    return mem2, best_ord[1:1 + int(best_ord[0])].tolist()
