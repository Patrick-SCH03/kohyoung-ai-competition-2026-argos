"""Pair re-cover polish of a finished camera-bound plan: two neighbouring FOVs are re-split together, members and centres.

For two free FOVs of the same step and type (visit-order neighbours, and up to NEIGHBOURS spatial neighbours within REACH
FOV sizes; no fiducial, no big-part tile), the members are pooled (at most MAX_PARTS) and re-split by the canonical
centres (the x/y ends of the members' feasible-centre rectangles): a centre covers group A, the rest U needs a common
rectangle (U empty: the pair merges into one FOV). Centres are placed in the margin-shrunk common rectangles to keep the
travel of the two slots short, both slot assignments tried; a part both centres cover goes to the earlier slot; members
are ordered longest job first. The TOP splits with the shortest travel (and a merge) are simulated from the first changed
visit position at the current core count k*, resumed from the stored core states (the same operations in the same order
as the simulator, so the same floats); a strictly lower CT(k*) + 10 k* is kept, then k* is re-swept and the states are
rebuilt. Only plans whose camera finishes after the inspection floor are searched (the camera-bound test).
The pairs are visited in the order of a logistic score of cheap pair features (fitted offline on generated development
boards; the constants below); a pair that found no improving move is skipped until a move is kept within visit-order
distance 2 of its slots; the search stops after half a round of pairs without a kept move.
The search runs on copies; recover() returns new FOV objects and the visit order, and the caller keeps them only if they
verify and their exact score (ct_sim.evaluate_sweep) is strictly lower. Deterministic caps on visited pairs and
simulations; the wall-clock cap is a safety stop. OK is False when numba is missing or a compile fails: the caller skips
the stage."""
import time

import numpy as np

try:
    from .geometry import FOV
except ImportError:  # loaded as a plain module
    from geometry import FOV

NEIGHBOURS, REACH, TOP, PLACE_TOP, MAX_PARTS = 6, 1.5, 3, 8, 64
MARGIN, TOL, Z_DELAY = 1e-3, 1e-6, 5.0
# pair score: features (consecutive, centre distance in FOV sizes, travel of the legs around both slots, that travel
# over the median leg, pooled members, side capture, visit position, failed visits = 0 when the list is built)
G_MU = (0.2687431623656606, 0.7986985090772093, 0.6325409814466524, 3.8754450196982537, 27.523601904884483,
        0.9904594890276079, 0.44196599218979393, 0.5586266812536199)
G_SD = (0.44330607377675835, 0.4525438151699134, 0.3689542858912936, 1.022452771104128, 11.933737603323557,
        0.09720848534350134, 0.2724008036399732, 0.7809858730272907)
G_W = (0.13594535488115583, -0.40464984649517455, -0.5217294907063034, 0.19375477678066166, 0.08896511119153032,
       -0.06554557177517607, -0.16597814107118616, -0.9230932836591629)
G_B = -3.3468080848573263
QUIET_FRAC = 0.5

OK = False
try:
    from numba import njit

    @njit('float64(float64, float64, float64)', cache=False, nogil=True)
    def _axt(d, v, a):
        """ct_sim.axis_time for one distance (the same operations)."""
        th = v * v / a
        if d >= th:
            return 2.0 * v / a + (d - th) / v
        return 2.0 * np.sqrt(d / a)

    @njit('float64(float64, float64, float64, float64, float64, float64, float64, float64)', cache=False, nogil=True)
    def _tt(x0, y0, x1, y1, vx, ax, vy, ay):
        """ct_sim.cost_map's entry: max of the two axis times."""
        tx = _axt(abs(x0 - x1), vx, ax)
        ty = _axt(abs(y0 - y1), vy, ay)
        return tx if tx >= ty else ty

    @njit('UniTuple(float64, 2)(float64[::1], int64, float64[::1], float64[::1], float64[::1], int64[::1], int64[::1], '
          'int64, float64, float64[:, ::1], float64[:, ::1], int64)', cache=False, nogil=True)
    def _sim(legs, n, t_img, t_recon, flat, offs, steps, k, z_delay, snap_in, snap_out, start):
        """polish_nb._sim_from on a plan held in visit order (legs[i] = travel time from position i-1 to i): resumed at
        `start` from snap_in[start]; snap_out[i] receives the core state before position i for i >= start."""
        if k < 4 or n < 1 or snap_out.shape[0] < n or snap_out.shape[1] < k or (
                start > 0 and (start >= snap_in.shape[0] or snap_in.shape[1] < k)):
            raise ValueError('recover: table shapes')
        tl = np.empty(k)
        if start == 0:
            for c in range(k):
                tl[c] = t_img[0]
        else:
            for c in range(k):
                tl[c] = snap_in[start, c]
        for i in range(start, n):
            for c in range(k):
                snap_out[i, c] = tl[c]
            if i:
                t0 = tl[0] + legs[i]
                if steps[i - 1] != steps[i]:
                    t0 += z_delay
                t0 += t_img[i]
                tl[0] = t0
                for c in range(1, k):
                    if tl[c] < t0:
                        tl[c] = t0
            rc = 1 if tl[1] <= tl[2] else 2
            tl[rc] += t_recon[i]
            ready = tl[rc]
            for q in range(offs[i], offs[i + 1]):
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

    @njit('Tuple((float64, int64))(float64[::1], int64, float64[::1], float64[::1], float64[::1], int64[::1], '
          'int64[::1], int64, float64[:, ::1])', cache=False, nogil=True)
    def _sweep(legs, n, t_img, t_recon, flat, offs, steps, kmax, scratch):
        """polish_nb._sweep (ct_sim._sweep_core without a hint, beta 10) on a plan held in visit order."""
        last = n - 1
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
            m, t0 = _sim(legs, n, t_img, t_recon, flat, offs, steps, k, 5.0, scratch, scratch, 0)
            c = m + 10.0 * k
            if c < best_c:
                best_c = c
                best_k = k
            if m - (t0 + lb) <= 10.0 + 1e-12:
                break
        return best_c, best_k

    @njit('UniTuple(float64, 2)(float64, float64)', cache=False, nogil=True)
    def _shrink(lo, hi):
        if hi - lo > 2 * MARGIN:
            return lo + MARGIN, hi - MARGIN
        mid = (lo + hi) / 2.0
        return mid, mid

    _DB = 0x03F79D71B4CB0A89  # a de Bruijn sequence B(2, 6): lowest set bit -> its index in one multiply
    _DB_TABLE = np.zeros(64, dtype=np.int64)
    for _i in range(64):
        _DB_TABLE[(((1 << _i) * _DB) & 0xFFFFFFFFFFFFFFFF) >> 58] = _i
    assert sorted(_DB_TABLE.tolist()) == list(range(64))
    _DB_U = np.uint64(_DB)

    @njit('int64(uint64)', cache=False, nogil=True)
    def _ctz(v):
        """Index of the lowest set bit of v (v != 0)."""
        return _DB_TABLE[np.int64(((v & (~v + np.uint64(1))) * _DB_U) >> np.uint64(58))]

    @njit('float64[::1](float64[::1], float64[::1], float64[::1], float64[::1], uint64, int64)', cache=False,
          nogil=True)
    def _box(lx, hx, ly, hy, mask, m):
        """Common rectangle of the parts in mask: x0, x1, y0, y1 (unshrunk)."""
        out = np.empty(4)
        out[0], out[1], out[2], out[3] = -np.inf, np.inf, -np.inf, np.inf
        for p in range(m):
            if (mask >> np.uint64(p)) & np.uint64(1):
                if lx[p] > out[0]:
                    out[0] = lx[p]
                if hx[p] < out[1]:
                    out[1] = hx[p]
                if ly[p] > out[2]:
                    out[2] = ly[p]
                if hy[p] < out[3]:
                    out[3] = hy[p]
        return out

    @njit('int64(float64[::1], float64[::1], float64[::1], float64[::1], float64[:, ::1])', cache=False, nogil=True)
    def _points(bx, nbx, nby, nbf, pts):
        """Candidate centres in the shrunk box bx (x0, x1, y0, y1): corners, middle, the neighbours' projections."""
        x0, x1, y0, y1 = bx[0], bx[1], bx[2], bx[3]
        c = 0
        for a in (x0, x1):
            for b in (y0, y1):
                pts[c, 0], pts[c, 1] = a, b
                c += 1
        pts[c, 0], pts[c, 1] = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        c += 1
        for q in range(nbx.shape[0]):
            if nbf[q] > 0.5:
                px = nbx[q] if nbx[q] > x0 else x0
                px = px if px < x1 else x1
                py = nby[q] if nby[q] > y0 else y0
                py = py if py < y1 else y1
                pts[c, 0], pts[c, 1] = px, py
                c += 1
        return c

    @njit('Tuple((int64, int64))(float64[::1], float64[::1], float64[::1], float64[::1], int64, float64[::1], '
          'float64[::1], int64, float64, float64, float64, float64, float64[:, ::1], uint64[:, ::1], float64[::1], '
          'float64[::1])', cache=False, nogil=True)
    def _splits(lx, hx, ly, hy, m, nb, nbf, consecutive, vx, ax, vy, ay, out_c, out_g, out_t, merge_box):
        """Re-splits of the pooled parts (rectangles lx..hy, m <= 64) of a pair at slots i < j; nb = (prev_i, next_i,
        prev_j, next_j) as x, y pairs with presence flags nbf. Writes up to out_c.shape[0] splits with the shortest
        travel (centres c1 at slot i, c2 at slot j; member masks; travel) and returns (count, merge flag); a merge
        writes the shrunk common rectangle of all parts into merge_box."""
        top = out_c.shape[0]
        full = np.uint64(0xFFFFFFFFFFFFFFFF) if m == 64 else (np.uint64(1) << np.uint64(m)) - np.uint64(1)
        xs = np.unique(np.concatenate((lx[:m], hx[:m])))
        ys = np.unique(np.concatenate((ly[:m], hy[:m])))
        # each part's bit in five orders: its own index, and by bound so that the lowest set bit of a set's mask is its
        # extreme: lx descending, hx ascending, ly descending, hy ascending (a set's common rectangle in O(1))
        rk = np.empty((5, m), dtype=np.int64)
        vals = np.empty((4, m))
        for p in range(m):
            rk[0, p] = p
        for o in range(4):
            if o == 0:
                srt = np.argsort(-lx[:m], kind='mergesort')
            elif o == 1:
                srt = np.argsort(hx[:m], kind='mergesort')
            elif o == 2:
                srt = np.argsort(-ly[:m], kind='mergesort')
            else:
                srt = np.argsort(hy[:m], kind='mergesort')
            for r_ in range(m):
                p = srt[r_]
                rk[o + 1, p] = r_
                vals[o, r_] = lx[p] if o == 0 else (hx[p] if o == 1 else (ly[p] if o == 2 else hy[p]))
        xm = np.zeros((5, xs.shape[0]), dtype=np.uint64)
        ym = np.zeros((5, ys.shape[0]), dtype=np.uint64)
        for a in range(xs.shape[0]):
            for p in range(m):
                if lx[p] - TOL <= xs[a] and xs[a] <= hx[p] + TOL:
                    for o in range(5):
                        xm[o, a] |= np.uint64(1) << np.uint64(rk[o, p])
        for b in range(ys.shape[0]):
            for p in range(m):
                if ly[p] - TOL <= ys[b] and ys[b] <= hy[p] + TOL:
                    for o in range(5):
                        ym[o, b] |= np.uint64(1) << np.uint64(rk[o, p])
        npt = xs.shape[0] * ys.shape[0]
        raw = np.empty(npt, dtype=np.uint64)
        src = np.empty((npt, 2), dtype=np.int64)
        r = 0
        for a in range(xs.shape[0]):
            for b in range(ys.shape[0]):
                v = xm[0, a] & ym[0, b]
                if v != np.uint64(0):
                    raw[r] = v
                    src[r, 0], src[r, 1] = a, b
                    r += 1
        srt = np.argsort(raw[:r], kind='mergesort')
        nmask = 0
        for q in range(r):
            if q == 0 or raw[srt[q]] != raw[srt[q - 1]]:
                nmask += 1
        masks = np.empty(nmask, dtype=np.uint64)
        first = np.empty(nmask, dtype=np.int64)
        u_ = -1
        for q in range(r):
            if q == 0 or raw[srt[q]] != raw[srt[q - 1]]:
                u_ += 1
                masks[u_] = raw[srt[q]]
                first[u_] = srt[q]
        merge = 0
        # proxy pass: box centres, best of the two slot assignments
        proxy = np.full(nmask, np.inf)
        boxes = np.empty((nmask, 8))
        nbx = np.array([nb[0], nb[2], nb[4], nb[6]])
        nby = np.array([nb[1], nb[3], nb[5], nb[7]])
        ba = np.empty(4)
        bu = np.empty(4)
        for t in range(nmask):
            A = masks[t]
            U = full & ~A
            a_, b_ = src[first[t], 0], src[first[t], 1]
            for o in range(4):
                ba[o] = vals[o, _ctz(xm[o + 1, a_] & ym[o + 1, b_])]
            if U == np.uint64(0):
                x0, x1 = _shrink(ba[0], ba[1])
                y0, y1 = _shrink(ba[2], ba[3])
                merge_box[0], merge_box[1], merge_box[2], merge_box[3] = x0, x1, y0, y1
                merge = 1
                continue
            for o in range(4):
                bu[o] = vals[o, _ctz(full & ~(xm[o + 1, a_] & ym[o + 1, b_]))]
            if bu[0] > bu[1] + TOL or bu[2] > bu[3] + TOL:
                continue
            ax0, ax1 = _shrink(ba[0], ba[1])
            ay0, ay1 = _shrink(ba[2], ba[3])
            ux0, ux1 = _shrink(bu[0], bu[1])
            uy0, uy1 = _shrink(bu[2], bu[3])
            boxes[t, 0], boxes[t, 1], boxes[t, 2], boxes[t, 3] = ax0, ax1, ay0, ay1
            boxes[t, 4], boxes[t, 5], boxes[t, 6], boxes[t, 7] = ux0, ux1, uy0, uy1
            ca = ((ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0)
            cu = ((ux0 + ux1) / 2.0, (uy0 + uy1) / 2.0)
            best = np.inf
            for sw in range(2):
                c1 = ca if sw == 0 else cu
                c2 = cu if sw == 0 else ca
                t1 = 0.0
                if nbf[0] > 0.5:
                    t1 += _tt(nb[0], nb[1], c1[0], c1[1], vx, ax, vy, ay)
                if consecutive:
                    t1 += _tt(c1[0], c1[1], c2[0], c2[1], vx, ax, vy, ay)
                else:
                    if nbf[1] > 0.5:
                        t1 += _tt(c1[0], c1[1], nb[2], nb[3], vx, ax, vy, ay)
                    if nbf[2] > 0.5:
                        t1 += _tt(nb[4], nb[5], c2[0], c2[1], vx, ax, vy, ay)
                if nbf[3] > 0.5:
                    t1 += _tt(c2[0], c2[1], nb[6], nb[7], vx, ax, vy, ay)
                if t1 < best:
                    best = t1
            proxy[t] = best
        order = np.argsort(proxy, kind='mergesort')
        pts1 = np.empty((9, 2))
        pts2 = np.empty((10, 2))
        bx = np.empty(4)
        cnt = 0
        keys = np.empty((top, 2), dtype=np.uint64)
        for oi in range(min(PLACE_TOP, nmask)):
            t = order[oi]
            if not np.isfinite(proxy[t]):
                break
            A = masks[t]
            U = full & ~A
            for sw in range(2):
                g1 = A if sw == 0 else U
                g2 = U if sw == 0 else A
                o1 = 0 if sw == 0 else 4
                o2 = 4 if sw == 0 else 0
                bx[0], bx[1], bx[2], bx[3] = boxes[t, o1], boxes[t, o1 + 1], boxes[t, o1 + 2], boxes[t, o1 + 3]
                n1 = _points(bx, nbx, nby, nbf, pts1)
                best_t, b1x, b1y, b2x, b2y = np.inf, 0.0, 0.0, 0.0, 0.0
                for u in range(n1):
                    c1x, c1y = pts1[u, 0], pts1[u, 1]
                    bx[0], bx[1], bx[2], bx[3] = boxes[t, o2], boxes[t, o2 + 1], boxes[t, o2 + 2], boxes[t, o2 + 3]
                    n2 = _points(bx, nbx, nby, nbf, pts2)
                    x0, x1, y0, y1 = bx[0], bx[1], bx[2], bx[3]
                    px = c1x if c1x > x0 else x0
                    px = px if px < x1 else x1
                    py = c1y if c1y > y0 else y0
                    py = py if py < y1 else y1
                    pts2[n2, 0], pts2[n2, 1] = px, py
                    n2 += 1
                    for w in range(n2):
                        c2x, c2y = pts2[w, 0], pts2[w, 1]
                        tt = 0.0
                        if nbf[0] > 0.5:
                            tt += _tt(nb[0], nb[1], c1x, c1y, vx, ax, vy, ay)
                        if consecutive:
                            tt += _tt(c1x, c1y, c2x, c2y, vx, ax, vy, ay)
                        else:
                            if nbf[1] > 0.5:
                                tt += _tt(c1x, c1y, nb[2], nb[3], vx, ax, vy, ay)
                            if nbf[2] > 0.5:
                                tt += _tt(nb[4], nb[5], c2x, c2y, vx, ax, vy, ay)
                        if nbf[3] > 0.5:
                            tt += _tt(c2x, c2y, nb[6], nb[7], vx, ax, vy, ay)
                        if tt < best_t:
                            best_t, b1x, b1y, b2x, b2y = tt, c1x, c1y, c2x, c2y
                # a part of group 2 that the first centre also covers goes to the first (earlier) slot
                f1, f2 = g1, g2
                for p in range(m):
                    bit = np.uint64(1) << np.uint64(p)
                    if (f2 & bit) and lx[p] - TOL <= b1x and b1x <= hx[p] + TOL and ly[p] - TOL <= b1y \
                            and b1y <= hy[p] + TOL:
                        f2 &= ~bit
                        f1 |= bit
                if f2 == np.uint64(0):
                    continue
                dup = False
                for q in range(cnt):
                    if keys[q, 0] == f1 and keys[q, 1] == f2:
                        dup = True
                        break
                if dup:
                    continue
                # keep the `top` shortest-travel splits (insertion into the sorted output)
                pos = cnt
                while pos > 0 and out_t[pos - 1] > best_t:
                    pos -= 1
                if pos >= top:
                    continue
                last = cnt if cnt < top else top - 1
                for q in range(last, pos, -1):
                    out_t[q] = out_t[q - 1]
                    out_c[q, 0], out_c[q, 1], out_c[q, 2], out_c[q, 3] = out_c[q - 1, 0], out_c[q - 1, 1], \
                        out_c[q - 1, 2], out_c[q - 1, 3]
                    out_g[q, 0], out_g[q, 1] = out_g[q - 1, 0], out_g[q - 1, 1]
                    keys[q, 0], keys[q, 1] = keys[q - 1, 0], keys[q - 1, 1]
                out_t[pos] = best_t
                out_c[pos, 0], out_c[pos, 1], out_c[pos, 2], out_c[pos, 3] = b1x, b1y, b2x, b2y
                out_g[pos, 0], out_g[pos, 1] = f1, f2
                keys[pos, 0], keys[pos, 1] = f1, f2
                if cnt < top:
                    cnt += 1
        return cnt, merge

    @njit('float64(float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], int64[::1], '
          'int64[::1], int64[::1], int64, float64[::1], int64[::1], int64[::1], int64[::1], uint64, uint64, float64, '
          'float64, float64, float64, int64, int64, int64, int64, float64[:, ::1], float64[:, ::1], float64, float64, '
          'float64, float64, float64, float64, float64[::1], int64[::1])', cache=False, nogil=True)
    def _eval_split(legs, x, y, t_img, t_recon, flat, offs, steps, types, n, ptime, pside, ptype, M, g1, g2, c1x, c1y,
                    c2x, c2y, si, sj, k, kfull, snap, scratch, capture, side_capture, vx, ax, vy, ay, flat_buf,
                    offs_buf):
        """recover._evaluate's split path in one call: the two re-split FOVs (members longest job first, ties by part
        index) at slots si < sj, the type-order test (inf when broken), the changed legs and capture times (restored
        before returning), the members' durations re-laid into flat_buf/offs_buf, then the simulation from si at k
        resumed from snap (or, kfull > 0, the full core sweep up to kfull). Returns CT(k) + 10 k (or the sweep cost)."""
        m = M.shape[0]
        # members of each group, longest job first, ties by part index
        n1 = 0
        n2 = 0
        for p in range(m):
            if (g1 >> np.uint64(p)) & np.uint64(1):
                n1 += 1
            elif (g2 >> np.uint64(p)) & np.uint64(1):
                n2 += 1
        a1 = np.empty(n1, dtype=np.int64)
        a2 = np.empty(n2, dtype=np.int64)
        i1 = 0
        i2 = 0
        for p in range(m):
            if (g1 >> np.uint64(p)) & np.uint64(1):
                a1[i1] = M[p]
                i1 += 1
            elif (g2 >> np.uint64(p)) & np.uint64(1):
                a2[i2] = M[p]
                i2 += 1
        a1 = np.sort(a1)
        a2 = np.sort(a2)
        a1 = a1[np.argsort(-ptime[a1], kind='mergesort')]
        a2 = a2[np.argsort(-ptime[a2], kind='mergesort')]
        t1 = 0
        s1 = 0
        for q in range(n1):
            if ptype[a1[q]] > t1:
                t1 = ptype[a1[q]]
            if pside[a1[q]] == 1:
                s1 = 1
        t2 = 0
        s2 = 0
        for q in range(n2):
            if ptype[a2[q]] > t2:
                t2 = ptype[a2[q]]
            if pside[a2[q]] == 1:
                s2 = 1
        # the visit order's types must stay non-increasing around both slots
        if si > 0 and types[si - 1] < t1:
            return np.inf
        nxt = t2 if sj == si + 1 else types[si + 1]
        if t1 < nxt:
            return np.inf
        if sj > si + 1 and types[sj - 1] < t2:
            return np.inf
        if sj + 1 < n and t2 < types[sj + 1]:
            return np.inf
        # durations re-laid: [..si) g1 (si, sj) g2 (sj..]
        w = 0
        for q in range(offs[si]):
            flat_buf[w] = flat[q]
            w += 1
        offs_buf[0] = 0
        for i in range(si):
            offs_buf[i + 1] = offs[i + 1]
        for q in range(n1):
            flat_buf[w] = ptime[a1[q]]
            w += 1
        offs_buf[si + 1] = w
        for i in range(si + 1, sj):
            for q in range(offs[i], offs[i + 1]):
                flat_buf[w] = flat[q]
                w += 1
            offs_buf[i + 1] = w
        for q in range(n2):
            flat_buf[w] = ptime[a2[q]]
            w += 1
        offs_buf[sj + 1] = w
        for i in range(sj + 1, n):
            for q in range(offs[i], offs[i + 1]):
                flat_buf[w] = flat[q]
                w += 1
            offs_buf[i + 1] = w
        s_li = legs[si]
        s_li1 = legs[si + 1] if si + 1 < n else 0.0
        s_lj = legs[sj]
        s_lj1 = legs[sj + 1] if sj + 1 < n else 0.0
        s_ii = t_img[si]
        s_ij = t_img[sj]
        if si > 0:
            legs[si] = _tt(x[si - 1], y[si - 1], c1x, c1y, vx, ax, vy, ay)
        if sj == si + 1:
            legs[sj] = _tt(c1x, c1y, c2x, c2y, vx, ax, vy, ay)
        else:
            legs[si + 1] = _tt(c1x, c1y, x[si + 1], y[si + 1], vx, ax, vy, ay)
            legs[sj] = _tt(x[sj - 1], y[sj - 1], c2x, c2y, vx, ax, vy, ay)
        if sj + 1 < n:
            legs[sj + 1] = _tt(c2x, c2y, x[sj + 1], y[sj + 1], vx, ax, vy, ay)
        big = capture if capture > side_capture else side_capture
        t_img[si] = big if s1 == 1 else capture
        t_img[sj] = big if s2 == 1 else capture
        if kfull > 0:
            cost, _k = _sweep(legs, n, t_img, t_recon, flat_buf, offs_buf, steps, kfull, scratch)
        else:
            ctk, _c0 = _sim(legs, n, t_img, t_recon, flat_buf, offs_buf, steps, k, 5.0, snap, scratch, si)
            cost = ctk + 10.0 * k
        legs[si] = s_li
        if si + 1 < n:
            legs[si + 1] = s_li1
        legs[sj] = s_lj
        if sj + 1 < n:
            legs[sj + 1] = s_lj1
        t_img[si] = s_ii
        t_img[sj] = s_ij
        return cost

    OK = True
except Exception:  # numba missing or a compile failed: the caller skips the stage
    OK = False


def _free(P, f):
    return bool(f.members) and not f.exclusive and not f.fixed and not any(P.big[m] or P.type[m] == 2 for m in f.members)


def _type(P, mem):
    return int(max(P.type[m] for m in mem)) if mem else 0


def _lpt(P, mem):
    return sorted((int(m) for m in mem), key=lambda m: (-P.time[m], m))


class _State:
    """The plan in visit order with the simulator tables (copies; the caller's FOVs are never touched)."""

    def __init__(self, P, fovs, seq):
        self.P = P
        order = [fovs[q] for q in seq]
        self.x = np.array([f.x for f in order], dtype=np.float64)
        self.y = np.array([f.y for f in order], dtype=np.float64)
        self.mem = [list(int(m) for m in f.members) for f in order]
        self.step = np.array([int(f.step) for f in order], dtype=np.int64)
        self.fixed = [bool(f.fixed) for f in order]
        self.excl = [bool(f.exclusive) for f in order]
        self.free = [_free(P, f) for f in order]
        self.ptime = np.array(P.time, dtype=np.float64)  # writable copies (pandas may hand out read-only arrays)
        self.pside = np.array(P.side, dtype=np.int64)
        self.ptype = np.array(P.type, dtype=np.int64)
        self.rebuild()

    @property
    def n(self):
        return len(self.mem)

    def img(self, mem):
        P = self.P
        return max(P.capture, P.side_capture) if any(P.side[m] == 1 for m in mem) else P.capture

    def leg(self, x0, y0, x1, y1):
        P = self.P
        return _tt(x0, y0, x1, y1, P.vx, P.ax, P.vy, P.ay)

    def rebuild(self):
        P, n = self.P, self.n
        self.legs = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            self.legs[i] = self.leg(self.x[i - 1], self.y[i - 1], self.x[i], self.y[i])
        self.t_img = np.array([self.img(m) for m in self.mem], dtype=np.float64)
        self.t_recon = np.full(n, P.recon, dtype=np.float64)
        self.sizes = np.array([len(m) for m in self.mem], dtype=np.int64)
        self.offs = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(self.sizes, out=self.offs[1:])
        self.flat = np.array([P.time[q] for m in self.mem for q in m], dtype=np.float64)
        self.types = [_type(P, m) for m in self.mem]
        self.types_arr = np.array(self.types, dtype=np.int64)
        self.flat_buf = np.empty(self.flat.shape[0], dtype=np.float64)
        self.offs_buf = np.empty(n + 1, dtype=np.int64)

    def fovs(self):
        return [FOV(self.x[i], self.y[i], self.mem[i], int(self.step[i]), fixed=self.fixed[i], exclusive=self.excl[i])
                for i in range(self.n)]


def _pairs(st):
    P, n = st.P, st.n
    key = [(int(st.step[i]), st.types[i]) for i in range(n)]
    out = set()
    for i in range(n - 1):
        if st.free[i] and st.free[i + 1] and key[i] == key[i + 1]:
            out.add((i, i + 1))
    for i in range(n):
        if not st.free[i]:
            continue
        d = np.maximum(np.abs(st.x - st.x[i]) / P.fw, np.abs(st.y - st.y[i]) / P.fh)
        got = 0
        for j in np.argsort(d, kind='stable'):
            if got >= NEIGHBOURS or d[j] > REACH:
                break
            j = int(j)
            if j != i and st.free[j] and key[j] == key[i]:
                out.add((min(i, j), j if j > i else i))
                got += 1
    return sorted(out)


def _order(st, pl):
    """The pair list by the pair score, highest first (ties in list order)."""
    if not pl:
        return pl
    P, n = st.P, st.n
    legs = st.legs
    med = float(np.median(legs[1:])) if n > 2 else 1.0
    f = np.zeros((len(pl), len(G_W)))
    for r, (i, j) in enumerate(pl):
        around = legs[i] + (legs[i + 1] if i + 1 < n else 0.0) + legs[j] + (legs[j + 1] if j + 1 < n else 0.0)
        if j == i + 1:
            around -= legs[j]  # the shared leg once
        f[r] = (float(j == i + 1), max(abs(st.x[i] - st.x[j]) / P.fw, abs(st.y[i] - st.y[j]) / P.fh), around,
                around / max(med, 1e-9), len(st.mem[i]) + len(st.mem[j]),
                float(st.t_img[i] > P.capture or st.t_img[j] > P.capture), i / max(n - 1, 1), 0)
    score = ((f - np.array(G_MU)) / np.array(G_SD)) @ np.array(G_W) + np.array(G_B)
    idx = sorted(range(len(pl)), key=lambda r: (-score[r], r))
    return [pl[r] for r in idx]


def recover(P, fovs, seq, cfg, t_start, budget):
    """Returns (fovs, seq, stats): new FOV objects in visit order and range(n), or the inputs unchanged."""
    stats = dict(ran=False)
    if not OK or len(seq) < 3:
        stats['skip'] = 'kernels' if not OK else 'small'
        return fovs, seq, stats
    t0 = time.perf_counter()
    sims_cap = int(cfg.get('recover_sims', 800))
    pairs_cap = int(cfg.get('recover_pairs', 4000))
    max_parts = min(MAX_PARTS, int(cfg.get('recover_parts', MAX_PARTS)))
    stop_at = min(t0 + float(cfg.get('recover_ms', 150.0)) / 1000.0, t_start + 0.97 * budget)
    st = _State(P, fovs, seq)
    n = st.n
    kmax = max(4, int(P.max_core))
    snap = np.zeros((n, kmax))
    scratch = np.zeros((n, kmax))
    cost, k = _sweep(st.legs, n, st.t_img, st.t_recon, st.flat, st.offs, st.step, kmax, scratch)
    ct, c0 = _sim(st.legs, n, st.t_img, st.t_recon, st.flat, st.offs, st.step, k, Z_DELAY, scratch, snap, 0)
    load = float(st.flat.sum())
    gate = c0 >= load / max(k - 3, 1)
    stats.update(gate=bool(gate), base=float(cost), k0=int(k))
    if not gate and not cfg.get('recover_all', False):  # recover_all (tests only): search every plan
        stats['gate_ms'] = 1000 * (time.perf_counter() - t0)
        return fovs, seq, stats
    stats['ran'] = True
    hook = cfg.get('_recover_hook')  # tests only: called after every simulation and every kept move
    full_sweep = bool(cfg.get('recover_full_sweep', False))  # tests only: every candidate swept over all k
    sims = visited = accepted = merges = skipped_big = dl_skips = 0
    out_c = np.empty((TOP, 4))
    out_g = np.empty((TOP, 2), dtype=np.uint64)
    out_t = np.empty(TOP)
    merge_box = np.empty(4)
    nb = np.empty(8)
    nbf = np.empty(4)
    pl = _order(st, _pairs(st))
    kp, quiet = 0, 0
    epoch, changed, last_fail = 0, np.zeros(n, dtype=np.int64), {}  # a failed pair waits for a kept move nearby
    while (pl and quiet < max(1, int(QUIET_FRAC * len(pl))) and sims < sims_cap and visited < pairs_cap
           and time.perf_counter() < stop_at):
        si, sj = pl[kp % len(pl)]
        kp += 1
        if (si, sj) in last_fail:
            near = [q for q in (si - 1, si, si + 1, sj - 1, sj, sj + 1) if 0 <= q < n]
            if max(int(changed[q]) for q in near) <= last_fail[(si, sj)]:
                quiet += 1
                dl_skips += 1
                continue
        visited += 1
        M = np.array(sorted(set(st.mem[si]) | set(st.mem[sj])), dtype=np.int64)
        m = M.shape[0]
        if m > max_parts:
            skipped_big += 1
            quiet += 1
            continue
        lx, hx = np.ascontiguousarray(P.lx[M], dtype=np.float64), np.ascontiguousarray(P.hx[M], dtype=np.float64)
        ly, hy = np.ascontiguousarray(P.ly[M], dtype=np.float64), np.ascontiguousarray(P.hy[M], dtype=np.float64)
        cons = sj == si + 1
        for q, (s, off) in enumerate(((si, -1), (si, 1), (sj, -1), (sj, 1))):
            p = s + off
            ok = 0 <= p < n and not (cons and ((s == si and off == 1) or (s == sj and off == -1)))
            nbf[q] = 1.0 if ok else 0.0
            nb[2 * q], nb[2 * q + 1] = (st.x[p], st.y[p]) if ok else (0.0, 0.0)
        cnt, merge = _splits(lx, hx, ly, hy, m, nb, nbf, 1 if cons else 0, P.vx, P.ax, P.vy, P.ay,
                             out_c, out_g, out_t, merge_box)
        best = None
        cands = []
        if merge:
            for keep in (si, sj):
                cands.append(('merge', keep))
        for c in range(cnt):
            cands.append(('split', c))
        for kind, c in cands:
            if sims >= sims_cap:
                break
            ev = _evaluate(st, P, si, sj, kind, c, M, out_c, out_g, merge_box, k, snap, scratch,
                           kmax if full_sweep else 0)
            if ev is None:
                continue
            sims += 1
            if hook is not None:
                hook('eval', st, ev, k)
            if ev[0] < cost - 1e-9 and (best is None or ev[0] < best[0]):
                best = ev
        if best is None:
            quiet += 1
            last_fail[(si, sj)] = epoch
            continue
        _apply(st, best)
        accepted += 1
        merges += best[1] == 'merge'
        n = st.n
        if snap.shape[0] < n or n < snap.shape[0]:
            snap = np.zeros((n, kmax))
            scratch = np.zeros((n, kmax))
        cost, k = _sweep(st.legs, n, st.t_img, st.t_recon, st.flat, st.offs, st.step, kmax, scratch)
        _sim(st.legs, n, st.t_img, st.t_recon, st.flat, st.offs, st.step, k, Z_DELAY, scratch, snap, 0)
        if hook is not None:
            hook('accept', st, (cost, k), snap)
        quiet = 0
        epoch += 1
        if best[1] == 'merge':  # slots shifted: new ordered pairs from the top, no failure marks
            changed, last_fail = np.zeros(n, dtype=np.int64), {}
            pl = _order(st, _pairs(st))
            kp = 0
        else:
            for q in (si - 1, si, si + 1, sj - 1, sj, sj + 1):
                if 0 <= q < n:
                    changed[q] = epoch
    stats.update(sims=sims, visited=visited, accepted=accepted, merges=merges, skipped_big=skipped_big,
                 dl_skips=dl_skips,
                 final=float(cost), k=int(k), ms=1000 * (time.perf_counter() - t0),
                 capped=bool(sims >= sims_cap or visited >= pairs_cap), timed_out=bool(time.perf_counter() >= stop_at))
    if not accepted:
        return fovs, seq, stats
    new = st.fovs()
    return new, list(range(len(new))), stats


def _members(M, mask, m):
    return [int(M[p]) for p in range(m) if (int(mask) >> p) & 1]


def _evaluate(st, P, si, sj, kind, c, M, out_c, out_g, merge_box, k, snap, scratch, kmax_full=0):
    """Simulate one candidate from its first changed position at k; returns (cost, kind, payload) or None.
    kmax_full > 0 (tests only): the candidate's full core sweep instead (the cost is then min over k)."""
    m = M.shape[0]
    if kind == 'split':
        x1, y1, x2, y2 = (float(v) for v in out_c[c])
        g1m, g2m = out_g[c, 0], out_g[c, 1]
        cost = _eval_split(st.legs, st.x, st.y, st.t_img, st.t_recon, st.flat, st.offs, st.step, st.types_arr, st.n,
                           st.ptime, st.pside, st.ptype, M, g1m, g2m, x1, y1, x2, y2, si, sj, k, kmax_full, snap,
                           scratch, P.capture, P.side_capture, P.vx, P.ax, P.vy, P.ay, st.flat_buf, st.offs_buf)
        if not np.isfinite(cost):
            return None  # the type order would break
        return (cost, 'split', (si, sj, x1, y1, x2, y2, int(g1m), int(g2m), M))
    keep = c
    drop = sj if keep == si else si
    bx = merge_box
    n = st.n
    pk = keep - 1 if keep - 1 != drop else keep - 2
    nk = keep + 1 if keep + 1 != drop else keep + 2
    near = [(st.x[q], st.y[q]) for q in (pk, nk) if 0 <= q < n]
    pts = [(bx[0], bx[2]), (bx[0], bx[3]), (bx[1], bx[2]), (bx[1], bx[3]), ((bx[0] + bx[1]) / 2, (bx[2] + bx[3]) / 2)]
    for (px, py) in near:
        pts.append((min(max(px, bx[0]), bx[1]), min(max(py, bx[2]), bx[3])))

    def tr(p):
        return sum(st.leg(q[0], q[1], p[0], p[1]) for q in near)
    cx, cy = min(pts, key=tr)
    g = _lpt(P, M.tolist())
    keep_idx = [q for q in range(n) if q != drop]
    xs = np.array([cx if q == keep else st.x[q] for q in keep_idx], dtype=np.float64)
    ys = np.array([cy if q == keep else st.y[q] for q in keep_idx], dtype=np.float64)
    mem = [g if q == keep else st.mem[q] for q in keep_idx]
    steps = np.ascontiguousarray(st.step[keep_idx])
    n2 = n - 1
    legs = np.zeros(n2, dtype=np.float64)
    for i in range(1, n2):
        legs[i] = st.leg(xs[i - 1], ys[i - 1], xs[i], ys[i])
    img = np.array([st.img(mm) for mm in mem], dtype=np.float64)
    sizes = np.array([len(mm) for mm in mem], dtype=np.int64)
    offs = np.zeros(n2 + 1, dtype=np.int64)
    np.cumsum(sizes, out=offs[1:])
    flat = np.array([P.time[q] for mm in mem for q in mm], dtype=np.float64)
    start = min(si, sj)
    rec = st.t_recon[:n2].copy()
    if kmax_full:
        return (_sweep(legs, n2, img, rec, flat, offs, steps, kmax_full, scratch)[0], 'merge', (keep, drop, cx, cy, g))
    ctk, _ = _sim(legs, n2, img, rec, flat, offs, steps, k, Z_DELAY, snap, scratch, start)
    return (ctk + 10.0 * k, 'merge', (keep, drop, cx, cy, g))


def _replace(st, repl):
    """flat and offs with the members of the given slots replaced."""
    P = st.P
    n = st.n
    sizes = st.sizes.copy()
    for s, g in repl.items():
        sizes[s] = len(g)
    offs = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(sizes, out=offs[1:])
    parts = []
    prev = 0
    for s in sorted(repl):
        parts.append(st.flat[st.offs[prev]:st.offs[s]])
        parts.append(np.array([P.time[q] for q in repl[s]], dtype=np.float64))
        prev = s + 1
    parts.append(st.flat[st.offs[prev]:])
    return np.ascontiguousarray(np.concatenate(parts)), offs


def split_lists(P, pay):
    """The (si, sj, x1, y1, x2, y2, members at si, members at sj) of a split payload (members longest job first)."""
    si, sj, x1, y1, x2, y2, g1m, g2m, M = pay
    m = M.shape[0]
    return si, sj, x1, y1, x2, y2, _lpt(P, _members(M, g1m, m)), _lpt(P, _members(M, g2m, m))


def _apply(st, best):
    _, kind, pay = best
    if kind == 'split':
        si, sj, x1, y1, x2, y2, g1, g2 = split_lists(st.P, pay)
        st.x[si], st.y[si], st.mem[si] = x1, y1, g1
        st.x[sj], st.y[sj], st.mem[sj] = x2, y2, g2
        for s in (si, sj):
            st.fixed[s], st.excl[s], st.free[s] = False, False, True
    else:
        keep, drop, cx, cy, g = pay
        st.x[keep], st.y[keep], st.mem[keep] = cx, cy, g
        st.fixed[keep], st.excl[keep], st.free[keep] = False, False, True
        st.x = np.delete(st.x, drop)
        st.y = np.delete(st.y, drop)
        del st.mem[drop]
        st.step = np.delete(st.step, drop)
        del st.fixed[drop], st.excl[drop], st.free[drop]
    st.rebuild()
