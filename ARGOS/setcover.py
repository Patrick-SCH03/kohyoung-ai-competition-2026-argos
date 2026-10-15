"""Sparse, deterministic set-cover local search for the ordinary parts, on a snapped candidate grid.

Candidates: x values = cluster representatives of the parts' hx (sorted; a gap > snap starts a new cluster, the
representative is the cluster's first = smallest value), the same for y with hy; every (x_i, y_j) pair is a column,
plus the centres of the start cover (the greedy cover's ordinary FOVs). A part covers the columns whose x lies in
[lx, hx] and y in [ly, hy] (+-TOL): contiguous index ranges found by searchsorted, so the incidence is sparse (CSR
both ways).
Local search: RWLS (Gao, Weise & Li 2015) with row weighting, configuration checking and oldest-first ties, a seeded
LCG and an iteration budget (no wall clock). Every row keeps the sum of the indices of its selected covering columns,
so the one other covering column (after a drop leaves the row covered once, or an add makes it covered twice) is
found without scanning the row. RWLSRun splits one run into resumable segments with the same trajectory.
Optional reduction: grid columns dominated by a 4-neighbour (every part covering the column also covers the
neighbour) can be removed before the search (reduce_columns).
All kernels compile at import with explicit signatures. Without numba, or when a kernel fails to compile, the
plain-Python kernels of setcover_py take their place (PY = True: the same results for the same budgets, much slower,
so the solver scales the budgets down); ENABLED = False only if those fail too.
"""
import numpy as np

try:
    from numba import njit

    @njit('Tuple((int64[::1], int64[::1], int64[::1], int64[::1]))(float64[::1], float64[::1], float64[::1], '
          'float64[::1], float64[::1], float64[::1], float64[:, ::1], float64)', cache=False, nogil=True)
    def _incidence(lx, hx, ly, hy, xs, ys, own, tol):
        ne, nx, ny, no = lx.shape[0], xs.shape[0], ys.shape[0], own.shape[0]
        nc = nx * ny + no
        a = np.searchsorted(xs, lx - tol, 'left')
        b = np.searchsorted(xs, hx + tol, 'right')
        c = np.searchsorted(ys, ly - tol, 'left')
        d = np.searchsorted(ys, hy + tol, 'right')
        e_ptr = np.zeros(ne + 1, dtype=np.int64)
        for e in range(ne):
            k = max(b[e] - a[e], 0) * max(d[e] - c[e], 0)
            for o in range(no):
                if lx[e] - tol <= own[o, 0] <= hx[e] + tol and ly[e] - tol <= own[o, 1] <= hy[e] + tol:
                    k += 1
            e_ptr[e + 1] = e_ptr[e] + k
        e_col = np.empty(e_ptr[ne], dtype=np.int64)
        c_cnt = np.zeros(nc + 1, dtype=np.int64)
        for e in range(ne):
            q = e_ptr[e]
            for i in range(a[e], b[e]):
                for j in range(c[e], d[e]):
                    col = i * ny + j
                    e_col[q] = col
                    c_cnt[col + 1] += 1
                    q += 1
            for o in range(no):
                if lx[e] - tol <= own[o, 0] <= hx[e] + tol and ly[e] - tol <= own[o, 1] <= hy[e] + tol:
                    col = nx * ny + o
                    e_col[q] = col
                    c_cnt[col + 1] += 1
                    q += 1
        c_ptr = np.cumsum(c_cnt)
        fill = c_ptr[:-1].copy()
        c_el = np.empty(e_ptr[ne], dtype=np.int64)
        for e in range(ne):
            for q in range(e_ptr[e], e_ptr[e + 1]):
                col = e_col[q]
                c_el[fill[col]] = e
                fill[col] += 1
        return e_ptr, e_col, c_ptr, c_el

    @njit('boolean(int64, int64, int64, int64, int64, int64)', cache=False, nogil=True)
    def _better(s, t, j, bs, bt, bj):
        """Higher score, then older timestamp, then lower column index."""
        return s > bs or (s == bs and (t < bt or (t == bt and j < bj)))

    @njit('int64(int64, boolean[::1], int64[::1], int64[::1], int64[::1], boolean[::1], int64[::1], int64[::1], '
          'int64[::1], int64[::1], int64[::1], int64[::1], int64, int64[::1])', cache=False, nogil=True)
    def _drop(j, sel, cnt, w, score, cc, e_ptr, e_col, c_ptr, c_el, unc, upos, n_unc, ssum):
        """Remove column j; rows it leaves uncovered join the uncovered list; neighbours' scores and CC flags update."""
        sel[j] = False
        for q in range(c_ptr[j], c_ptr[j + 1]):
            e = c_el[q]
            cnt[e] -= 1
            ssum[e] -= j
            if cnt[e] == 0:
                unc[n_unc] = e
                upos[e] = n_unc
                n_unc += 1
                for r in range(e_ptr[e], e_ptr[e + 1]):
                    k = e_col[r]
                    if k != j:
                        score[k] += w[e]
                        cc[k] = True
            elif cnt[e] == 1:
                score[ssum[e]] -= w[e]  # the one selected column still covering e
        s = 0
        for q in range(c_ptr[j], c_ptr[j + 1]):
            if cnt[c_el[q]] == 0:
                s += w[c_el[q]]
        score[j] = s
        return n_unc

    @njit('int64(int64, boolean[::1], int64[::1], int64[::1], int64[::1], boolean[::1], int64[::1], int64[::1], '
          'int64[::1], int64[::1], int64[::1], int64[::1], int64, int64[::1])', cache=False, nogil=True)
    def _add(j, sel, cnt, w, score, cc, e_ptr, e_col, c_ptr, c_el, unc, upos, n_unc, ssum):
        """Add column j; rows it covers leave the uncovered list; neighbours' scores and CC flags update."""
        sel[j] = True
        for q in range(c_ptr[j], c_ptr[j + 1]):
            e = c_el[q]
            cnt[e] += 1
            ssum[e] += j
            if cnt[e] == 1:
                t = upos[e]
                last = unc[n_unc - 1]
                unc[t] = last
                upos[last] = t
                upos[e] = -1
                n_unc -= 1
                for r in range(e_ptr[e], e_ptr[e + 1]):
                    k = e_col[r]
                    if k != j:
                        score[k] -= w[e]
                        cc[k] = True
            elif cnt[e] == 2:
                score[ssum[e] - j] += w[e]  # the other selected column covering e
        s = 0
        for q in range(c_ptr[j], c_ptr[j + 1]):
            if cnt[c_el[q]] == 1:
                s -= w[c_el[q]]
        score[j] = s
        return n_unc

    @njit('int64[::1](int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64, int64)', cache=False, nogil=True)
    def rwls(e_ptr, e_col, c_ptr, c_el, init, iters, seed):
        """RWLS: row weighting, configuration checking, oldest-first ties. Returns the best feasible cover seen (sorted)
        or an empty array when none was feasible."""
        ne = e_ptr.shape[0] - 1
        nc = c_ptr.shape[0] - 1
        w = np.ones(ne, dtype=np.int64)
        cnt = np.zeros(ne, dtype=np.int64)
        sel = np.zeros(nc, dtype=np.bool_)
        score = np.zeros(nc, dtype=np.int64)
        ts = np.zeros(nc, dtype=np.int64)
        cc = np.ones(nc, dtype=np.bool_)
        lst = np.empty(nc, dtype=np.int64)
        pos = np.full(nc, -1, dtype=np.int64)
        unc = np.empty(ne, dtype=np.int64)
        upos = np.full(ne, -1, dtype=np.int64)
        ssum = np.zeros(ne, dtype=np.int64)
        n_sel = 0
        for j in init:
            if not sel[j]:
                sel[j] = True
                lst[n_sel] = j
                pos[j] = n_sel
                n_sel += 1
                for q in range(c_ptr[j], c_ptr[j + 1]):
                    cnt[c_el[q]] += 1
                    ssum[c_el[q]] += j
        n_unc = 0
        for e in range(ne):
            if cnt[e] == 0:
                unc[n_unc] = e
                upos[e] = n_unc
                n_unc += 1
        for j in range(nc):
            s = 0
            for q in range(c_ptr[j], c_ptr[j + 1]):
                e = c_el[q]
                if sel[j]:
                    if cnt[e] == 1:
                        s -= w[e]
                elif cnt[e] == 0:
                    s += w[e]
            score[j] = s
        best = lst[:n_sel].copy()
        best_n = n_sel if n_unc == 0 else nc + 1
        last_added = -1
        rng = np.uint64(seed) * np.uint64(2654435761) + np.uint64(1)
        for it in range(1, iters + 1):
            if n_unc == 0:
                if n_sel < best_n:
                    best_n = n_sel
                    best = lst[:n_sel].copy()
                bj, bs, bt = -1, -(1 << 60), 1 << 60
                for t in range(n_sel):
                    j = lst[t]
                    if _better(score[j], ts[j], j, bs, bt, bj):
                        bj, bs, bt = j, score[j], ts[j]
                if bj < 0:  # nothing selected and nothing to cover (no rows): the empty cover is the answer
                    break
                n_unc = _drop(bj, sel, cnt, w, score, cc, e_ptr, e_col, c_ptr, c_el, unc, upos, n_unc, ssum)
                cc[bj] = False
                ts[bj] = it
                t = pos[bj]
                lst[t] = lst[n_sel - 1]
                pos[lst[t]] = t
                pos[bj] = -1
                n_sel -= 1
                continue
            bj, bs, bt = -1, -(1 << 60), 1 << 60
            for t in range(n_sel):
                j = lst[t]
                if j != last_added and _better(score[j], ts[j], j, bs, bt, bj):
                    bj, bs, bt = j, score[j], ts[j]
            if bj >= 0:
                n_unc = _drop(bj, sel, cnt, w, score, cc, e_ptr, e_col, c_ptr, c_el, unc, upos, n_unc, ssum)
                cc[bj] = False
                ts[bj] = it
                t = pos[bj]
                lst[t] = lst[n_sel - 1]
                pos[lst[t]] = t
                pos[bj] = -1
                n_sel -= 1
            rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            e0 = unc[np.int64(rng >> np.uint64(33)) % n_unc]
            aj, as_, at = -1, -(1 << 60), 1 << 60
            fj, fs, ft = -1, -(1 << 60), 1 << 60
            for r in range(e_ptr[e0], e_ptr[e0 + 1]):
                k = e_col[r]
                if sel[k]:
                    continue
                if cc[k] and _better(score[k], ts[k], k, as_, at, aj):
                    aj, as_, at = k, score[k], ts[k]
                if _better(score[k], ts[k], k, fs, ft, fj):
                    fj, fs, ft = k, score[k], ts[k]
            if aj < 0:
                aj = fj
            if aj < 0:  # the chosen row has no unselected candidate column: stop
                break
            n_unc = _add(aj, sel, cnt, w, score, cc, e_ptr, e_col, c_ptr, c_el, unc, upos, n_unc, ssum)
            ts[aj] = it
            lst[n_sel] = aj
            pos[aj] = n_sel
            n_sel += 1
            last_added = aj
            for t in range(n_unc):
                e = unc[t]
                w[e] += 1
                for r in range(e_ptr[e], e_ptr[e + 1]):
                    score[e_col[r]] += 1
        if n_unc == 0 and n_sel < best_n:
            best = lst[:n_sel].copy()
            best_n = n_sel
        if best_n > nc:  # never feasible: empty result, the caller rejects it
            return np.empty(0, dtype=np.int64)
        return np.sort(best)

    @njit('void(int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64, int64[::1], int64[::1], int64[::1], '
          'boolean[::1], int64[::1], boolean[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], '
          'uint64[::1])', cache=False, nogil=True)
    def _rwls_init(e_ptr, e_col, c_ptr, c_el, init, seed, w, cnt, ssum, sel, score, cc, lst, pos, unc, upos, best, sc, rs):
        """rwls's set-up, verbatim, on caller-allocated arrays (w = 1, cc = True, pos / upos = -1, the rest 0 / any);
        scalars go to sc = [n_sel, n_unc, best_n, last_added, iterations done, stopped] and rs = [rng]."""
        ne = e_ptr.shape[0] - 1
        nc = c_ptr.shape[0] - 1
        n_sel = 0
        for j in init:
            if not sel[j]:
                sel[j] = True
                lst[n_sel] = j
                pos[j] = n_sel
                n_sel += 1
                for q in range(c_ptr[j], c_ptr[j + 1]):
                    cnt[c_el[q]] += 1
                    ssum[c_el[q]] += j
        n_unc = 0
        for e in range(ne):
            if cnt[e] == 0:
                unc[n_unc] = e
                upos[e] = n_unc
                n_unc += 1
        for j in range(nc):
            s = 0
            for q in range(c_ptr[j], c_ptr[j + 1]):
                e = c_el[q]
                if sel[j]:
                    if cnt[e] == 1:
                        s -= w[e]
                elif cnt[e] == 0:
                    s += w[e]
            score[j] = s
        for t in range(n_sel):
            best[t] = lst[t]
        sc[0] = n_sel
        sc[1] = n_unc
        sc[2] = n_sel if n_unc == 0 else nc + 1
        sc[3] = -1
        sc[4] = 0
        sc[5] = 0
        rs[0] = np.uint64(seed) * np.uint64(2654435761) + np.uint64(1)

    @njit('void(int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], boolean[::1], '
          'int64[::1], int64[::1], boolean[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], '
          'uint64[::1], int64)', cache=False, nogil=True)
    def _rwls_seg(e_ptr, e_col, c_ptr, c_el, w, cnt, ssum, sel, score, ts, cc, lst, pos, unc, upos, best, sc, rs, iters):
        """rwls's main loop, verbatim, for iterations sc[4] + 1 .. iters of a saved state, so a run split at any
        iteration ends in the state of the unsplit run. rwls's closing best-update runs at the end of every segment; the
        next segment's first feasible check then sees n_sel == best_n and changes nothing, as in the unsplit run."""
        nc = c_ptr.shape[0] - 1
        n_sel, n_unc, best_n, last_added = sc[0], sc[1], sc[2], sc[3]
        rng = rs[0]
        stopped = sc[5] != 0
        done = sc[4]
        if not stopped:
            for it in range(sc[4] + 1, iters + 1):
                done = it
                if n_unc == 0:
                    if n_sel < best_n:
                        best_n = n_sel
                        for t in range(n_sel):
                            best[t] = lst[t]
                    bj, bs, bt = -1, -(1 << 60), 1 << 60
                    for t in range(n_sel):
                        j = lst[t]
                        if _better(score[j], ts[j], j, bs, bt, bj):
                            bj, bs, bt = j, score[j], ts[j]
                    if bj < 0:
                        stopped = True
                        break
                    n_unc = _drop(bj, sel, cnt, w, score, cc, e_ptr, e_col, c_ptr, c_el, unc, upos, n_unc, ssum)
                    cc[bj] = False
                    ts[bj] = it
                    t = pos[bj]
                    lst[t] = lst[n_sel - 1]
                    pos[lst[t]] = t
                    pos[bj] = -1
                    n_sel -= 1
                    continue
                bj, bs, bt = -1, -(1 << 60), 1 << 60
                for t in range(n_sel):
                    j = lst[t]
                    if j != last_added and _better(score[j], ts[j], j, bs, bt, bj):
                        bj, bs, bt = j, score[j], ts[j]
                if bj >= 0:
                    n_unc = _drop(bj, sel, cnt, w, score, cc, e_ptr, e_col, c_ptr, c_el, unc, upos, n_unc, ssum)
                    cc[bj] = False
                    ts[bj] = it
                    t = pos[bj]
                    lst[t] = lst[n_sel - 1]
                    pos[lst[t]] = t
                    pos[bj] = -1
                    n_sel -= 1
                rng = rng * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                e0 = unc[np.int64(rng >> np.uint64(33)) % n_unc]
                aj, as_, at = -1, -(1 << 60), 1 << 60
                fj, fs, ft = -1, -(1 << 60), 1 << 60
                for r in range(e_ptr[e0], e_ptr[e0 + 1]):
                    k = e_col[r]
                    if sel[k]:
                        continue
                    if cc[k] and _better(score[k], ts[k], k, as_, at, aj):
                        aj, as_, at = k, score[k], ts[k]
                    if _better(score[k], ts[k], k, fs, ft, fj):
                        fj, fs, ft = k, score[k], ts[k]
                if aj < 0:
                    aj = fj
                if aj < 0:
                    stopped = True
                    break
                n_unc = _add(aj, sel, cnt, w, score, cc, e_ptr, e_col, c_ptr, c_el, unc, upos, n_unc, ssum)
                ts[aj] = it
                lst[n_sel] = aj
                pos[aj] = n_sel
                n_sel += 1
                last_added = aj
                for t in range(n_unc):
                    e = unc[t]
                    w[e] += 1
                    for r in range(e_ptr[e], e_ptr[e + 1]):
                        score[e_col[r]] += 1
        if n_unc == 0 and n_sel < best_n:
            for t in range(n_sel):
                best[t] = lst[t]
            best_n = n_sel
        sc[0] = n_sel
        sc[1] = n_unc
        sc[2] = best_n
        sc[3] = last_added
        sc[4] = done if stopped else max(done, iters)
        sc[5] = 1 if stopped else 0
        rs[0] = rng

    @njit('int64[::1](int64[::1], int64[::1], int64[::1], float64[:, ::1], float64[::1], float64[::1])', cache=False,
          nogil=True)
    def assign(e_ptr, e_col, rank, pts, px, py):
        """For each element, the selected column (rank >= 0) nearest to the part centre; ties: lowest column."""
        ne = e_ptr.shape[0] - 1
        out = np.full(ne, -1, dtype=np.int64)
        for e in range(ne):
            bd, bj = np.inf, -1
            for r in range(e_ptr[e], e_ptr[e + 1]):
                j = e_col[r]
                k = rank[j]
                if k >= 0:
                    dx = pts[k, 0] - px[e]
                    dy = pts[k, 1] - py[e]
                    dd = dx * dx + dy * dy
                    if dd < bd or (dd == bd and j < bj):
                        bd, bj = dd, j
            out[e] = bj
        return out

    @njit('boolean[::1](int64[::1], int64[::1], int64, int64, int64, int64[::1], int64[::1], int64[::1], int64[::1])',
          cache=False, nogil=True)
    def dom_keep(c_ptr, c_el, n_grid, nx, ny, a, b, c, d):
        """Grid columns (i, j) = i * ny + j not dominated by a 4-neighbour; rows cover index rectangles
        [a, b) x [c, d). A neighbour holds R(p) when every row of p extends into it; p goes when such a neighbour has
        more rows or the same rows and a lower index. Empty grid columns go. Columns >= n_grid are kept."""
        nc = c_ptr.shape[0] - 1
        keep = np.ones(nc, dtype=np.bool_)
        for p in range(n_grid):
            cp = c_ptr[p + 1] - c_ptr[p]
            if cp == 0:
                keep[p] = False
                continue
            i = p // ny
            j = p - i * ny
            r_ok = i + 1 < nx
            l_ok = i > 0
            u_ok = j + 1 < ny
            d_ok = j > 0
            for q in range(c_ptr[p], c_ptr[p + 1]):
                e = c_el[q]
                if b[e] - 1 <= i:
                    r_ok = False
                if a[e] >= i:
                    l_ok = False
                if d[e] - 1 <= j:
                    u_ok = False
                if c[e] >= j:
                    d_ok = False
            if r_ok:
                nb = p + ny
                if c_ptr[nb + 1] - c_ptr[nb] > cp or nb < p:
                    keep[p] = False
                    continue
            if l_ok:
                nb = p - ny
                if c_ptr[nb + 1] - c_ptr[nb] > cp or nb < p:
                    keep[p] = False
                    continue
            if u_ok:
                nb = p + 1
                if c_ptr[nb + 1] - c_ptr[nb] > cp or nb < p:
                    keep[p] = False
                    continue
            if d_ok:
                nb = p - 1
                if c_ptr[nb + 1] - c_ptr[nb] > cp or nb < p:
                    keep[p] = False
                    continue
        return keep

    @njit('Tuple((int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1]))(int64[::1], int64[::1], '
          'boolean[::1])', cache=False, nogil=True)
    def dom_rebuild(e_ptr, e_col, keep):
        """The incidence restricted to kept columns (new ids in kept order), old ids per new id, new id per old id
        (-1 when dropped)."""
        nc = keep.shape[0]
        ne = e_ptr.shape[0] - 1
        new_of = np.full(nc, -1, dtype=np.int64)
        n2 = 0
        for j in range(nc):
            if keep[j]:
                new_of[j] = n2
                n2 += 1
        old = np.empty(n2, dtype=np.int64)
        for j in range(nc):
            if keep[j]:
                old[new_of[j]] = j
        e_ptr2 = np.zeros(ne + 1, dtype=np.int64)
        for e in range(ne):
            k = 0
            for q in range(e_ptr[e], e_ptr[e + 1]):
                if new_of[e_col[q]] >= 0:
                    k += 1
            e_ptr2[e + 1] = e_ptr2[e] + k
        e_col2 = np.empty(e_ptr2[ne], dtype=np.int64)
        cnt = np.zeros(n2 + 1, dtype=np.int64)
        for e in range(ne):
            w = e_ptr2[e]
            for q in range(e_ptr[e], e_ptr[e + 1]):
                cc = new_of[e_col[q]]
                if cc >= 0:
                    e_col2[w] = cc
                    cnt[cc + 1] += 1
                    w += 1
        c_ptr2 = np.cumsum(cnt)
        fill = c_ptr2[:-1].copy()
        c_el2 = np.empty(e_ptr2[ne], dtype=np.int64)
        for e in range(ne):
            for q in range(e_ptr2[e], e_ptr2[e + 1]):
                cc = e_col2[q]
                c_el2[fill[cc]] = e
                fill[cc] += 1
        return e_ptr2, e_col2, c_ptr2, c_el2, old, new_of

    ENABLED = True
    PY = False
except Exception:  # numba missing or a kernel failed to compile: the plain-Python kernels of setcover_py
    try:
        try:
            from .setcover_py import _incidence, rwls, assign, dom_keep, dom_rebuild, RWLSRunPy as _RWLSRunPy
        except ImportError:
            from setcover_py import _incidence, rwls, assign, dom_keep, dom_rebuild, RWLSRunPy as _RWLSRunPy
        ENABLED = True
        PY = True
    except Exception:  # no set-cover search at all: the solver keeps the greedy cover
        ENABLED = False
        PY = False


def reps(vals, snap):
    """Cluster representatives of the sorted values: a gap > snap starts a new cluster; each cluster is represented
    by its smallest value."""
    v = np.sort(np.asarray(vals, dtype=np.float64))
    out = [v[0]]
    last = v[0]
    for a in v[1:]:
        if a - last > snap:
            out.append(a)
        last = a
    return np.array(out, dtype=np.float64)


def undominated(reps, lo, hi, tol):
    """1-D dominance pruning of sorted candidate coordinates. Element e holds reps [a_e, b_e) (inside [lo-tol, hi+tol]).
    set(k) <= set(k-1) iff no element starts at k; set(k) <= set(k+1) iff no element ends at k. Drop k when
    (k >= 1 and nothing starts at k) or (nothing ends at k and something starts at k+1). A dropped rep always has a
    kept superset (left chain of subsets, or a strictly growing right chain), and set(i) <= set(i') on one axis gives
    column (i, j) <= (i', j) for every j, so some minimum cover uses kept coordinates only."""
    a = np.searchsorted(reps, lo - tol, 'left')
    b = np.searchsorted(reps, hi + tol, 'right')
    ok = b > a
    m = len(reps)
    starts = np.bincount(a[ok], minlength=m + 1)[:m]
    ends = np.bincount(b[ok] - 1, minlength=m)[:m]
    keep = np.ones(m, dtype=bool)
    keep[1:] &= starts[1:] > 0
    keep &= ~((ends == 0) & (np.concatenate((starts[1:], [0])) > 0))
    return reps[keep]


class RWLSRun:
    """A resumable rwls(e_ptr, e_col, c_ptr, c_el, init, iters, seed). run(iters) returns exactly what rwls with that
    budget returns (its trajectory does not depend on the budget); a later run with a larger budget continues from the
    saved state instead of repeating the first iterations. Budgets must not decrease."""

    def __init__(self, e_ptr, e_col, c_ptr, c_el, init, seed):
        ne, nc = e_ptr.shape[0] - 1, c_ptr.shape[0] - 1
        self.inc = (e_ptr, e_col, c_ptr, c_el)
        self.nc = nc
        self.w = np.ones(ne, dtype=np.int64)
        self.cnt = np.zeros(ne, dtype=np.int64)
        self.ssum = np.zeros(ne, dtype=np.int64)
        self.sel = np.zeros(nc, dtype=np.bool_)
        self.score = np.zeros(nc, dtype=np.int64)
        self.ts = np.zeros(nc, dtype=np.int64)
        self.cc = np.ones(nc, dtype=np.bool_)
        self.lst = np.zeros(nc, dtype=np.int64)
        self.pos = np.full(nc, -1, dtype=np.int64)
        self.unc = np.zeros(ne, dtype=np.int64)
        self.upos = np.full(ne, -1, dtype=np.int64)
        self.best = np.zeros(nc, dtype=np.int64)
        self.sc = np.zeros(6, dtype=np.int64)
        self.rs = np.zeros(1, dtype=np.uint64)
        _rwls_init(e_ptr, e_col, c_ptr, c_el, np.ascontiguousarray(init, dtype=np.int64), int(seed), self.w, self.cnt,
                   self.ssum, self.sel, self.score, self.cc, self.lst, self.pos, self.unc, self.upos, self.best, self.sc,
                   self.rs)

    @property
    def done(self):
        return int(self.sc[4])

    def run(self, iters):
        if iters < self.done:
            raise ValueError('RWLSRun budgets must not decrease')
        _rwls_seg(*self.inc, self.w, self.cnt, self.ssum, self.sel, self.score, self.ts, self.cc, self.lst, self.pos,
                  self.unc, self.upos, self.best, self.sc, self.rs, int(iters))
        bn = int(self.sc[2])
        if bn > self.nc:
            return np.empty(0, dtype=np.int64)
        return np.sort(self.best[:bn])


if PY:
    RWLSRun = _RWLSRunPy  # noqa: F811 (the resumable run on the Python state)


def reduce_columns(e_ptr, e_col, c_ptr, c_el, lx, hx, ly, hy, xs, ys, tol):
    """Exact 2-D dominance on the candidate grid of build_incidence (the start cover's centres after the grid are
    kept). Returns (e_ptr, e_col, c_ptr, c_el, old, new_of) of the reduced instance."""
    nx, ny = len(xs), len(ys)
    f = lambda v: np.ascontiguousarray(v, dtype=np.float64)
    a = np.ascontiguousarray(np.searchsorted(f(xs), f(lx) - tol, 'left'), dtype=np.int64)
    b = np.ascontiguousarray(np.searchsorted(f(xs), f(hx) + tol, 'right'), dtype=np.int64)
    c = np.ascontiguousarray(np.searchsorted(f(ys), f(ly) - tol, 'left'), dtype=np.int64)
    d = np.ascontiguousarray(np.searchsorted(f(ys), f(hy) + tol, 'right'), dtype=np.int64)
    keep = dom_keep(c_ptr, c_el, int(nx * ny), int(nx), int(ny), a, b, c, d)
    return dom_rebuild(e_ptr, e_col, keep)


def build_incidence(lx, hx, ly, hy, xs, ys, own, tol):
    """The part x candidate incidence (CSR both ways) of the grid xs x ys plus the start cover's centres `own`;
    returns (e_ptr, e_col, c_ptr, c_el, number of columns)."""
    own = np.ascontiguousarray(own, dtype=np.float64).reshape(-1, 2)
    f = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    e_ptr, e_col, c_ptr, c_el = _incidence(f(lx), f(hx), f(ly), f(hy), f(xs), f(ys), own, float(tol))
    return e_ptr, e_col, c_ptr, c_el, len(xs) * len(ys) + len(own)
