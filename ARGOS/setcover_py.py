"""The set-cover kernels of setcover.py in plain Python / numpy, used only when numba is missing or its kernels fail
to compile (setcover.PY). Same inputs, same outputs as the compiled kernels: the same incidence (CSR both ways), the same
RWLS trajectory for the same iteration budget (row weighting, configuration checking, oldest-first ties, the same LCG),
the same dominance reduction and the same assignment. The RWLS loop works on lists of Python ints (numpy scalars cost
several times more per operation in the interpreter). The caller scales the iteration budgets down
(solver.PY_ITERS_SCALE).
"""
import numpy as np

_M64 = (1 << 64) - 1


def _incidence(lx, hx, ly, hy, xs, ys, own, tol):
    ne, nx, ny, no = lx.shape[0], xs.shape[0], ys.shape[0], own.shape[0]
    nc = nx * ny + no
    a = np.searchsorted(xs, lx - tol, 'left')
    b = np.searchsorted(xs, hx + tol, 'right')
    c = np.searchsorted(ys, ly - tol, 'left')
    d = np.searchsorted(ys, hy + tol, 'right')
    if no:
        ox, oy = own[:, 0][None, :], own[:, 1][None, :]
        inside = (lx[:, None] - tol <= ox) & (ox <= hx[:, None] + tol) & (ly[:, None] - tol <= oy) & (oy <= hy[:, None] + tol)
    else:
        inside = np.zeros((ne, 0), dtype=bool)
    e_col_parts = []
    counts = np.zeros(ne, dtype=np.int64)
    for e in range(ne):
        ii = np.arange(a[e], b[e], dtype=np.int64)
        jj = np.arange(c[e], d[e], dtype=np.int64)
        grid = (ii[:, None] * ny + jj[None, :]).ravel() if len(ii) and len(jj) else np.empty(0, dtype=np.int64)
        extra = nx * ny + np.flatnonzero(inside[e]).astype(np.int64)
        cols = np.concatenate((grid, extra)) if len(extra) else grid
        e_col_parts.append(cols)
        counts[e] = len(cols)
    e_ptr = np.zeros(ne + 1, dtype=np.int64)
    np.cumsum(counts, out=e_ptr[1:])
    e_col = np.concatenate(e_col_parts).astype(np.int64) if ne else np.empty(0, dtype=np.int64)
    rows = np.repeat(np.arange(ne, dtype=np.int64), counts)
    order = np.argsort(e_col, kind='stable')  # by column, then by row (rows are already increasing)
    c_el = rows[order]
    c_cnt = np.bincount(e_col, minlength=nc).astype(np.int64) if len(e_col) else np.zeros(nc, dtype=np.int64)
    c_ptr = np.zeros(nc + 1, dtype=np.int64)
    np.cumsum(c_cnt, out=c_ptr[1:])
    return e_ptr, np.ascontiguousarray(e_col), c_ptr, np.ascontiguousarray(c_el)


class _State:
    """RWLS state in Python lists; init/seg follow setcover._rwls_init / setcover._rwls_seg line for line."""

    def __init__(self, e_ptr, e_col, c_ptr, c_el, init, seed):
        ne = len(e_ptr) - 1
        nc = len(c_ptr) - 1
        self.nc = nc
        ep, ec, cp, ce = e_ptr.tolist(), e_col.tolist(), c_ptr.tolist(), c_el.tolist()
        self.rcols = [ec[ep[e]:ep[e + 1]] for e in range(ne)]   # columns of each row
        self.crows = [ce[cp[j]:cp[j + 1]] for j in range(nc)]   # rows of each column
        self.w = [1] * ne
        self.cnt = [0] * ne
        self.ssum = [0] * ne
        self.sel = [False] * nc
        self.score = [0] * nc
        self.ts = [0] * nc
        self.cc = [True] * nc
        self.lst = []
        self.pos = [-1] * nc
        self.unc = []
        self.upos = [-1] * ne
        sel, lst, pos, cnt, ssum, crows = self.sel, self.lst, self.pos, self.cnt, self.ssum, self.crows
        for j in (int(x) for x in init):
            if not sel[j]:
                sel[j] = True
                pos[j] = len(lst)
                lst.append(j)
                for e in crows[j]:
                    cnt[e] += 1
                    ssum[e] += j
        for e in range(ne):
            if cnt[e] == 0:
                self.upos[e] = len(self.unc)
                self.unc.append(e)
        w, score = self.w, self.score
        for j in range(nc):
            s = 0
            if sel[j]:
                for e in crows[j]:
                    if cnt[e] == 1:
                        s -= w[e]
            else:
                for e in crows[j]:
                    if cnt[e] == 0:
                        s += w[e]
            score[j] = s
        self.best = list(lst)
        self.best_n = len(lst) if not self.unc else nc + 1
        self.last_added = -1
        self.done = 0
        self.stopped = False
        self.rng = (int(seed) * 2654435761 + 1) & _M64

    def _drop(self, j):
        sel, cnt, ssum, w, score, cc, unc, upos, rcols = (self.sel, self.cnt, self.ssum, self.w, self.score, self.cc,
                                                          self.unc, self.upos, self.rcols)
        sel[j] = False
        rows = self.crows[j]
        for e in rows:
            cnt[e] -= 1
            ssum[e] -= j
            c = cnt[e]
            if c == 0:
                upos[e] = len(unc)
                unc.append(e)
                we = w[e]
                for k in rcols[e]:
                    if k != j:
                        score[k] += we
                        cc[k] = True
            elif c == 1:
                score[ssum[e]] -= w[e]
        s = 0
        for e in rows:
            if cnt[e] == 0:
                s += w[e]
        score[j] = s

    def _add(self, j):
        sel, cnt, ssum, w, score, cc, unc, upos, rcols = (self.sel, self.cnt, self.ssum, self.w, self.score, self.cc,
                                                          self.unc, self.upos, self.rcols)
        sel[j] = True
        rows = self.crows[j]
        for e in rows:
            cnt[e] += 1
            ssum[e] += j
            c = cnt[e]
            if c == 1:
                t = upos[e]
                last = unc[-1]
                unc[t] = last
                upos[last] = t
                upos[e] = -1
                unc.pop()
                we = w[e]
                for k in rcols[e]:
                    if k != j:
                        score[k] -= we
                        cc[k] = True
            elif c == 2:
                score[ssum[e] - j] += w[e]
        s = 0
        for e in rows:
            if cnt[e] == 1:
                s -= w[e]
        score[j] = s

    def _remove_from_list(self, bj):
        lst, pos = self.lst, self.pos
        t = pos[bj]
        last = lst[-1]
        lst[t] = last
        pos[last] = t
        pos[bj] = -1
        lst.pop()

    def seg(self, iters):
        """Iterations done + 1 .. iters (setcover._rwls_seg); the closing best-update runs at the end."""
        score, ts, cc, sel, lst, unc, w, rcols = (self.score, self.ts, self.cc, self.sel, self.lst, self.unc, self.w,
                                                  self.rcols)
        done = self.done
        if not self.stopped:
            for it in range(self.done + 1, iters + 1):
                done = it
                if not unc:
                    if len(lst) < self.best_n:
                        self.best_n = len(lst)
                        self.best = list(lst)
                    bj, bs, bt = -1, -(1 << 60), 1 << 60
                    for j in lst:  # higher score, then older timestamp, then lower index
                        s = score[j]
                        if s > bs or (s == bs and (ts[j] < bt or (ts[j] == bt and j < bj))):
                            bj, bs, bt = j, s, ts[j]
                    if bj < 0:
                        self.stopped = True
                        break
                    self._drop(bj)
                    cc[bj] = False
                    ts[bj] = it
                    self._remove_from_list(bj)
                    continue
                la = self.last_added
                bj, bs, bt = -1, -(1 << 60), 1 << 60
                for j in lst:
                    if j != la:
                        s = score[j]
                        if s > bs or (s == bs and (ts[j] < bt or (ts[j] == bt and j < bj))):
                            bj, bs, bt = j, s, ts[j]
                if bj >= 0:
                    self._drop(bj)
                    cc[bj] = False
                    ts[bj] = it
                    self._remove_from_list(bj)
                self.rng = (self.rng * 6364136223846793005 + 1442695040888963407) & _M64
                e0 = unc[(self.rng >> 33) % len(unc)]
                aj, as_, at = -1, -(1 << 60), 1 << 60
                fj, fs, ft = -1, -(1 << 60), 1 << 60
                for k in rcols[e0]:
                    if sel[k]:
                        continue
                    s, t = score[k], ts[k]
                    if cc[k] and (s > as_ or (s == as_ and (t < at or (t == at and k < aj)))):
                        aj, as_, at = k, s, t
                    if s > fs or (s == fs and (t < ft or (t == ft and k < fj))):
                        fj, fs, ft = k, s, t
                if aj < 0:
                    aj = fj
                if aj < 0:
                    self.stopped = True
                    break
                self._add(aj)
                ts[aj] = it
                self.pos[aj] = len(lst)
                lst.append(aj)
                self.last_added = aj
                for e in unc:
                    w[e] += 1
                    for k in rcols[e]:
                        score[k] += 1
        if not unc and len(lst) < self.best_n:
            self.best = list(lst)
            self.best_n = len(lst)
        self.done = done if self.stopped else max(done, iters)

    def result(self):
        if self.best_n > self.nc:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.array(self.best[:self.best_n], dtype=np.int64))


def rwls(e_ptr, e_col, c_ptr, c_el, init, iters, seed):
    st = _State(e_ptr, e_col, c_ptr, c_el, init, seed)
    st.seg(int(iters))
    return st.result()


class RWLSRunPy:
    """setcover.RWLSRun on the Python state: run(iters) returns what rwls with that budget returns."""

    def __init__(self, e_ptr, e_col, c_ptr, c_el, init, seed):
        self.nc = len(c_ptr) - 1
        self._st = _State(e_ptr, e_col, c_ptr, c_el, init, seed)

    @property
    def done(self):
        return int(self._st.done)

    def run(self, iters):
        if iters < self.done:
            raise ValueError('RWLSRun budgets must not decrease')
        self._st.seg(int(iters))
        return self._st.result()


def assign(e_ptr, e_col, rank, pts, px, py):
    """For each element, the selected column (rank >= 0) nearest to the part centre; ties: lowest column."""
    ne = len(e_ptr) - 1
    out = np.full(ne, -1, dtype=np.int64)
    if ne == 0 or len(e_col) == 0:
        return out
    rows = np.repeat(np.arange(ne, dtype=np.int64), np.diff(e_ptr))
    k = rank[e_col]
    ok = k >= 0
    if not ok.any():
        return out
    r, j, kk = rows[ok], e_col[ok], k[ok]
    dx = pts[kk, 0] - px[r]
    dy = pts[kk, 1] - py[r]
    dd = dx * dx + dy * dy
    order = np.lexsort((j, dd, r))  # per row: smallest distance, then lowest column
    r_s = r[order]
    first = np.ones(len(r_s), dtype=bool)
    first[1:] = r_s[1:] != r_s[:-1]
    out[r_s[first]] = j[order][first]
    return out


def dom_keep(c_ptr, c_el, n_grid, nx, ny, a, b, c, d):
    """setcover.dom_keep with numpy: per grid column p = i * ny + j, a neighbour direction is open when every row of p
    extends into it (min over the column's rows)."""
    nc = len(c_ptr) - 1
    keep = np.ones(nc, dtype=bool)
    cnt = np.diff(c_ptr)
    big = np.int64(1) << 60
    # per-column reductions over rows (empty columns: neutral values, dropped below anyway)
    starts = c_ptr[:-1]
    has = cnt > 0
    def red(vals, fn, neutral):
        outv = np.full(nc, neutral, dtype=np.int64)
        if len(c_el):
            idx = starts[has]
            outv[has] = fn.reduceat(vals[c_el], idx)
        return outv
    min_b1 = red(b - 1, np.minimum, big)
    max_a = red(a, np.maximum, -big)
    min_d1 = red(d - 1, np.minimum, big)
    max_c = red(c, np.maximum, -big)
    p = np.arange(n_grid, dtype=np.int64)
    i = p // ny
    j = p - i * ny
    cp = cnt[:n_grid]
    r_ok = (i + 1 < nx) & (min_b1[:n_grid] > i)
    l_ok = (i > 0) & (max_a[:n_grid] < i)
    u_ok = (j + 1 < ny) & (min_d1[:n_grid] > j)
    d_ok = (j > 0) & (max_c[:n_grid] < j)
    drop = cp == 0
    rest = ~drop
    # the four checks in order; a column goes at the first open direction whose neighbour wins
    for ok, off in ((r_ok, ny), (l_ok, -ny), (u_ok, 1), (d_ok, -1)):
        cand = rest & ok
        nb = p + off
        nb_c = np.where(cand, nb, 0)
        win = cand & ((cnt[nb_c] > cp) | (nb_c < p))
        drop |= win
        rest &= ~win
    keep[:n_grid] = ~drop
    return keep


def dom_rebuild(e_ptr, e_col, keep):
    nc = len(keep)
    ne = len(e_ptr) - 1
    new_of = np.full(nc, -1, dtype=np.int64)
    kept = np.flatnonzero(keep).astype(np.int64)
    new_of[kept] = np.arange(len(kept), dtype=np.int64)
    old = kept.copy()
    m = new_of[e_col] >= 0
    rows = np.repeat(np.arange(ne, dtype=np.int64), np.diff(e_ptr))
    e_col2 = new_of[e_col][m]
    r2 = rows[m]
    e_ptr2 = np.zeros(ne + 1, dtype=np.int64)
    np.cumsum(np.bincount(r2, minlength=ne) if len(r2) else np.zeros(ne, dtype=np.int64), out=e_ptr2[1:])
    order = np.argsort(e_col2, kind='stable')
    c_el2 = r2[order]
    c_ptr2 = np.zeros(len(kept) + 1, dtype=np.int64)
    np.cumsum(np.bincount(e_col2, minlength=len(kept)) if len(e_col2) else np.zeros(len(kept), dtype=np.int64),
              out=c_ptr2[1:])
    return e_ptr2, np.ascontiguousarray(e_col2), c_ptr2, np.ascontiguousarray(c_el2), old, new_of
