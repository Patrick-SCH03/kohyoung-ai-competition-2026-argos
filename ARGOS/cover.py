"""Cover construction: where to put the FOV centres and which parts each FOV lists.

A FOV centre covers a non-big part exactly when it lies in that part's feasible-centre rectangle (geometry.py), so
covering is rectangle piercing:
  * GridCover: weighted max-coverage greedy on a regular candidate grid, with an incrementally maintained count map
    (how much part weight each grid point would cover); also places the tiles of big parts where they absorb the most
    neighbours.
  * remove_redundant: drops FOVs whose members other FOVs of the same step can take.
  * eliminate: dissolves a FOV when every member fits a neighbour that may still shift inside its members' common
    rectangle, with depth-2 ejection chains (move one member of a neighbour to a third FOV to make room).
  * recenter: puts each free FOV at the middle of its members' common rectangle (largest margin).
The compiled kernels (numba) run the same operations in the same order as the numpy code they replace and fall back
to it for the rest of the process on any dispatch failure."""
import numpy as np
try:
    from .geometry import TOL, FOV
except ImportError:
    from geometry import TOL, FOV

SIGN = np.array([1.0, -1.0, 1.0, -1.0])


def _weights(P, idx, mode, r):
    if isinstance(mode, str) and mode.startswith('invarea'):
        p = float(mode.split('^')[1]) if '^' in mode else 1.0
        wx = (P.hx[idx] - P.lx[idx]) + r
        wy = (P.hy[idx] - P.ly[idx]) + r
        w = (1.0 / (wx * wy)) ** p
        return w / w.max()
    return np.ones(len(idx), dtype=float)


try:  # compiled GridCover kernels: same per-cell subtraction order, same block maxima, same tie-break
    from numba import njit as _njit_g

    @_njit_g('void(float64[:, :], float64[:, :], int64, int64, int64, int64, int64)', cache=False, nogil=True)
    def _refresh_nb(count, bmax, B, i0, i1, j0, j1):
        nbx, nby = bmax.shape
        nx, ny = count.shape
        bi_hi = min(nbx, -(-i1 // B))
        bj_hi = min(nby, -(-j1 // B))
        for bi in range(i0 // B, bi_hi):
            for bj in range(j0 // B, bj_hi):
                m = -np.inf
                for i in range(bi * B, min((bi + 1) * B, nx)):
                    for j in range(bj * B, min((bj + 1) * B, ny)):
                        if count[i, j] > m:
                            m = count[i, j]
                bmax[bi, bj] = m

    @_njit_g('float64[:, ::1](int64, int64, int64[::1], int64[::1], int64[::1], int64[::1], float64[::1], int64[::1])',
             cache=False, nogil=True)
    def _count_nb(nx, ny, ix0, ix1, iy0, iy1, w, hc):
        # GridCover's count map: the same additions in the same order as the numpy path (four np.add.at passes in `hc`
        # order, then cumsum over axis 0 and over axis 1); count[i, j] only depends on diff[:i+1, :j+1].
        diff = np.zeros((nx + 1, ny + 1))
        for t in range(hc.shape[0]):
            q = hc[t]
            diff[ix0[q], iy0[q]] += w[q]
        for t in range(hc.shape[0]):
            q = hc[t]
            diff[ix1[q], iy0[q]] += -w[q]
        for t in range(hc.shape[0]):
            q = hc[t]
            diff[ix0[q], iy1[q]] += -w[q]
        for t in range(hc.shape[0]):
            q = hc[t]
            diff[ix1[q], iy1[q]] += w[q]
        for i in range(1, nx):
            for j in range(ny):
                diff[i, j] = diff[i - 1, j] + diff[i, j]
        count = np.empty((nx, ny))
        for i in range(nx):
            acc = diff[i, 0]
            count[i, 0] = acc
            for j in range(1, ny):
                acc = acc + diff[i, j]
                count[i, j] = acc
        return count

    @_njit_g('Tuple((int64, int64))(float64[:, :], float64[:, :], int64)', cache=False, nogil=True)
    def _best_nb(count, bmax, B):
        nbx, nby = bmax.shape
        nx, ny = count.shape
        g = -np.inf
        for bi in range(nbx):
            for bj in range(nby):
                if bmax[bi, bj] > g:
                    g = bmax[bi, bj]
        if g <= 1e-12:
            return -1, -1
        fi, fj = -1, -1
        for bi in range(nbx):
            for bj in range(nby):
                if bmax[bi, bj] != g:
                    continue
                # first cell (row-major) holding the block maximum, as np.argmax over the block
                bm, ba, bb = -np.inf, 0, 0
                for i in range(bi * B, min((bi + 1) * B, nx)):
                    for j in range(bj * B, min((bj + 1) * B, ny)):
                        if count[i, j] > bm:
                            bm, ba, bb = count[i, j], i, j
                if fi < 0 or ba < fi or (ba == fi and bb < fj):
                    fi, fj = ba, bb
        return fi, fj

    @_njit_g('int64(float64[:, :], float64[:, :], int64, boolean[::1], float64[::1], float64[::1], float64[::1], '
             'float64[::1], boolean[::1], int64[::1], int64[::1], int64[::1], int64[::1], float64[::1], float64, float64, float64, '
             'int64[::1])', cache=False, nogil=True)
    def _commit_nb(count, bmax, B, alive, lx, hx, ly, hy, has_cand, ix0, ix1, iy0, iy1, wts, x, y, tol, mem):
        # No allocation here: once count/bmax/alive start changing nothing can fail, so a caught error
        # always means the call never ran (members are written into the caller's buffer `mem`).
        n = lx.shape[0]
        c = 0
        for q in range(n):
            if alive[q] and lx[q] <= x + tol and x <= hx[q] + tol and ly[q] <= y + tol and y <= hy[q] + tol:
                mem[c] = q
                c += 1
        have = False
        b0 = b1 = b2 = b3 = 0
        for t in range(c):
            q = mem[t]
            if has_cand[q]:
                w = wts[q]
                for i in range(ix0[q], ix1[q]):
                    for j in range(iy0[q], iy1[q]):
                        count[i, j] -= w
                if not have:
                    b0, b1, b2, b3 = ix0[q], ix1[q], iy0[q], iy1[q]
                    have = True
                else:
                    b0 = min(b0, ix0[q])
                    b1 = max(b1, ix1[q])
                    b2 = min(b2, iy0[q])
                    b3 = max(b3, iy1[q])
        if have:
            _refresh_nb(count, bmax, B, b0, b1, b2, b3)
        for t in range(c):
            alive[mem[t]] = False
        return c

    _NUMBA_GRID = True
except Exception:  # numba unavailable: numpy GridCover methods below
    _NUMBA_GRID = False


def _disable_grid():
    # Numba checks argument types before running a kernel, so a failed call has not touched the
    # count map; the numpy methods continue from the same state for the rest of the process.
    global _NUMBA_GRID
    _NUMBA_GRID = False


class GridCover:
    """Weighted max-coverage greedy on a regular candidate grid with an incrementally
    maintained count map; supports constrained picks for big-component tiles."""

    def __init__(self, P, idx, r, weights='uniform'):
        self.P = P
        self.idx = np.asarray(idx, dtype=int)
        self.r = r
        n = len(self.idx)
        self.alive = np.ones(n, dtype=bool)
        if n == 0:
            self.gx = self.gy = np.zeros(0)
            self.count = np.zeros((0, 0))
            return
        self.lx, self.hx, self.ly, self.hy = P.lx[self.idx], P.hx[self.idx], P.ly[self.idx], P.hy[self.idx]
        self.wts = _weights(P, self.idx, weights, r)
        self.gx = np.arange(np.floor(self.lx.min() / r) * r, self.hx.max() + r, r)
        self.gy = np.arange(np.floor(self.ly.min() / r) * r, self.hy.max() + r, r)
        self.ix0 = np.searchsorted(self.gx, self.lx - TOL, 'left')
        self.ix1 = np.searchsorted(self.gx, self.hx + TOL, 'right')
        self.iy0 = np.searchsorted(self.gy, self.ly - TOL, 'left')
        self.iy1 = np.searchsorted(self.gy, self.hy + TOL, 'right')
        self.has_cand = (self.ix0 < self.ix1) & (self.iy0 < self.iy1)
        nx, ny = len(self.gx), len(self.gy)
        hc = np.where(self.has_cand)[0]
        self.count = None
        if _NUMBA_GRID:
            try:  # one compiled pass, same values (a failure before the kernel returns leaves nothing to undo)
                self.count = _count_nb(nx, ny, *(np.ascontiguousarray(a, dtype=np.int64) for a in
                                                 (self.ix0, self.ix1, self.iy0, self.iy1)),
                                       np.ascontiguousarray(self.wts, dtype=np.float64), hc.astype(np.int64))
            except Exception:
                _disable_grid()
                self.count = None
        if self.count is None:
            diff = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            np.add.at(diff, (self.ix0[hc], self.iy0[hc]), self.wts[hc])
            np.add.at(diff, (self.ix1[hc], self.iy0[hc]), -self.wts[hc])
            np.add.at(diff, (self.ix0[hc], self.iy1[hc]), -self.wts[hc])
            np.add.at(diff, (self.ix1[hc], self.iy1[hc]), self.wts[hc])
            self.count = np.cumsum(np.cumsum(diff, axis=0), axis=1)[:nx, :ny]
        self.B = 64
        self.nbx = -(-nx // self.B)
        self.nby = -(-ny // self.B)
        self.bmax = np.full((self.nbx, self.nby), -np.inf)
        self._refresh(0, nx, 0, ny)

    def _refresh(self, i0, i1, j0, j1):
        if _NUMBA_GRID:
            try:
                _refresh_nb(self.count, self.bmax, self.B, int(i0), int(i1), int(j0), int(j1))
                return
            except Exception:
                _disable_grid()
        B = self.B
        for bi in range(i0 // B, min(self.nbx, -(-i1 // B))):
            for bj in range(j0 // B, min(self.nby, -(-j1 // B))):
                self.bmax[bi, bj] = self.count[bi * B:(bi + 1) * B, bj * B:(bj + 1) * B].max()

    def best(self):
        if self.count.size == 0:
            return None
        if _NUMBA_GRID:
            try:
                i, j = _best_nb(self.count, self.bmax, self.B)
            except Exception:
                _disable_grid()
            else:
                if i < 0:
                    return None
                return float(self.gx[i]), float(self.gy[j]), float(self.count[i, j])
        g = self.bmax.max()
        if g <= 1e-12:
            return None
        B = self.B
        first = None
        for bi, bj in zip(*np.where(self.bmax == g)):
            sub = self.count[bi * B:(bi + 1) * B, bj * B:(bj + 1) * B]
            a, b = divmod(int(np.argmax(sub)), sub.shape[1])
            ij = (bi * B + a, bj * B + b)
            if first is None or ij < first:
                first = ij
        i, j = first
        return float(self.gx[i]), float(self.gy[j]), float(self.count[i, j])

    def best_in(self, x0, x1, y0, y1):
        if self.count.size == 0:
            return None
        i0 = int(np.searchsorted(self.gx, x0 - TOL, 'left'))
        i1 = int(np.searchsorted(self.gx, x1 + TOL, 'right'))
        j0 = int(np.searchsorted(self.gy, y0 - TOL, 'left'))
        j1 = int(np.searchsorted(self.gy, y1 + TOL, 'right'))
        if i0 >= i1 or j0 >= j1:
            return None
        sub = self.count[i0:i1, j0:j1]
        k = int(np.argmax(sub))
        a, b = divmod(k, sub.shape[1])
        return float(self.gx[i0 + a]), float(self.gy[j0 + b]), float(sub.flat[k])

    def commit(self, x, y):
        if len(self.idx) == 0:
            return np.zeros(0, dtype=int)
        if _NUMBA_GRID:
            buf = np.empty(len(self.idx), dtype=np.int64)
            try:
                c = _commit_nb(self.count, self.bmax, self.B, self.alive, self.lx, self.hx, self.ly, self.hy,
                               self.has_cand, self.ix0, self.ix1, self.iy0, self.iy1, self.wts, float(x), float(y), TOL, buf)
            except MemoryError:
                raise  # not a compiled-path failure: the numpy path would fail the same way
            except Exception:  # dispatch/type failure, raised before the kernel ran: state untouched
                _disable_grid()
            else:
                return self.idx[buf[:c]]
        m = self.alive & (self.lx <= x + TOL) & (x <= self.hx + TOL) & (self.ly <= y + TOL) & (y <= self.hy + TOL)
        mem = np.where(m)[0]
        box = None
        for q in mem:
            if self.has_cand[q]:
                self.count[self.ix0[q]:self.ix1[q], self.iy0[q]:self.iy1[q]] -= self.wts[q]
                b = (self.ix0[q], self.ix1[q], self.iy0[q], self.iy1[q])
                box = b if box is None else (min(box[0], b[0]), max(box[1], b[1]), min(box[2], b[2]), max(box[3], b[3]))
        if box is not None:
            self._refresh(*box)
        self.alive[mem] = False
        return self.idx[mem]

    def greedy(self):
        out = []
        while True:
            b = self.best()
            if b is None:
                break
            x, y, _ = b
            mem = self.commit(x, y)
            if len(mem) == 0:
                i = int(np.searchsorted(self.gx, x - TOL, 'left'))
                j = int(np.searchsorted(self.gy, y - TOL, 'left'))
                self.count[i, j] = 0.0
                self._refresh(i, i + 1, j, j + 1)
                continue
            out.append((x, y, mem))
        for q in np.where(self.alive)[0]:
            if not self.alive[q]:
                continue
            x = float((self.lx[q] + self.hx[q]) / 2.0)
            y = float((self.ly[q] + self.hy[q]) / 2.0)
            out.append((x, y, self.commit(x, y)))
        return out


def pierce_matrix(P, fovs, idx):
    idx = np.asarray(idx, dtype=int)
    C = np.zeros((len(fovs), len(idx)), dtype=bool)
    lx, hx, ly, hy = P.lx[idx], P.hx[idx], P.ly[idx], P.hy[idx]
    for f, fv in enumerate(fovs):
        C[f] = (lx <= fv.x + TOL) & (fv.x <= hx + TOL) & (ly <= fv.y + TOL) & (fv.y <= hy + TOL)
    return C


def remove_redundant(P, fovs):
    """Drop free FOVs whose members are all covered by other centres of the same step."""
    keep = [True] * len(fovs)
    for s in sorted(set(f.step for f in fovs)):
        fids = [f for f in range(len(fovs)) if fovs[f].step == s and not fovs[f].exclusive]
        comp = sorted({m for f in fids for m in fovs[f].members if not P.big[m]})
        if not comp:
            continue
        comp = np.asarray(comp, dtype=int)
        pos = {c: k for k, c in enumerate(comp)}
        C = pierce_matrix(P, [fovs[f] for f in fids], comp)
        owner = np.full(len(comp), -1, dtype=int)
        for r, f in enumerate(fids):
            for m in fovs[f].members:
                if not P.big[m]:
                    owner[pos[m]] = r
        active = np.ones(len(fids), dtype=bool)
        while True:
            cover_cnt = (C & active[:, None]).sum(axis=0)
            excl = np.zeros(len(fids), dtype=int)
            np.add.at(excl, owner[cover_cnt <= 1], 1)
            cands = [r for r in range(len(fids)) if active[r] and not fovs[fids[r]].fixed and excl[r] == 0
                     and not any(P.big[m] for m in fovs[fids[r]].members)]
            if not cands:
                break
            r = min(cands, key=lambda q: (len(fovs[fids[q]].members), q))
            active[r] = False
            fx, fy = fovs[fids[r]].x, fovs[fids[r]].y
            for m in fovs[fids[r]].members:
                k = pos[m]
                rows = np.where(C[:, k] & active)[0]
                best = min(rows, key=lambda q: ((fovs[fids[q]].x - fx) ** 2 + (fovs[fids[q]].y - fy) ** 2, q))
                fovs[fids[best]].members.append(m)
                owner[k] = best
            fovs[fids[r]].members = []
            keep[fids[r]] = False
    return [f for k, f in enumerate(fovs) if keep[k]]


def recenter(P, fovs):
    """Free FOV centre -> midpoint of its members' common rectangle (max margin)."""
    for f in fovs:
        if f.fixed or not f.members:
            continue
        if f.exclusive:
            f.x, f.y = float(P.cx[f.members[0]]), float(P.cy[f.members[0]])
            continue
        mem = np.asarray(f.members, dtype=int)
        if P.big[mem].any():
            continue
        f.x = float((P.lx[mem].max() + P.hx[mem].min()) / 2.0)
        f.y = float((P.ly[mem].max() + P.hy[mem].min()) / 2.0)
    return fovs


try:  # compiled inner kernels of eliminate(); control flow stays in Python (same order, same tie-breaks)
    from numba import njit as _njit

    @_njit('Tuple((int64, float64, float64, float64, float64))(float64[:, ::1], boolean[::1], float64, float64, float64, float64, float64)',
           cache=False, nogil=True)
    def _best_receiver_nb(Rt, cand, lxm, hxm, lym, hym, tol):
        """Direct move: first row with the largest feasible intersection area (argmax over area, -1 if infeasible)."""
        n = Rt.shape[0]
        best, best_a, any_feas = -1, -np.inf, False
        bx0 = bx1 = by0 = by1 = 0.0
        for k in range(n):
            x0 = Rt[k, 0] if Rt[k, 0] > lxm else lxm
            x1 = Rt[k, 1] if Rt[k, 1] < hxm else hxm
            y0 = Rt[k, 2] if Rt[k, 2] > lym else lym
            y1 = Rt[k, 3] if Rt[k, 3] < hym else hym
            feas = cand[k] and (x0 <= x1 + tol) and (y0 <= y1 + tol)
            if feas:
                any_feas = True
                dx = x1 - x0
                dy = y1 - y0
                a = (dx if dx > 0.0 else 0.0) * (dy if dy > 0.0 else 0.0)
            else:
                a = -1.0
            if a > best_a:
                best, best_a = k, a
                bx0, bx1, by0, by1 = x0, x1, y0, y1
        if not any_feas:
            return -1, 0.0, 0.0, 0.0, 0.0
        return best, bx0, bx1, by0, by1

    @_njit('Tuple((int64, int64, int64, float64, float64, float64, float64, float64, float64, float64, float64))'
           '(float64[:, ::1], boolean[::1], int64[::1], int64[::1], int64[::1], '
           'float64[::1], float64[::1], float64[::1], float64[::1], int64, float64)', cache=False, nogil=True)
    def _chain_nb(Rt, cand, near, flat, offs, lx, hx, ly, hy, m, tol):
        """Depth-2 ejection chain for component m over the ranked neighbours `near` (tentative member lists in
        flat/offs). Returns (position in near, q index, receiver k, k-rect, j-rect) or position -1."""
        n = Rt.shape[0]
        for pj in range(near.shape[0]):
            j = near[pj]
            s, e = offs[pj], offs[pj + 1]
            nq = e - s
            if nq < 2:
                continue
            # leave-one-out intersections via prefix / suffix maxima of (lx, -hx, ly, -hy): exact max operations
            pre = np.empty((nq, 4))
            suf = np.empty((nq, 4))
            for q in range(nq):
                c = flat[s + q]
                v0, v1, v2, v3 = lx[c], -hx[c], ly[c], -hy[c]
                if q:
                    v0 = v0 if v0 > pre[q - 1, 0] else pre[q - 1, 0]
                    v1 = v1 if v1 > pre[q - 1, 1] else pre[q - 1, 1]
                    v2 = v2 if v2 > pre[q - 1, 2] else pre[q - 1, 2]
                    v3 = v3 if v3 > pre[q - 1, 3] else pre[q - 1, 3]
                pre[q, 0], pre[q, 1], pre[q, 2], pre[q, 3] = v0, v1, v2, v3
            for q in range(nq - 1, -1, -1):
                c = flat[s + q]
                v0, v1, v2, v3 = lx[c], -hx[c], ly[c], -hy[c]
                if q < nq - 1:
                    v0 = v0 if v0 > suf[q + 1, 0] else suf[q + 1, 0]
                    v1 = v1 if v1 > suf[q + 1, 1] else suf[q + 1, 1]
                    v2 = v2 if v2 > suf[q + 1, 2] else suf[q + 1, 2]
                    v3 = v3 if v3 > suf[q + 1, 3] else suf[q + 1, 3]
                suf[q, 0], suf[q, 1], suf[q, 2], suf[q, 3] = v0, v1, v2, v3
            for q in range(nq):
                l0 = l1 = l2 = l3 = -np.inf
                if q > 0:
                    l0, l1, l2, l3 = pre[q - 1, 0], pre[q - 1, 1], pre[q - 1, 2], pre[q - 1, 3]
                if q < nq - 1:
                    l0 = l0 if l0 > suf[q + 1, 0] else suf[q + 1, 0]
                    l1 = l1 if l1 > suf[q + 1, 1] else suf[q + 1, 1]
                    l2 = l2 if l2 > suf[q + 1, 2] else suf[q + 1, 2]
                    l3 = l3 if l3 > suf[q + 1, 3] else suf[q + 1, 3]
                rx0, rx1, ry0, ry1 = l0, -l1, l2, -l3
                ax0 = rx0 if rx0 > lx[m] else lx[m]
                ax1 = rx1 if rx1 < hx[m] else hx[m]
                ay0 = ry0 if ry0 > ly[m] else ly[m]
                ay1 = ry1 if ry1 < hy[m] else hy[m]
                if not (ax0 <= ax1 + tol and ay0 <= ay1 + tol):
                    continue
                mp = flat[s + q]
                best, best_a, any_feas = -1, -np.inf, False
                bx0 = bx1 = by0 = by1 = 0.0
                for k in range(n):
                    kx0 = Rt[k, 0] if Rt[k, 0] > lx[mp] else lx[mp]
                    kx1 = Rt[k, 1] if Rt[k, 1] < hx[mp] else hx[mp]
                    ky0 = Rt[k, 2] if Rt[k, 2] > ly[mp] else ly[mp]
                    ky1 = Rt[k, 3] if Rt[k, 3] < hy[mp] else hy[mp]
                    feas = cand[k] and k != j and (kx0 <= kx1 + tol) and (ky0 <= ky1 + tol)
                    if feas:
                        any_feas = True
                        dx = kx1 - kx0
                        dy = ky1 - ky0
                        a = (dx if dx > 0.0 else 0.0) * (dy if dy > 0.0 else 0.0)
                    else:
                        a = -1.0
                    if a > best_a:
                        best, best_a = k, a
                        bx0, bx1, by0, by1 = kx0, kx1, ky0, ky1
                if not any_feas:
                    continue
                return pj, q, best, bx0, bx1, by0, by1, ax0, ax1, ay0, ay1
        return -1, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    _NUMBA_COVER = True
except Exception:  # numba unavailable: original numpy eliminate below
    _NUMBA_COVER = False


def eliminate(P, fovs, max_rounds=5, chains=True):
    """Dissolve FOVs whose members all fit into neighbouring FOVs (rules in _eliminate_py). The compiled version runs
    first; if it fails, the FOVs are restored and the numpy version runs from the same state."""
    global _NUMBA_COVER
    if _NUMBA_COVER:
        saved = [(f, f.members, f.x, f.y) for f in fovs]  # the pass only reassigns these three fields
        try:
            return _eliminate_nb(P, fovs, max_rounds, chains)
        except Exception:
            _NUMBA_COVER = False
            for f, members, x, y in saved:
                f.members, f.x, f.y = members, x, y
            fovs = [s[0] for s in saved]
    return _eliminate_py(P, fovs, max_rounds, chains)


def _eliminate_nb(P, fovs, max_rounds=5, chains=True):
    lx, hx, ly, hy = (np.ascontiguousarray(a, dtype=np.float64) for a in (P.lx, P.hx, P.ly, P.hy))
    big = P.big

    def rect_of(f):
        if f.fixed or any(big[m] for m in f.members):
            return (f.x, f.x, f.y, f.y)
        mem = np.asarray(f.members, dtype=int)
        return (float(lx[mem].max()), float(hx[mem].min()), float(ly[mem].max()), float(hy[mem].min()))

    fovs = list(fovs)
    for _ in range(max_rounds):
        n = len(fovs)
        R = np.ascontiguousarray(np.array([rect_of(f) for f in fovs], dtype=float).reshape(n, 4))
        steps = np.array([f.step for f in fovs])
        recv_ok = np.array([not f.exclusive for f in fovs])
        hasbig = np.array([any(big[m] for m in f.members) for f in fovs])
        movable = np.array([not (f.fixed or hasbig[k]) for k, f in enumerate(fovs)])
        fx = np.array([f.x for f in fovs], dtype=float)
        fy = np.array([f.y for f in fovs], dtype=float)
        alive = np.ones(n, dtype=bool)
        removed_any = False
        for f in sorted(range(n), key=lambda k: (len(fovs[k].members), k)):
            F = fovs[f]
            if not alive[f] or F.fixed or F.exclusive or not F.members or hasbig[f]:
                continue
            cand = alive & recv_ok & (steps == F.step)
            cand[f] = False
            if not cand.any():
                continue
            cand = np.ascontiguousarray(cand)
            Rt = R.copy()
            mem_t = {}
            ok = True
            near_pool = None
            for m in F.members:
                j, x0, x1, y0, y1 = _best_receiver_nb(Rt, cand, lx[m], hx[m], ly[m], hy[m], TOL)
                if j >= 0:
                    Rt[j] = (x0, x1, y0, y1)
                    mem_t.setdefault(j, list(fovs[j].members)).append(int(m))
                    continue
                if not chains:
                    ok = False
                    break
                mcx, mcy = (lx[m] + hx[m]) / 2.0, (ly[m] + hy[m]) / 2.0
                if near_pool is None:
                    near_pool = np.where(cand & movable)[0]
                if len(near_pool):
                    dist = (fx[near_pool] - mcx) ** 2 + (fy[near_pool] - mcy) ** 2
                    order = np.lexsort((near_pool, dist))[:8]
                    near = near_pool[order]
                else:
                    near = near_pool
                lists = [mem_t.get(int(j2), fovs[int(j2)].members) for j2 in near]
                acc, offs_l, flat_l = 0, [0], []
                for L in lists:
                    acc += len(L)
                    offs_l.append(acc)
                    flat_l.extend(L)
                offs = np.array(offs_l, dtype=np.int64)
                flat = np.array(flat_l, dtype=np.int64)
                pj, qi, k, kx0, kx1, ky0, ky1, ax0, ax1, ay0, ay1 = _chain_nb(
                    Rt, cand, np.ascontiguousarray(near, dtype=np.int64), flat, offs, lx, hx, ly, hy, int(m), TOL)
                if pj < 0:
                    ok = False
                    break
                j = int(near[pj])
                mj = lists[pj]
                mp = int(mj[qi])
                Rt[k] = (kx0, kx1, ky0, ky1)
                mem_t.setdefault(int(k), list(fovs[int(k)].members)).append(mp)
                mem_t[j] = [q for q in mj if q != mp] + [int(m)]
                Rt[j] = (ax0, ax1, ay0, ay1)
            if not ok:
                continue
            for j, lst in mem_t.items():
                fovs[j].members = lst
                R[j] = Rt[j]
                if not fovs[j].fixed:
                    fovs[j].x = float((R[j, 0] + R[j, 1]) / 2.0)
                    fovs[j].y = float((R[j, 2] + R[j, 3]) / 2.0)
                    fx[j], fy[j] = fovs[j].x, fovs[j].y
            F.members = []
            alive[f] = False
            removed_any = True
        fovs = [fv for k, fv in enumerate(fovs) if alive[k]]
        if not removed_any:
            break
    return fovs


def _eliminate_py(P, fovs, max_rounds=5, chains=True):
    """Dissolve a FOV when every member fits into neighbours that can still shift inside their
    members' common rectangle (depth-2 ejection chains when chains=True).  Fixed tiles receive
    only what their centre already covers; exclusive (fiducial) FOVs never receive.  Leave-one-out
    rectangles are cached per (receiver, member tuple); neighbour ranking is vectorised."""
    lx, hx, ly, hy = P.lx, P.hx, P.ly, P.hy
    big = P.big

    def rect_of(f):
        if f.fixed or any(big[m] for m in f.members):
            return (f.x, f.x, f.y, f.y)
        mem = np.asarray(f.members, dtype=int)
        return (float(lx[mem].max()), float(hx[mem].min()), float(ly[mem].max()), float(hy[mem].min()))

    loo_cache = {}     # (j, tuple(members)) -> loo rects : pure function of the member tuple

    def loo_rects(key, mj):
        r = loo_cache.get(key)
        if r is not None:
            return r
        mj_arr = np.asarray(mj, dtype=int)
        n = len(mj_arr)
        L = np.empty((n, 4))
        L[:, 0] = lx[mj_arr]
        L[:, 1] = -hx[mj_arr]
        L[:, 2] = ly[mj_arr]
        L[:, 3] = -hy[mj_arr]
        pre = np.maximum.accumulate(L, axis=0)
        suf = np.maximum.accumulate(L[::-1], axis=0)[::-1]
        loo = np.full((n, 4), -np.inf)
        loo[1:] = pre[:-1]
        np.maximum(loo[:-1], suf[1:], out=loo[:-1])
        r = (loo[:, 0], -loo[:, 1], loo[:, 2], -loo[:, 3])
        loo_cache[key] = r
        return r

    fovs = list(fovs)
    for _ in range(max_rounds):
        n = len(fovs)
        R = np.array([rect_of(f) for f in fovs], dtype=float).reshape(n, 4)
        steps = np.array([f.step for f in fovs])
        recv_ok = np.array([not f.exclusive for f in fovs])
        hasbig = np.array([any(big[m] for m in f.members) for f in fovs])
        movable = np.array([not (f.fixed or hasbig[k]) for k, f in enumerate(fovs)])
        fx = np.array([f.x for f in fovs], dtype=float)
        fy = np.array([f.y for f in fovs], dtype=float)
        alive = np.ones(n, dtype=bool)
        removed_any = False
        for f in sorted(range(n), key=lambda k: (len(fovs[k].members), k)):
            F = fovs[f]
            if not alive[f] or F.fixed or F.exclusive or not F.members or hasbig[f]:
                continue
            cand = alive & recv_ok & (steps == F.step)
            cand[f] = False
            if not cand.any():
                continue
            Rt = R.copy()
            mem_t = {}
            ok = True
            near_pool = None
            for m in F.members:
                Q = np.maximum(Rt * SIGN, (lx[m], -hx[m], ly[m], -hy[m]))
                x0, x1, y0, y1 = Q[:, 0], -Q[:, 1], Q[:, 2], -Q[:, 3]
                feas = cand & (x0 <= x1 + TOL) & (y0 <= y1 + TOL)
                if feas.any():
                    area = np.where(feas, np.maximum(x1 - x0, 0.0) * np.maximum(y1 - y0, 0.0), -1.0)
                    j = int(np.argmax(area))
                    Rt[j] = (x0[j], x1[j], y0[j], y1[j])
                    mem_t.setdefault(j, list(fovs[j].members)).append(int(m))
                    continue
                if not chains:
                    ok = False
                    break
                done = False
                mcx, mcy = (lx[m] + hx[m]) / 2.0, (ly[m] + hy[m]) / 2.0
                if near_pool is None:
                    near_pool = np.where(cand & movable)[0]
                if len(near_pool):
                    dist = (fx[near_pool] - mcx) ** 2 + (fy[near_pool] - mcy) ** 2
                    order = np.lexsort((near_pool, dist))[:8]
                    near = near_pool[order]
                else:
                    near = near_pool
                for j in near:
                    j = int(j)
                    mj = mem_t.get(j, fovs[j].members)
                    if len(mj) < 2:
                        continue
                    rx0, rx1, ry0, ry1 = loo_rects((j, tuple(mj)), mj)
                    ax0 = np.maximum(rx0, lx[m]); ax1 = np.minimum(rx1, hx[m])
                    ay0 = np.maximum(ry0, ly[m]); ay1 = np.minimum(ry1, hy[m])
                    for qi in np.where((ax0 <= ax1 + TOL) & (ay0 <= ay1 + TOL))[0]:
                        mp = int(mj[qi])
                        kx0 = np.maximum(Rt[:, 0], lx[mp]); kx1 = np.minimum(Rt[:, 1], hx[mp])
                        ky0 = np.maximum(Rt[:, 2], ly[mp]); ky1 = np.minimum(Rt[:, 3], hy[mp])
                        kf = cand & (kx0 <= kx1 + TOL) & (ky0 <= ky1 + TOL)
                        kf[j] = False
                        if not kf.any():
                            continue
                        karea = np.where(kf, np.maximum(kx1 - kx0, 0.0) * np.maximum(ky1 - ky0, 0.0), -1.0)
                        k = int(np.argmax(karea))
                        Rt[k] = (kx0[k], kx1[k], ky0[k], ky1[k])
                        mem_t.setdefault(k, list(fovs[k].members)).append(mp)
                        mem_t[j] = [q for q in mj if q != mp] + [int(m)]
                        Rt[j] = (float(ax0[qi]), float(ax1[qi]), float(ay0[qi]), float(ay1[qi]))
                        done = True
                        break
                    if done:
                        break
                if not done:
                    ok = False
                    break
            if not ok:
                continue
            for j, lst in mem_t.items():
                fovs[j].members = lst
                R[j] = Rt[j]
                if not fovs[j].fixed:
                    fovs[j].x = float((R[j, 0] + R[j, 1]) / 2.0)
                    fovs[j].y = float((R[j, 2] + R[j, 3]) / 2.0)
                    fx[j], fy[j] = fovs[j].x, fovs[j].y
            F.members = []
            alive[f] = False
            removed_any = True
        fovs = [fv for k, fv in enumerate(fovs) if alive[k]]
        if not removed_any:
            break
    return fovs


def cover_group(P, idx, r, weights, two_phase_side):
    """Cover non-big components of one step; optionally side components first, normals absorbed."""
    out = []
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return out
    step = int(P.step[idx[0]])
    if not two_phase_side:
        return [FOV(x, y, mem, step) for x, y, mem in GridCover(P, idx, r, weights).greedy()]
    side_idx = idx[P.side[idx] == 1]
    norm_idx = idx[P.side[idx] == 0]
    assigned = np.zeros(P.n, dtype=bool)
    for x, y, mem in GridCover(P, side_idx, r, weights).greedy():
        free = norm_idx[~assigned[norm_idx]]
        extra = free[P.contains(free, x, y)]
        members = list(mem) + list(extra)
        assigned[members] = True
        out.append(FOV(x, y, members, step))
    rest = norm_idx[~assigned[norm_idx]]
    for x, y, mem in GridCover(P, rest, r, weights).greedy():
        out.append(FOV(x, y, mem, step))
    return out
