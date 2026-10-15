"""Slide free FOV centres inside their members' common rectangle to shorten travel to the
previous/next FOV (exact max(Tx,Ty) model, safety margin kept)."""
import numpy as np
try:
    from .geometry import TOL
except ImportError:
    from geometry import TOL

MARGIN = 1e-3     # mm kept away from the exact rectangle edge whenever the slack allows


def _axis_t(d, v, a):
    th = v * v / a
    if d >= th:
        return 2.0 * v / a + (d - th) / v
    return 2.0 * np.sqrt(d / a)


def travel(P, x0, y0, x1, y1):
    return max(_axis_t(abs(x1 - x0), P.vx, P.ax), _axis_t(abs(y1 - y0), P.vy, P.ay))


def free_rect(P, f):
    """Common feasible rectangle of a free FOV's members, shrunk by MARGIN; None if fixed."""
    if f.fixed and any(P.big[m] for m in f.members):
        return None
    if not f.members or f.exclusive:
        return None
    mem = np.asarray(f.members, dtype=int)
    if P.big[mem].any():
        return None
    x0, x1 = float(P.lx[mem].max()), float(P.hx[mem].min())
    y0, y1 = float(P.ly[mem].max()), float(P.hy[mem].min())
    if x1 - x0 > 2 * MARGIN:
        x0, x1 = x0 + MARGIN, x1 - MARGIN
    else:
        x0 = x1 = (x0 + x1) / 2.0
    if y1 - y0 > 2 * MARGIN:
        y0, y1 = y0 + MARGIN, y1 - MARGIN
    else:
        y0 = y1 = (y0 + y1) / 2.0
    return x0, x1, y0, y1


try:  # compiled refine: same candidate order, same travel arithmetic, same acceptance threshold
    from numba import njit as _njit

    @_njit('float64(float64, float64, float64)', cache=False, nogil=True)
    def _axis_t_nb(d, v, a):
        th = v * v / a
        if d >= th:
            return 2.0 * v / a + (d - th) / v
        return 2.0 * np.sqrt(d / a)

    @_njit('float64(float64, float64, float64, float64, float64, float64, float64, float64)', cache=False, nogil=True)
    def _travel_nb(x0, y0, x1, y1, vx, ax, vy, ay):
        tx = _axis_t_nb(abs(x1 - x0), vx, ax)
        ty = _axis_t_nb(abs(y1 - y0), vy, ay)
        return tx if tx >= ty else ty

    @_njit('float64(float64[::1], float64[::1], float64[:, ::1], boolean[::1], int64, int64, float64, float64, float64, float64)',
           cache=False, nogil=True)
    def _refine_nb(xs, ys, rect, has, rounds, lattice, vx, ax, vy, ay):
        n = xs.shape[0]
        gain_total = 0.0
        cx = np.empty(9 + lattice * lattice)
        cy = np.empty(9 + lattice * lattice)
        gl_x = np.empty(lattice)
        gl_y = np.empty(lattice)
        for _ in range(rounds):
            gain = 0.0
            for pos in range(n):
                if not has[pos]:
                    continue
                x0, x1, y0, y1 = rect[pos, 0], rect[pos, 1], rect[pos, 2], rect[pos, 3]
                hp = pos > 0
                hn = pos + 1 < n
                if not hp and not hn:
                    continue
                px = xs[pos - 1] if hp else 0.0
                py = ys[pos - 1] if hp else 0.0
                nx_ = xs[pos + 1] if hn else 0.0
                ny_ = ys[pos + 1] if hn else 0.0
                c = 0
                cx[c] = xs[pos]; cy[c] = ys[pos]; c += 1
                cx[c] = (x0 + x1) / 2; cy[c] = (y0 + y1) / 2; c += 1
                cx[c] = x0; cy[c] = y0; c += 1
                cx[c] = x0; cy[c] = y1; c += 1
                cx[c] = x1; cy[c] = y0; c += 1
                cx[c] = x1; cy[c] = y1; c += 1
                if hp:
                    cx[c] = min(max(px, x0), x1); cy[c] = min(max(py, y0), y1); c += 1
                if hn:
                    cx[c] = min(max(nx_, x0), x1); cy[c] = min(max(ny_, y0), y1); c += 1
                if hp and hn:
                    mx = (px + nx_) / 2
                    my = (py + ny_) / 2
                    cx[c] = min(max(mx, x0), x1); cy[c] = min(max(my, y0), y1); c += 1
                if lattice > 1 and (x1 > x0 or y1 > y0):
                    sx = (x1 - x0) / (lattice - 1)
                    sy = (y1 - y0) / (lattice - 1)
                    for t in range(lattice):     # np.linspace: i * step + start, last point = stop
                        gl_x[t] = t * sx + x0
                        gl_y[t] = t * sy + y0
                    gl_x[lattice - 1] = x1
                    gl_y[lattice - 1] = y1
                    for a in range(lattice):
                        for b in range(lattice):
                            cx[c] = gl_x[a]; cy[c] = gl_y[b]; c += 1
                cur = 0.0
                if hp:
                    cur += _travel_nb(px, py, xs[pos], ys[pos], vx, ax, vy, ay)
                if hn:
                    cur += _travel_nb(xs[pos], ys[pos], nx_, ny_, vx, ax, vy, ay)
                bx, by, bc = xs[pos], ys[pos], cur
                for t in range(c):
                    cc = 0.0
                    if hp:
                        cc += _travel_nb(px, py, cx[t], cy[t], vx, ax, vy, ay)
                    if hn:
                        cc += _travel_nb(cx[t], cy[t], nx_, ny_, vx, ax, vy, ay)
                    if cc < bc - 1e-12:
                        bx, by, bc = cx[t], cy[t], cc
                if bc < cur - 1e-12:
                    xs[pos] = bx
                    ys[pos] = by
                    gain += cur - bc
            gain_total += gain
            if gain < 1e-9:
                break
        return gain_total

    _NUMBA_REFINE = True
except Exception:  # numba unavailable: Python refine below
    _NUMBA_REFINE = False


def refine_positions(P, fovs, seq, rounds=2, lattice=5):
    """In-place: slide free FOV centres to shorten the path. Returns total gain (s)."""
    if _NUMBA_REFINE and len(seq):
        n = len(seq)
        xs = np.array([float(fovs[k].x) for k in seq])
        ys = np.array([float(fovs[k].y) for k in seq])
        rect = np.zeros((n, 4))
        has = np.zeros(n, dtype=np.bool_)
        for pos, k in enumerate(seq):
            r = free_rect(P, fovs[k])
            if r is not None:
                rect[pos] = r
                has[pos] = True
        try:
            g = _refine_nb(xs, ys, rect, has, int(rounds), int(lattice), float(P.vx), float(P.ax), float(P.vy), float(P.ay))
        except Exception:  # FOVs are untouched until write-back below: rerun the original from the same state
            _disable_refine()
        else:
            for pos, k in enumerate(seq):
                if has[pos] and (xs[pos] != fovs[k].x or ys[pos] != fovs[k].y):
                    fovs[k].x, fovs[k].y = float(xs[pos]), float(ys[pos])
            return g
    return _refine_positions_py(P, fovs, seq, rounds, lattice)


def _disable_refine():
    global _NUMBA_REFINE
    _NUMBA_REFINE = False


def _refine_positions_py(P, fovs, seq, rounds=2, lattice=5):
    """In-place: slide free FOV centres to shorten the path. Returns total gain (s)."""
    gain_total = 0.0
    n = len(seq)
    if n == 0:
        return 0.0
    rects = {k: free_rect(P, fovs[k]) for k in seq}
    for _ in range(rounds):
        gain = 0.0
        for pos, k in enumerate(seq):
            r = rects[k]
            if r is None:
                continue
            x0, x1, y0, y1 = r
            f = fovs[k]
            prv = fovs[seq[pos - 1]] if pos > 0 else None
            nxt = fovs[seq[pos + 1]] if pos + 1 < n else None
            if prv is None and nxt is None:
                continue

            def cost(x, y):
                c = 0.0
                if prv is not None:
                    c += travel(P, prv.x, prv.y, x, y)
                if nxt is not None:
                    c += travel(P, x, y, nxt.x, nxt.y)
                return c

            cands = [(f.x, f.y), ((x0 + x1) / 2, (y0 + y1) / 2), (x0, y0), (x0, y1), (x1, y0), (x1, y1)]
            pts = []
            if prv is not None:
                pts.append((prv.x, prv.y))
            if nxt is not None:
                pts.append((nxt.x, nxt.y))
            if prv is not None and nxt is not None:
                pts.append(((prv.x + nxt.x) / 2, (prv.y + nxt.y) / 2))
            for (px, py) in pts:
                cands.append((min(max(px, x0), x1), min(max(py, y0), y1)))
            if lattice > 1 and (x1 > x0 or y1 > y0):
                for gx in np.linspace(x0, x1, lattice):
                    for gy in np.linspace(y0, y1, lattice):
                        cands.append((float(gx), float(gy)))
            cur = cost(f.x, f.y)
            best, best_c = (f.x, f.y), cur
            for (cx, cy) in cands:
                c = cost(cx, cy)
                if c < best_c - 1e-12:
                    best, best_c = (cx, cy), c
            if best_c < cur - 1e-12:
                f.x, f.y = float(best[0]), float(best[1])
                gain += cur - best_c
        gain_total += gain
        if gain < 1e-9:
            break
    return gain_total
