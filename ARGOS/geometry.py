"""Problem container and geometric primitives.

Semantics mirror utils/checker.py exactly:
  * effective window per component: full FOV (side=0) or half FOV (side=1), same centre
  * 'big' = component exceeds its effective window (strict >) -> must be tiled
  * a FOV centre c covers a non-big component i  <=>  c lies in the component's
    "feasible-centre rectangle"  [lx_i, hx_i] x [ly_i, hy_i]
        lx = br_x - W/2 , hx = tl_x + W/2 , ly = br_y - H/2 , hy = tl_y + H/2
    (W, H = effective window).  Covering == piercing these rectangles.
"""
import math
import numpy as np

TOL = 1e-6          # internal tolerance; checker grants 1e-5 in our favour


class Problem:
    def __init__(self, comp, size, param):
        n = len(comp)
        self.n = n
        self.tlx = comp['tl_x'].to_numpy(dtype=float)
        self.tly = comp['tl_y'].to_numpy(dtype=float)
        self.brx = comp['br_x'].to_numpy(dtype=float)
        self.bry = comp['br_y'].to_numpy(dtype=float)
        self.type = comp['type'].to_numpy(dtype=int)
        self.side = (comp['side'].to_numpy() == 1).astype(int) if 'side' in comp.columns else np.zeros(n, dtype=int)
        self.step = comp['step'].to_numpy(dtype=int) if 'step' in comp.columns else np.zeros(n, dtype=int)
        self.time = comp['time'].to_numpy(dtype=float) if 'time' in comp.columns else np.zeros(n, dtype=float)

        self.fw = float(size['fov_size'].iloc[0])
        self.fh = float(size['fov_size'].iloc[1])
        self.pw = float(size['pcb_size'].iloc[0])
        self.ph = float(size['pcb_size'].iloc[1])

        self.capture = float(param['capture_time'].iloc[0])
        self.recon = float(param['recon_time'].iloc[0])
        if 'side_capture_time' in param.columns:
            self.side_capture = float(param['side_capture_time'].iloc[0])
        else:
            self.side_capture = self.capture / 2.0
        self.max_core = max(4, int(param['max_core'].iloc[0]))
        self.vx = float(param['v_x'].iloc[0])
        self.vy = float(param['v_y'].iloc[0])
        self.ax = float(param['a_x'].iloc[0])
        self.ay = float(param['a_y'].iloc[0])

        self.w = self.brx - self.tlx
        self.h = self.bry - self.tly
        self.cx = (self.tlx + self.brx) / 2.0
        self.cy = (self.tly + self.bry) / 2.0
        self.W = np.where(self.side == 1, self.fw / 2.0, self.fw)
        self.H = np.where(self.side == 1, self.fh / 2.0, self.fh)
        self.big = (self.w > self.W) | (self.h > self.H)
        # feasible-centre rectangles (meaningful for non-big only)
        self.lx = self.brx - self.W / 2.0
        self.hx = self.tlx + self.W / 2.0
        self.ly = self.bry - self.H / 2.0
        self.hy = self.tly + self.H / 2.0

    # ------------------------------------------------------------------
    def contains(self, idx, x, y, tol=TOL):
        """Boolean mask over idx: centre (x, y) covers component idx[k] (non-big semantics)."""
        return ((self.lx[idx] <= x + tol) & (x <= self.hx[idx] + tol) &
                (self.ly[idx] <= y + tol) & (y <= self.hy[idx] + tol))

    def tiles(self, i):
        """Minimal contiguous tiling of big component i with its effective window.
        Tiles start at the component's top-left; the last row/col is clamped to end
        exactly at the bottom-right, so the union covers the component."""
        W, H = float(self.W[i]), float(self.H[i])
        w, h = float(self.w[i]), float(self.h[i])
        nx = math.ceil(w / W) if w > W else 1
        ny = math.ceil(h / H) if h > H else 1
        xs = [min(self.tlx[i] + W / 2.0 + k * W, self.brx[i] - W / 2.0) for k in range(nx)]
        ys = [min(self.tly[i] + H / 2.0 + k * H, self.bry[i] - H / 2.0) for k in range(ny)]
        if nx == 1:
            xs = [self.cx[i]]
        if ny == 1:
            ys = [self.cy[i]]
        return [(x, y) for x in xs for y in ys]


class FOV:
    """One camera shot. members = component indices listed in this FOV (comp_idx)."""
    __slots__ = ('x', 'y', 'members', 'fixed', 'step', 'exclusive')

    def __init__(self, x, y, members, step, fixed=False, exclusive=False):
        self.x = float(x)
        self.y = float(y)
        self.members = list(int(m) for m in members)
        self.step = int(step)
        self.fixed = bool(fixed)          # tile of a big component: centre must not move
        self.exclusive = bool(exclusive)  # fiducial FOV: never receives other members

    def type_of(self, P):
        return int(max((P.type[m] for m in self.members), default=0))
