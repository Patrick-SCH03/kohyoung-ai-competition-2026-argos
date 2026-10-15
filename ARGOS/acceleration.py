"""Compiled kernels of the checker's pipeline simulation (ct_sim.simulate).

The kernels run the same IEEE double operations in the same order as the Python reference (no fastmath), so every
result is the same float. simulate_fast packs the inputs once per input set (several k of one core sweep reuse them)
and falls back to the Python loop for the rest of the process on any dispatch failure. _simulate_seq_nb evaluates a
visit order directly on the full per-FOV tables (ct_sim.SeqEval), without building permuted copies. The caller
(ct_sim) supplies its reference implementation, so this module stays standalone.
"""


try:  # compiled kernel: same operations in the same order as the Python loop below (IEEE double, no fastmath)
    import numpy as _np
    from numba import njit as _njit

    @_njit('UniTuple(float64, 2)(float64[:, ::1], float64[::1], float64[::1], float64[::1], int64[::1], int64, int64[::1], float64)',
           cache=False, nogil=True)
    def _simulate_nb(cm, t_img, t_recon, flat, offs, max_core, steps, z_delay):
        n = t_img.shape[0]
        tl = _np.empty(max_core)
        for k in range(max_core):
            tl[k] = t_img[0]
        for i in range(n):
            if i:
                t0 = tl[0] + cm[i - 1, i]
                if steps[i - 1] != steps[i]:
                    t0 += z_delay
                t0 += t_img[i]
                tl[0] = t0
                for k in range(1, max_core):
                    if tl[k] < t0:
                        tl[k] = t0
            rc = 1 if tl[1] <= tl[2] else 2
            tl[rc] += t_recon[i]
            ready = tl[rc]
            for q in range(offs[i], offs[i + 1]):
                duration = flat[q]
                best = 3
                value = tl[3] if tl[3] > ready else ready
                for k in range(4, max_core):
                    candidate = tl[k] if tl[k] > ready else ready
                    if candidate < value:
                        value = candidate
                        best = k
                tl[best] = value + duration
        m = tl[0]
        for k in range(1, max_core):
            if tl[k] > m:
                m = tl[k]
        return m, tl[0]

    @_njit('Tuple((float64, float64, int64))(float64[:, ::1], float64[::1], float64[::1], float64[::1], int64[::1], int64, int64[::1], float64)',
           cache=False, nogil=True)
    def _simulate_crit_nb(cm, t_img, t_recon, flat, offs, max_core, steps, z_delay):
        """ct_sim.simulate(..., want_crit=True): also the sequence position of the FOV whose task ends last."""
        n = t_img.shape[0]
        tl = _np.empty(max_core)
        last = _np.zeros(max_core, dtype=_np.int64)
        for k in range(max_core):
            tl[k] = t_img[0]
        for i in range(n):
            if i:
                t0 = tl[0] + cm[i - 1, i]
                if steps[i - 1] != steps[i]:
                    t0 += z_delay
                t0 += t_img[i]
                tl[0] = t0
                for k in range(1, max_core):
                    if tl[k] < t0:
                        tl[k] = t0
            rc = 1 if tl[1] <= tl[2] else 2
            tl[rc] += t_recon[i]
            last[rc] = i
            ready = tl[rc]
            for q in range(offs[i], offs[i + 1]):
                duration = flat[q]
                best = 3
                value = tl[3] if tl[3] > ready else ready
                for k in range(4, max_core):
                    candidate = tl[k] if tl[k] > ready else ready
                    if candidate < value:
                        value = candidate
                        best = k
                tl[best] = value + duration
                last[best] = i
        ct = tl[0]
        kmax = 0
        for k in range(1, max_core):
            if tl[k] > ct:
                ct = tl[k]
                kmax = k
        crit = n - 1 if kmax == 0 else last[kmax]
        return ct, tl[0], crit

    _NUMBA = True
except Exception:  # numba missing or failed to compile: keep the pure-Python path
    _NUMBA = False

try:  # its own block, so a failure here leaves the kernels above in use (SeqEval then builds the copies)
    if not _NUMBA:
        raise ImportError('numba kernels unavailable')

    @_njit('UniTuple(float64, 2)(float64[:, ::1], int64[::1], float64[::1], float64[::1], float64[::1], int64[::1], '
           'int64, int64[::1], float64)', cache=False, nogil=True)
    def _simulate_seq_nb(cm, seq, t_img, t_recon, flat, offs, max_core, steps, z_delay):
        """_simulate_nb on the visit order `seq` of FOV indices, reading cm[seq[i-1], seq[i]], t_img[seq[i]],
        t_recon[seq[i]], the jobs flat[offs[seq[i]]:offs[seq[i] + 1]] and steps[seq[i]] in place of the permuted copies
        (the same values in the same operation order, so the same floats)."""
        n = seq.shape[0]
        tl = _np.empty(max_core)
        for k in range(max_core):
            tl[k] = t_img[seq[0]]
        for i in range(n):
            f = seq[i]
            if i:
                g = seq[i - 1]
                t0 = tl[0] + cm[g, f]
                if steps[g] != steps[f]:
                    t0 += z_delay
                t0 += t_img[f]
                tl[0] = t0
                for k in range(1, max_core):
                    if tl[k] < t0:
                        tl[k] = t0
            rc = 1 if tl[1] <= tl[2] else 2
            tl[rc] += t_recon[f]
            ready = tl[rc]
            for q in range(offs[f], offs[f + 1]):
                duration = flat[q]
                best = 3
                value = tl[3] if tl[3] > ready else ready
                for k in range(4, max_core):
                    candidate = tl[k] if tl[k] > ready else ready
                    if candidate < value:
                        value = candidate
                        best = k
                tl[best] = value + duration
        m = tl[0]
        for k in range(1, max_core):
            if tl[k] > m:
                m = tl[k]
        return m, tl[0]

    _SEQ = True
except Exception:
    _SEQ = False

_PACK_CACHE = [None, None]  # (input objects, packed arrays): the sweep calls the kernel for several k on the same inputs


def _pack(cm, t_img, t_recon, t_comps, fov_steps):
    # Contract: callers must not mutate cm / t_img / t_recon / t_comps (including inner lists) / fov_steps in place
    # between calls; a changed schedule must be passed as new objects. Identity is checked against objects kept alive
    # in the cache, so a freed object's id can never be reused here. Every call site (ct_sim.plan_times, order,
    # anneal, assign, release) builds fresh objects per evaluation.
    key = (cm, t_img, t_recon, t_comps, fov_steps)
    prev = _PACK_CACHE[0]
    if prev is not None and all(a is b for a, b in zip(prev, key)):
        return _PACK_CACHE[1]
    acc, offs_l, flat_l = 0, [0], []
    for c in t_comps:  # integer offsets: identical to np.cumsum of the lengths
        acc += len(c)
        offs_l.append(acc)
        flat_l.extend(c)
    offs = _np.array(offs_l, dtype=_np.int64)
    flat = _np.array(flat_l, dtype=_np.float64)
    packed = (_np.ascontiguousarray(cm, dtype=_np.float64), _np.asarray(t_img, dtype=_np.float64),
              _np.asarray(t_recon, dtype=_np.float64), flat, offs, _np.asarray(fov_steps, dtype=_np.int64))
    _PACK_CACHE[0], _PACK_CACHE[1] = key, packed
    return packed


def _disable():
    # A compiled call failed (dispatch/type error before execution): use the Python loop from now on.
    global _NUMBA
    _NUMBA = False
    _PACK_CACHE[0] = _PACK_CACHE[1] = None


def simulate_fast(cm, t_img, t_recon, t_comps, max_core, fov_steps, z_delay=5., want_crit=False, *, reference):
    """ct_sim.simulate: the compiled kernels when available, else the Python loop below (want_crit, or fewer than four
    cores: the caller's reference implementation)."""
    if want_crit and _NUMBA and max_core >= 4 and len(t_img):
        try:  # any failure of the compiled path (dispatch/type error) falls back to the reference for this call
            c, ti, tr, flat, offs, st = _pack(cm, t_img, t_recon, t_comps, fov_steps)
            ct, c0, crit = _simulate_crit_nb(c, ti, tr, flat, offs, int(max_core), st, float(z_delay))
            return ct, c0, int(crit)
        except Exception:
            _disable()
    if want_crit or max_core < 4:
        return reference(cm, t_img, t_recon, t_comps, max_core, fov_steps, z_delay, want_crit)
    n = len(t_img)
    if not n:
        return 0., 0.
    if _NUMBA:
        try:
            c, ti, tr, flat, offs, st = _pack(cm, t_img, t_recon, t_comps, fov_steps)
            return _simulate_nb(c, ti, tr, flat, offs, int(max_core), st, float(z_delay))
        except Exception:
            _disable()
    tl = [float(t_img[0])] * max_core
    others = range(1, max_core)
    inspection = range(4, max_core)
    for i in range(n):
        if i:
            t0 = tl[0] + float(cm[i - 1, i])
            if fov_steps[i - 1] != fov_steps[i]:
                t0 += z_delay
            t0 += float(t_img[i])
            tl[0] = t0
            for k in others:
                if tl[k] < t0:
                    tl[k] = t0
        rc = 1 if tl[1] <= tl[2] else 2
        tl[rc] += float(t_recon[i])
        ready = tl[rc]
        for duration in t_comps[i]:
            best = 3
            value = tl[3] if tl[3] > ready else ready
            for k in inspection:
                candidate = tl[k] if tl[k] > ready else ready
                if candidate < value:
                    value, best = candidate, k
            tl[best] = value + duration
    return max(tl), tl[0]
