"""Order search on a finished plan under the scoring objective (min over k of CT(k) + 10 k).

anneal_order: simulated annealing on the visit order. The structural order candidates keep step blocks intact; on
inspection-bound boards the tail shrinks when heavy load is spread so the inspection cores never starve, which can
require splitting or relocating whole step runs (paying a +5 s transition). Only the normal-FOV suffix is permuted,
so the type order is preserved.
polish_order: a near-zero-temperature version of the same moves inside the barcode and normal blocks; finished plans
still hold order-only gains that the annealing leaves (its tail gate skips camera-bound plans, and its start
temperature walks away from a good order).
polish_members: moves normal parts between the FOVs that cover them (same step), keeping the visit order except that
an emptied FOV leaves it.
All three are deterministic (seeded generators, fixed evaluation counts); the deadline is only a safety net. The
polishes run as compiled loops (polish_nb) with a larger budget when numba is available, else as the Python loops
here."""
import math
import time
import random
import numpy as np
try:  # simulate and sweep are also imported here: analysis tools patch these module attributes
    from .ct_sim import simulate, sweep, SeqEval  # noqa: F401
    from . import polish_nb as _pnb
except ImportError:
    from ct_sim import simulate, sweep, SeqEval  # noqa: F401
    import polish_nb as _pnb


def anneal_order(P, fovs, seq, cm, t_img, t_comps, steps, evals=None, budget_ms=350.0, min_tail=2.0, t0=0.5,
                 max_extend=1.5, min_gain=0.1, scale='none', deadline=None, seed=0):
    """Simulated annealing of the normal-FOV suffix of `seq` (insert, reverse, swap, block and same-step-run moves),
    scored at the current best k and re-swept on every new best. Skipped when the inspection tail is short or the plan
    is already close to the balanced-load bound. The budget scales with the gap to that bound ('gap'); it is spent in
    chunks: stop after a quarter without gain, extend by half while the last chunk still gained >= min_gain. Returns
    the best order when strictly better than `seq`, else `seq`."""
    n = len(seq)
    types = [fovs[k].type_of(P) for k in range(len(fovs))]
    fn = next((i for i, k in enumerate(seq) if types[k] == 0), n)
    if n - fn < 4:
        return seq

    ev = SeqEval(cm, t_img, [P.recon] * len(t_img), t_comps, steps)  # evaluates orders without permuted copies

    def full(sq):
        r = ev.sweep(sq, P.max_core)
        return r[0], r[1], r[2] - r[3]

    def at_k(sq, k):
        return ev.simulate(sq, k)[0] + 10.0 * k

    cur = list(seq)
    cur_s, k, tail = full(cur)
    if tail < min_tail:
        return seq
    # skip when the schedule is already load-bound: tail close to the balanced-load bound
    core0 = cur_s - 10.0 * k - tail
    load = sum(sum(t_comps[q]) for q in seq)
    lb = max(core0, t_img[seq[0]] + P.recon + load / max(k - 3, 1))
    if (cur_s - 10.0 * k) - lb < min_tail:
        return seq
    if evals is None:
        ncomp = sum(len(t_comps[q]) for q in seq)
        est_ms = 3e-5 * n * n + 2e-4 * ncomp
        gap = (cur_s - 10.0 * k) - lb
        if scale == 'size':
            budget_ms *= min(1.0, max(0.3, n / 40.0))
        elif scale == 'gap':
            budget_ms *= min(1.0, max(0.3, gap / 6.0))
        evals = int(min(4000, max(500, budget_ms / est_ms)))
    start_s = cur_s
    best, best_s = list(cur), cur_s
    rng = random.Random(seed)
    # chunked budget: stop after a quarter without any gain; otherwise keep extending in
    # half-budget chunks while the last chunk still improved the best by >= min_gain
    probe = evals // 4
    chunk = max(1, evals // 2)
    limit = evals
    max_evals = evals * max_extend
    chunk_ref = start_s
    it = 0
    while it < limit:
        if deadline is not None and (it & 63) == 0 and time.perf_counter() > deadline:
            break
        if it == probe and best_s >= start_s - 1e-9:
            break
        if it == limit - 1 and limit < max_evals:
            if best_s < chunk_ref - min_gain:
                chunk_ref = best_s
                limit = min(max_evals, limit + chunk)
        T = t0 * (1.0 - it / limit) + 1e-3
        it += 1
        cand = list(cur)
        mv = rng.random()
        i = rng.randrange(fn, n)
        if mv < 0.35:
            j = rng.randrange(fn, n)
            cand.insert(j, cand.pop(i))
        elif mv < 0.60:
            j = rng.randrange(fn, n)
            i, j = min(i, j), max(i, j)
            cand[i:j + 1] = cand[i:j + 1][::-1]
        elif mv < 0.72:
            j = rng.randrange(fn, n)
            cand[i], cand[j] = cand[j], cand[i]
        elif mv < 0.87:
            L = rng.randint(2, 4)
            if i + L > n:
                continue
            blk = cand[i:i + L]
            del cand[i:i + L]
            j = rng.randrange(fn, len(cand) + 1)
            cand[j:j] = blk
        else:
            s = steps[cand[i]]
            a = i
            while a > fn and steps[cand[a - 1]] == s and i - a < 12:
                a -= 1
            b = i
            while b + 1 < n and steps[cand[b + 1]] == s and b - i < 12:
                b += 1
            blk = cand[a:b + 1]
            del cand[a:b + 1]
            j = rng.randrange(fn, len(cand) + 1)
            cand[j:j] = blk
        s = at_k(cand, k)
        d = s - cur_s
        if d < 0 or rng.random() < math.exp(-d / T):
            cur, cur_s = cand, s
            if s < best_s - 1e-9:
                best = list(cand)
                best_s, k2, _ = full(best)
                if k2 != k:
                    k, cur_s = k2, best_s
    return best if best_s < full(seq)[0] - 1e-9 else seq



def polish_order(P, fovs, seq, cm, t_img, t_comps, steps, stall=1000, cap=8000, deadline=None, seed=0, nb_scale=0):
    """Greedy polish of a finished visit order under the scoring objective: the annealing moves at a near-zero
    temperature (T = 1e-3, equal scores accepted), which found more per evaluation than any hotter schedule. Moves stay
    inside the barcode block and the normal block, so the type order and the fiducial FOVs (first, in index order) are
    untouched. cm, t_img, t_comps, steps are tables over all FOV indices; seq is a visit order. Returns the best order
    when its full score is strictly lower, else seq."""
    n = len(seq)
    types = [fovs[q].type_of(P) for q in seq]
    blocks, a = [], 0
    for i in range(1, n + 1):
        if i == n or types[i] != types[a]:
            if types[a] != 2 and i - a >= 2:
                blocks.append((a, i))
            a = i
    if not blocks or cap <= 0:
        return seq
    weights = [b - a for a, b in blocks]
    ev = SeqEval(cm, t_img, [P.recon] * len(t_img), t_comps, steps)
    if ev.fast is None:  # without the compiled evaluator an evaluation costs about ten times more than the gain pays
        return seq
    cur = list(seq)
    cur_s, k = ev.sweep(cur, P.max_core)[:2]
    start_s = cur_s
    best, best_s = list(cur), cur_s
    if nb_scale > 0 and _pnb.OK:  # the same loop compiled, nb_scale times the budget
        try:
            got, got_s = _pnb.order(ev.fast, cur, blocks, weights, k, cur_s, P.max_core, stall * nb_scale,
                                    cap * nb_scale, deadline, seed)
            return got if got_s < start_s - 1e-9 else seq
        except Exception:  # the Python loop below, with the plain budget
            _pnb.OK = False
    rng = random.Random(seed)
    T = 1e-3
    last = 0
    for it in range(cap):
        if it - last > stall:
            break
        if deadline is not None and (it & 63) == 0 and time.perf_counter() > deadline:
            break
        a, b = rng.choices(blocks, weights)[0]
        cand = list(cur)
        i = rng.randrange(a, b)
        mv = rng.random()
        if mv < 0.35:
            j = rng.randrange(a, b)
            cand.insert(j, cand.pop(i))
        elif mv < 0.60:
            j = rng.randrange(a, b)
            i, j = min(i, j), max(i, j)
            cand[i:j + 1] = cand[i:j + 1][::-1]
        elif mv < 0.72:
            j = rng.randrange(a, b)
            cand[i], cand[j] = cand[j], cand[i]
        elif mv < 0.87:
            L = rng.randint(2, 4)
            if i + L > b:
                continue
            blk = cand[i:i + L]
            del cand[i:i + L]
            j = rng.randrange(a, b - L + 1)
            cand[j:j] = blk
        else:
            s = steps[cand[i]]
            lo = i
            while lo > a and steps[cand[lo - 1]] == s and i - lo < 12:
                lo -= 1
            hi = i
            while hi + 1 < b and steps[cand[hi + 1]] == s and hi - i < 12:
                hi += 1
            blk = cand[lo:hi + 1]
            del cand[lo:hi + 1]
            j = rng.randrange(a, b - len(blk) + 1)
            cand[j:j] = blk
        s = ev.simulate(cand, k)[0] + 10.0 * k
        if ev.fast is None:  # the compiled evaluator failed: stop here
            break
        d = s - cur_s
        if d < 0 or rng.random() < math.exp(-d / T):
            cur, cur_s = cand, s
            if s < best_s - 1e-9:
                full, k2 = ev.sweep(cand, P.max_core)[:2]
                best, best_s, last = list(cand), full, it
                if k2 != k:
                    k, cur_s = k2, full
    return best if best_s < start_s - 1e-9 else seq


def polish_members(P, fovs, seq, cm, t_img, t_comps, steps, stall=250, cap=2000, deadline=None, seed=0, nb_scale=0):
    """Greedy moves of normal parts between the FOVs that cover them (same step), on a finished plan (order kept,
    except that an emptied FOV leaves it). cm, t_img, t_comps, steps are tables over all FOV indices (not modified); seq
    is the visit order. Returns (members per FOV, visit order) when the full score is strictly lower, else None."""
    if cap <= 0 or not seq:
        return None
    t_img, t_comps = list(t_img), [list(c) for c in t_comps]
    ev = SeqEval(cm, t_img, [P.recon] * len(t_img), t_comps, steps)
    if ev.fast is None:
        return None
    ti, flat, offs = ev.fast[2], ev.fast[4], ev.fast[5]
    mem = [list(f.members) for f in fovs]
    where = {m: q for q, ms in enumerate(mem) for m in ms}
    live = [q for q in seq if not fovs[q].exclusive]
    parts = [m for q in live for m in mem[q] if not P.big[m] and P.type[m] == 0]
    if not parts or len(live) < 2:
        return None
    xs = np.asarray([fovs[q].x for q in live], dtype=np.float64)
    ys = np.asarray([fovs[q].y for q in live], dtype=np.float64)
    inside = P.contains(np.asarray(parts)[:, None], xs, ys)
    cover = {}
    for r, m in enumerate(parts):
        gs = [live[c] for c in np.nonzero(inside[r])[0].tolist() if fovs[live[c]].step == P.step[m]]
        if len(gs) >= 2:
            cover[m] = gs
    cand = sorted(cover)
    if not cand:
        return None
    if nb_scale > 0 and _pnb.OK:  # the same loop compiled, nb_scale times the budget
        try:
            s0, k0 = ev.sweep(list(seq), P.max_core)[:2]
            got = _pnb.members(ev.fast, P, mem, list(seq), cand, cover, k0, s0, P.max_core, stall * nb_scale,
                               cap * nb_scale, deadline, seed)
            if got is not _pnb.FALLBACK:
                return got
        except Exception:  # the Python loop below, with the plain budget
            _pnb.OK = False

    def img(q):
        return max(P.capture, P.side_capture) if any(P.side[x] == 1 for x in mem[q]) else P.capture

    def move(m, g, j=None):
        a = where[m]
        src, dst = mem[a], mem[g]
        i = src.index(m)
        src.pop(i)
        if j is None:
            j = next((c for c, x in enumerate(dst) if (-P.time[x], x) > (-P.time[m], m)), len(dst))
        dst.insert(j, m)
        ps = int(offs[a]) + i
        val = flat[ps]
        if a < g:
            ins = int(offs[g]) - 1 + j
            flat[ps:ins] = flat[ps + 1:ins + 1]
            flat[ins] = val
            offs[a + 1:g + 1] -= 1
        else:
            ins = int(offs[g]) + j
            flat[ins + 1:ps + 1] = flat[ins:ps]
            flat[ins] = val
            offs[g + 1:a + 1] += 1
        for q in (a, g):
            ti[q] = t_img[q] = img(q)
            t_comps[q] = [float(P.time[x]) for x in mem[q]]
        where[m] = g
        dropped = None
        if not src:
            dropped = order.index(a)
            order.pop(dropped)
        return m, a, i, dropped

    order = list(seq)
    cur_s, k = ev.sweep(order, P.max_core)[:2]
    start_s = best_s = cur_s
    best = None
    rng = random.Random(seed)
    T = 1e-3
    last = 0
    for it in range(cap):
        if it - last > stall:
            break
        if deadline is not None and (it & 63) == 0 and time.perf_counter() > deadline:
            break
        m = rng.choice(cand)
        a = where[m]
        gs = [g for g in cover[m] if g != a and mem[g]]
        if not gs:
            continue
        m, a, i, dropped = move(m, rng.choice(gs))
        s = ev.simulate(order, k)[0] + 10.0 * k
        if ev.fast is None:  # the compiled evaluator failed: stop here
            break
        d = s - cur_s
        if d < 0 or rng.random() < math.exp(-d / T):
            cur_s = s
            if s < best_s - 1e-9:
                full, k2 = ev.sweep(order, P.max_core)[:2]
                best_s, last = full, it
                best = ([list(x) for x in mem], list(order))
                if k2 != k:
                    k, cur_s = k2, full
        else:
            if dropped is not None:
                order.insert(dropped, a)
            move(m, a, i)
    return best if best is not None and best_s < start_s - 1e-9 else None
