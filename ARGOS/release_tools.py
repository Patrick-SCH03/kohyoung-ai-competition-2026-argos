"""Exact scheduling helpers of the release relocation (release.py): candidate positions, the release trace and the
full core sweep of a permutation, with the checker's dispatch rules and arithmetic."""
import numpy as np

try:
    from .ct_sim import simulate
except ImportError:
    from ct_sim import simulate


def select_fixed(base_plan, candidates):
    """The candidate with the lowest fixed-k plan value (then the lowest insert position), if strictly below
    base_plan."""
    best = min(candidates, key=lambda row: (row['fixed_plan'], row['insert']), default=None)
    return best if best is not None and best['fixed_plan'] < base_plan - 1e-9 else None


def earlier_positions(position, types, steps, limit=12):
    """Earlier positions inside the same-step run of normal FOVs that ends at `position` (at most `limit`, spread
    evenly over the run); none for a barcode or fiducial FOV."""
    if limit < 1:
        raise ValueError('At least one position is required')
    if types[position] != 0:
        return []
    first = position
    while first and types[first - 1] == 0 and steps[first - 1] == steps[position]:
        first -= 1
    positions = list(range(first, position))
    if len(positions) <= limit:
        return positions
    if limit == 1:
        return positions[:1]
    return [positions[round(i * (len(positions) - 1) / (limit - 1))] for i in range(limit)]


def trace_release(cm, images, recon, jobs, k, steps, eligible):
    """The checker's schedule at k cores, also tracking for every inspection core the eligible job that started its
    current busy period (the core was idle when the job was released; the earliest release wins). Returns (CT, core-0
    end, the visit position of that job on a core that ends the schedule, or None)."""
    if k < 4:
        raise ValueError('At least four cores are required')
    if len(eligible) != len(images):
        raise ValueError('Eligibility and image lengths differ')
    if not images:
        return 0., 0., None
    timeline = [float(images[0])] * k
    tags = [None] * k
    for i, batch in enumerate(jobs):
        if i:
            camera = timeline[0] + float(cm[i - 1, i])
            if steps[i - 1] != steps[i]:
                camera += 5.
            camera += float(images[i])
            timeline[0] = camera
            for core in range(1, k):
                if timeline[core] < camera:
                    timeline[core] = camera
                    tags[core] = None
        rc = 1 if timeline[1] <= timeline[2] else 2
        timeline[rc] += float(recon[i])
        ready = timeline[rc]
        for duration in batch:
            best = 3
            earliest = timeline[3] if timeline[3] > ready else ready
            for core in range(4, k):
                value = timeline[core] if timeline[core] > ready else ready
                if value < earliest:
                    earliest, best = value, core
            previous = timeline[best]
            if previous < ready:
                tags[best] = None
            if previous <= ready and eligible[i]:
                release = (ready, -duration, i)
                if tags[best] is None or release < tags[best]:
                    tags[best] = release
            timeline[best] = earliest + duration
    ct = max(timeline)
    critical = [tags[core] for core in range(3, k)
                if timeline[core] == ct and tags[core] is not None]
    return ct, timeline[0], min(critical)[2] if critical else None


def score_sequence(cm, images, recon, jobs, steps, max_core, seq):
    """Full core sweep of the permutation `seq`: (min CT(k) + 10 k, its k, its CT, the number of simulations)."""
    if not seq:
        return 0., 4, 0., 0
    ordered_cm = cm[np.ix_(seq, seq)]
    img = [images[i] for i in seq]
    rec = [recon[i] for i in seq]
    work = [jobs[i] for i in seq]
    st = [steps[i] for i in seq]
    best = (float('inf'), 4, float('inf'))
    for core in range(4, max_core + 1):
        ct, _ = simulate(ordered_cm, img, rec, work, core, st)
        value = (ct + 10 * core, core, ct)
        if value < best:
            best = value
    return best + (max_core - 3,)
