"""Release relocation: the last finishing stage of the visit order.

The inspection job that ends the schedule (traced back to the FOV whose reconstruction released it) often waits for a
busy inspection core. Moving that FOV earlier inside its own same-step run of normal FOVs releases the job sooner.
Each round traces the release source at the current k, tries up to 12 earlier positions at that k, and keeps the best
move only if a full core sweep confirms a strictly lower score; at most max_rounds rounds, never past the deadline.
The scheduling inputs are never modified: every candidate is a new permutation."""
import time

import numpy as np

try:
    from .release_tools import select_fixed, earlier_positions, trace_release, score_sequence
    from .ct_sim import simulate
except ImportError:
    from release_tools import select_fixed, earlier_positions, trace_release, score_sequence
    from ct_sim import simulate


def improve_release_sequence(cm, images, recon, jobs, types, steps, max_core,
                             deadline=float('inf'), max_rounds=8):
    n = len(images)
    if max_core < 4 or max_rounds < 1:
        raise ValueError('At least four cores and one round are required')
    if {len(recon), len(jobs), len(types), len(steps)} != {n} or cm.shape != (n, n):
        raise ValueError('Scheduling input lengths differ')
    seq = list(range(n))
    baseline = score_sequence(cm, images, recon, jobs, steps, max_core, seq)
    current = baseline[:3]
    result = dict(baseline=current, sequence=seq, value=current, prefixes=[], proposals=[],
                  fixed_evaluations=0, baseline_core_simulations=baseline[3],
                  accepted_core_simulations=0, trace_replays=0, stop_reason='round_limit')

    def ordered(order):
        return (cm[np.ix_(order, order)], [images[i] for i in order], [recon[i] for i in order],
                [jobs[i] for i in order], [steps[i] for i in order])

    for round_number in range(1, max_rounds + 1):
        if time.perf_counter() >= deadline:
            result['stop_reason'] = 'deadline'
            break
        matrix, img, rec, work, st = ordered(seq)
        ty = [types[i] for i in seq]
        eligible = [i > 0 and ty[i] == ty[i - 1] == 0 and st[i] == st[i - 1] for i in range(n)]
        traced_ct, _, source = trace_release(matrix, img, rec, work, current[1], st, eligible)
        result['trace_replays'] += 1
        assert abs(traced_ct - current[2]) < 1e-9
        if source is None:
            result['stop_reason'] = 'no_target'
            break
        rest = seq[:source] + seq[source + 1:]
        candidates = []
        for insert in earlier_positions(source, ty, st):
            if time.perf_counter() >= deadline:
                result['stop_reason'] = 'deadline'
                break
            proposal = rest[:insert] + [seq[source]] + rest[insert:]
            matrix2, img2, rec2, work2, st2 = ordered(proposal)
            fixed_ct, _ = simulate(matrix2, img2, rec2, work2, current[1], st2)
            record = dict(round=round_number, source=seq[source], source_position=source,
                          insert=insert, k=current[1], fixed_plan=fixed_ct + 10 * current[1],
                          fixed_ct=fixed_ct, sequence=proposal)
            candidates.append(record)
            result['proposals'].append(record)
            result['fixed_evaluations'] += 1
        if time.perf_counter() >= deadline:
            result['stop_reason'] = 'deadline'
            break
        best = select_fixed(current[0], candidates)
        if best is None:
            result['stop_reason'] = 'no_improvement'
            break
        proposal = best['sequence']
        checked = score_sequence(cm, images, recon, jobs, steps, max_core, proposal)
        result['accepted_core_simulations'] += checked[3]
        assert checked[0] <= best['fixed_plan'] + 1e-9 and checked[0] < current[0] - 1e-9
        assert sorted(proposal) == list(range(n))
        assert [types[i] for i in proposal] == list(types) and [steps[i] for i in proposal] == list(steps)
        current, seq = checked[:3], proposal
        result['prefixes'].append(dict(round=round_number, source=best['source'],
            source_position=source, insert=best['insert'], sequence=seq, value=current))
        result['sequence'], result['value'] = seq, current
    return result
