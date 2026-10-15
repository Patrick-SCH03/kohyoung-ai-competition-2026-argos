# ARGOS algorithm design

[한국어](ALGORITHM.md) · **English**

This document explains how the ARGOS solver builds the inspection plan of one board, and why each design choice was made. Code locations are given in parentheses.

---

## 1. Problem and objective

The output for a board is a list of FOVs (one camera shot each). Every FOV has a centre and the list of parts it inspects, and **row order is visit order**.

The score comes from the checker's cycle-time model (`ct_sim.py`).

- **Core 0 (camera):** travels between FOVs and captures. Travel time is the larger of the x and y times on trapezoidal (or triangular) velocity profiles; a change of height step adds 5 s.
- **Reconstruction:** after capture, the less busy of cores 1 and 2 reconstructs the FOV in 3D.
- **Inspection:** the FOV's inspection jobs go, in `comp_idx` order, to the least-loaded inspection core (3 … k−1).
- **CT(k)** is the finish time of the last core; the score is `min over k [CT(k) + 10k]` (k = 4 … max_core) plus the computation time.

Three decisions are coupled. Fewer FOVs cut capture and reconstruction, but crowding parts into one FOV can starve the inspection cores; the visit order changes both travel and when inspection jobs are released. Minimising any single proxy can make the real score worse.

---

## 2. Design principles

1. **Judge by the real score only.** We re-implemented the checker's pipeline and core selection with the same order of operations, giving bit-identical values. Every candidate cover, order and polish move is scored by this simulator, and a change is kept **only if it is strictly better**.
2. **Stop by iteration count, not by the clock.** Computation time is part of the score, so search budgets are small and fixed. Searches stop on iteration counts with fixed seeds, so the same input gives the same plan (except where the arrival time of a parallel worker's reply changes the result). The clock is only a safety net.
3. **Always have a valid answer.** A trivial plan (one FOV per part) is written at the first moment and replaced atomically whenever a better plan exists. An exception in any stage keeps the plan already written.
4. **Separate cheap and expensive decisions.** When there are many candidates, such as several covers, a cheap proxy filters first and the exact score makes the final choice.

---

## 3. Geometry: piercing rectangles (`geometry.py`)

The **feasible-centre rectangle** of part i is the region where a FOV centre keeps the whole part inside the FOV:

```
lx = br_x − W/2,  hx = tl_x + W/2,  ly = br_y − H/2,  hy = tl_y + H/2
```

W, H are the effective window. Side-view parts (side=1) use the half-size window with the same centre. Parts larger than their window (big) cannot fit one FOV and are covered by the union of several FOV windows.

Seen this way, **covering = choosing points that pierce every rectangle**. One point is one FOV.

---

## 4. Cover: fewer FOVs

### 4.1 Greedy cover (`cover.py`, `solver.build_fovs`)

Each height step is covered separately, since parts of different steps cannot share a FOV.

1. Each fiducial gets its own FOV, centred on the fiducial.
2. Tiles of big parts are placed where they absorb the most neighbouring parts.
3. The remaining parts are covered by a **weighted max-coverage greedy on a candidate grid** (GridCover). Every grid point keeps the part weight it would cover, updated incrementally after each pick.
4. **Redundant FOV removal:** a FOV whose parts other FOVs can all take is dropped.
5. **FOV elimination:** a FOV is dissolved when all its parts fit neighbouring FOVs, whose centres may still move inside their parts' common rectangle; depth-2 ejection chains (move one of a neighbour's parts to a third FOV to make room) are tried too.
6. **Recentring:** each remaining FOV moves to the middle of its parts' common rectangle (largest margin).

The greedy is sensitive to weights and grid, so we build **four cover variants** with different settings (weights 1/area^1.5, 1/area or uniform; grid 0.5 mm or 1 mm; side-view parts first or not). The cover with the smallest total capture time (a proxy) is finished. With parallel workers, the other variants are finished at the same time and chosen by exact score (Section 7).

### 4.2 Set-cover local search (`setcover.py`)

The greedy is weak on two kinds of boards: boards whose parts sit on lattice lines, and boards where the camera (capture and travel) is the bottleneck. There the ordinary parts are re-covered as a **set-cover problem**.

- **Candidates (columns):** the crossings of a grid built from clustered rectangle end coordinates (hx, hy), plus the centres of the greedy cover. The columns a part covers form contiguous ranges in x and y, so the incidence is stored sparse (CSR both ways).
- **Search:** RWLS (Gao, Weise & Li 2015) with row weighting, configuration checking and oldest-first ties, a fixed seed and an iteration budget. Every row keeps the sum of the indices of its selected covering columns, so the one other column covering it is found without scanning the row.
- **2-D dominance reduction:** a candidate is unnecessary when every part covering it also covers a neighbouring candidate. Searching the reduced instance as well gives a smaller alternative cover.
- **When it runs:** only on boards where at least half of the ordinary parts lie on lattice lines, or when the finished plan is camera-bound. A new cover is kept only if it passes the rule check and its exact score is **strictly lower**.

---

## 5. Order (`order.py`)

- **Rule blocks:** fiducials → barcodes (type 1) → normal parts (type 0); inside the blocks, FOVs are grouped by step to keep the 5 s step changes few.
- **Paths:** nearest neighbour on the exact travel-time matrix, improved by 2-opt and Or-opt.
- **Portfolio of structural candidates:** the shortest path is not always best. We also build heavy FOVs first, the light FOV last, a heavy prelude, reversed step order, deadline repair and similar candidates, score all of them with the simulator and keep the best.
- **Critical-FOV relocation:** the FOV whose inspection finishes last is moved to the best earlier position in its block, repeated while the score improves.

---

## 6. Finishing polish

Small moves collect the gains a finished plan still holds. Each is kept only if the exact score is strictly better.

| Stage | What it does | Code |
|---|---|---|
| Centre refinement | Slides FOV centres inside their parts' common rectangle towards the previous and next FOV (a safety margin kept) | `refine.py` |
| Heavy-part reassignment | Moves heavy inspection jobs to an earlier FOV that also covers the part, so inspection starts sooner | `assign.py` |
| Release relocation | Moves the FOV that releases the job ending the schedule earlier inside its same-step run | `release.py` |
| Order annealing | On inspection-bound boards, anneals the order of the normal FOVs (insert, reverse, swap, block moves) | `anneal.py` |
| Order and member polish | The same moves at a near-zero temperature, and normal parts moved to other FOVs that cover them | `anneal.py`, `polish_nb.py` |
| Pair re-split | On camera-bound plans, pools the parts of two neighbouring FOVs and re-splits them by two new centres; only the changed part of the schedule is re-simulated | `recover.py` |

---

## 7. Parallel portfolio (`parallel.py`)

The contest Q&A allows multiprocessing inside the solver, and import time is not scored. So **worker processes start at import** (spawn on Windows, fork on Linux).

- As soon as a board is read, cover variants 1–3 (plus one extra variant when enough cores exist) go to the workers; the main process solves variant 0.
- After writing the main plan, the solver takes a worker's plan that arrived in time if its exact score is **strictly lower** (the main plan wins ties).
- **CPU guard:** a calibration at start checks that every process gets CPU, and every board records the workers' CPU share. When the machine is too busy the pool pauses and retries a few boards later. If a worker dies or hangs, the pool stops and the solver runs alone.

---

## 8. Checking and writing (`verify.py`, `atomic_io.py`)

- **Self-check:** before writing, every rule (full coverage, half window for side parts, union cover of big parts, no step mixing, fiducials alone, centred and first, type order) is checked with the checker's tolerances, and violations are repaired conservatively.
- **Atomic writes:** the plan is written to a temporary file and swapped in, so an interrupted write never leaves a broken file. A swap blocked by a Windows file lock is retried; if it still fails, the valid plan already on disk is kept.

---

## 9. Acceleration (`acceleration.py`, `polish_nb.py`, `setcover.py`)

The simulator, the cover kernels and the polishing loops are compiled with numba. The compiled code performs **the same operations in the same order** as the Python code and gives the same floating-point values. Without numba, or if compilation fails, pure-Python kernels of the same algorithms take over.

---

## 10. Tried but not adopted

Measured during development on fresh boards from the official generator (score per board, lower is better).

| Attempt | Result | Decision |
|---|---|---|
| Exact minimum cover (integer programming) | about −0.10 from the cover itself, but the solve time costs more | a loss once time counts |
| Grid candidates + MILP selection of FOVs | −0.01 to −0.02 even with zero computation time | no gain |
| Picking the best of several plans from cheap features | gains on development boards vanished or reversed on new boards | does not generalise |
| Restarts and two-branch searches | less variance, but on slower machines the time cost exceeds the gain | not adopted |

Most of the remaining room is in the cover (the number of FOVs). For a fixed cover, the gains left in order, polishing and positions are close to saturated.
