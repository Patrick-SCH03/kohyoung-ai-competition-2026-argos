"""ARGOS - FOV planning for the ICCAS 2026 KohYoung AI Contest (AOI inspection cycle-time optimisation).

The grader imports this package and calls solve(data_folder) once per board. solve() reads input_component.csv,
input_size.csv and input_parameter.csv and writes output_fov.csv: one row per FOV in visit order (centre x, y and the
listed parts comp_idx). The score of a board is the checker's min over k of CT(k) + 10 k, plus the solve() time.

Modules, in pipeline order:
  geometry       Problem (the board, with each part's feasible-centre rectangle) and FOV
  cover          greedy rectangle-piercing covers on candidate grids, redundant-FOV removal, FOV elimination
  setcover       set-cover local search (RWLS) of the ordinary parts, on lattice and camera-bound boards
  setcover_py    the same kernels in plain Python, used only without numba
  order          visit order: rule blocks, travel paths, scored structural candidates, critical-FOV relocation
  refine         sliding free FOV centres inside their slack to shorten travel
  assign         moving heavy parts to earlier FOVs that also cover them
  anneal         order annealing, and the greedy order and member polishes
  polish_nb      the polishes as compiled loops
  release        moving the FOV that releases the last inspection job earlier
  release_tools  exact scheduling helpers of release
  ct_sim         exact pipeline simulation and core sweep (the checker's arithmetic)
  acceleration   compiled kernels of ct_sim
  verify         self-check with the checker's rules, and conservative repair
  atomic_io      atomic output replacement with bounded retries on Windows file contention
  parallel       the worker pool that finishes the other cover variants at the same time
  recover        the last stage on camera-bound plans: neighbouring FOV pairs re-split, members and centres together
                 (main process only; kept only when verified and strictly better)
  solver         the per-board pipeline and its anytime writes

Guarantees: every output written is complete and valid by the checker's rules (verified, repaired otherwise; a trivial
one-FOV-per-part plan is on disk from the first moment); the same inputs give the same plan when no time limit or
worker timing intervenes (seeded generators, fixed iteration budgets, no clock in the cover search); every stage has a
time limit inside time_budget; the package also works without numba (plain-Python kernels, smaller budgets).
"""
from .solver import solve
