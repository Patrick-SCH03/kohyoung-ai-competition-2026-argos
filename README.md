# 4th Koh Young AI Contest — ICCAS 2026, Team ARGOS (3rd place)

**English** | [한국어](README.ko.md)

3rd place (Bronze) at the 4th Koh Young AI Contest. Oral presentation at ICCAS 2026 (Sapporo, Japan). [Certificate](assets/certificate.jpg)

| Members |
|---|
| [@Patrick-SCH03](https://github.com/Patrick-SCH03) |
| [@xjx06127](https://github.com/xjx06127) |
| [@naeun211](https://github.com/naeun211) |
| [@rosuasu0721-cloud](https://github.com/rosuasu0721-cloud) |
| [@hoon0503](https://github.com/hoon0503) |

## Problem

AOI (Automatic Optical Inspection) equipment inspects the components on a PCB automatically.
A camera head moves across the PCB and captures one FOV (Field of View, the rectangular area taken in a single shot) at a time. Each FOV then goes through 3D reconstruction and inspection.
Several cores share this work in parallel.

```
time ─────────────────────────────────────────────▶

Core 0  [cap1] move [cap2] move [cap3]
Core 1        [rec1]                  [rec3]
Core 2                    [rec2]
Core 3              [ins1a]     [ins2a]     [ins3a]
Core 4              [ins1b]     [ins2b]     [ins3b]  ← CT
```

cap = Capture, rec = 3D Recon, ins = Inspection. Numbers are FOVs; a and b are components in that FOV. The 3D Recon of a FOV runs while the camera moves to the next one, and its components are inspected as soon as the Recon ends.

- **Core 0**: move + Capture
- **Core 1, 2**: 3D Recon (fixed time per FOV)
- **Core 3 and up**: component Inspection (time differs per component; each goes to the least busy core)
- **CT (Cycle Time)**: the time the last core finishes

### Three decisions

| | Decision | Output |
|---|---|---|
| **STEP 1** | **FOV Placement**: where and how many FOVs | `x`, `y` |
| **STEP 2** | **Component Assignment**: which components each FOV inspects | `comp_idx` |
| **STEP 3** | **Inspection Order**: the FOV visit order (Fiducial → Barcode → Normal) | row order |

The three depend on each other, which makes the problem NP-hard.
Fewer FOVs mean fewer captures but more inspection packed into each FOV, and the order changes both the move time and when each inspection can start.

### Score

Lower is better, summed over boards.

```
score = CT(k*) + 10k* + RT                    # RT = solver runtime (s)
k*    = argmin_{k=4..max_core} CT(k) + 10k    # the checker picks the core count
```

### Rules

- Every component must be fully inside a FOV. A `side=1` component must fit in the **Side FOV**, which is half the size of the FOV and shares its center.
- Components larger than the FOV are covered by multiple FOVs with no gaps (tiling).
- A FOV cannot mix **steps** (height levels), and each step change between consecutive FOVs adds **+5 s**.
- The order must be **Fiducial (alone, first) → Barcode → Normal**.

## Core idea: judge every plan by the real score

<h3 align="center">Simulation-based solver that judges every plan by the real score</h3>

Fewest FOVs and Shortest path are good guides, not goals.

- **Shortest path trap**: an inspection starts only after its FOV has been captured and reconstructed. Capture a heavy FOV late and the CT grows, even if the camera finishes early.
- **Fewest FOVs limit**: cutting the FOV count to the minimum with ILP did not improve the score in our solver. It only increased RT.

So **greedy covers and the shortest path build the first plan, and the real score decides what to keep.**
The solver contains a simulator that reproduces the checker's computation. For every candidate it computes CT for each k = 4..max_core and compares `min_k CT(k) + 10k`.

### Pipeline

```
5 greedy covers       # one per process
    │
    ▼
best of the 7 candidate orders
    │
    ▼
center refinement     # shift FOV centers toward the previous and next FOVs, re-run the candidates
    │
    ▼
finish                # critical relocate, heavy reassign, simulated annealing, release
    │
    ▼
polish                # small changes to the order and component assignment
    │
    ▼
best of the 5 plans → output_fov.csv
```

From center refinement to polish, a change is kept only if the score improves. Each stage is described in STEP 3.

## STEP 1, 2: FOV Cover

### Valid box

For each component, find the rectangle where a FOV center must lie for the component to fit inside the FOV.

```
lx = br_x - W/2,   hx = tl_x + W/2,   ly = br_y - H/2,   hy = tl_y + H/2
W, H = FOV size (half for side components)
```

FOV Placement then becomes **stabbing every valid box with as few points as possible** (rectangle piercing, NP-hard). Each point is one FOV. Each step is solved separately.

### Weighted greedy

For every grid point, sum the weights of the components a FOV there would cover. Place a FOV at the highest sum and remove the covered components. Repeat until every component is covered.

```
w_i = ( 1 / ((width_i + r)(height_i + r)) )^p
width_i, height_i = size of the valid box of component i (hx - lx, hy - ly)
r                 = grid spacing
```

- `(width + r)(height + r) / r²` is the number of grid points inside the valid box, i.e. the FOV centers that can cover the component. The fewer there are (big and side components), the larger the weight.
- Leaving such a component for later often costs one extra FOV just for it.
- A larger `p` takes these components first. `p = 0` is the plain greedy.

No single setting is best on every board, so we build 5 variants.

| Variant | p | r | Side first |
|---|---|---|---|
| V0 | 1.5 | 0.5 mm | |
| V1 | 1 | 0.5 mm | ✓ |
| V2 | 1 | 1.0 mm | |
| V3 | 0 | 1.0 mm | ✓ |
| V4 | 0 | 0.5 mm | |

### After the greedy

- **Redundant removal**: delete a FOV whose components all fit in other FOVs already.
- **Elimination**: delete a FOV whose components can all move to nearby FOVs. A receiving FOV may shift its center inside its free box (the overlap of its components' valid boxes). If a component has nowhere to go, a nearby FOV hands one of its own components to a third FOV to make room (depth-2 ejection chain).
- **Recenter**: move each remaining FOV center to the middle of its free box.

### RWLS

On boards where components line up on a grid and on camera-bound boards (the camera is the bottleneck), RWLS (Row Weighting Local Search) tries to build one more cover with fewer FOVs. The plan switches to it only if its full plan, order included, scores lower.

## STEP 3: Inspection Order

The order is built around **when inspections finish**, not around move distance.

### Why not the shortest path

An inspection starts only after its FOV has been captured and reconstructed. If a heavy FOV is captured late, everything waits on that one inspection and the CT grows, even if the camera finishes early. This is why the shortest path is not always the best.

### Steps

1. **Blocks**: group FOVs by type (Fiducial / Barcode / Normal) and step, and order the blocks to keep step changes few.
2. **Path inside a block**: nearest neighbor → 2-opt → Or-opt on the move time.
3. **7 candidate orders**, and the lowest-score one is kept.
   - Shortest path: the path from step 2 as is
   - Light last: the lightest FOV goes last
   - Heaviest first: the heaviest FOV goes first
   - Heavy front: several heavy FOVs go to the front
   - Deadline fix: a FOV whose inspection cannot finish before the last capture moves earlier
   - Reversed steps: reverse the order of the step blocks
   - Prelude: put 1, 2 or 4 heavy FOVs first, even at the cost of a step change
4. **Block insertion, center refinement**: try placing short step runs inside other runs, shift FOV centers toward the previous and next FOVs in the order to cut move time, then compare the candidates again.
5. **Finish**:
   - critical relocate: move the FOV holding the last-finishing inspection earlier
   - heavy reassign: move heavy components to a FOV that also covers them and is captured earlier
   - simulated annealing: shuffle the order
   - release: move the FOV that releases the last-finishing inspection earlier within its step run
6. **Polish**: small changes to the order and to which FOV inspects each component.

Every change in steps 4 to 6 is kept **only if the score improves**.

## Results

Official evaluation data, 96 boards.

| | ARGOS | Koh Young baseline |
|---|---:|---:|
| Total score | **13,351.1** | 14,117.0 |
| ΣCT / ΣRT / Σ10k | 7,668.8 / 42.4 / 5,640 | 8,539.2 / 147.8 / 5,430 |
| Avg FOVs | 49.9 | 61.1 |
| Avg RT per board | 0.44 s | 1.54 s |

3rd of 13 teams. All 96 boards passed with no rule violations.
Official values. Each sum is rounded separately, so the parts add up to 13,351.2.

## Repository layout

```
├── ARGOS/                  # solver (as submitted)
│   ├── solver.py           #   entry point solve(), overall flow
│   ├── geometry.py         #   input reading, valid boxes, tile FOV positions for big components
│   ├── cover.py            #   weighted greedy, redundant removal, elimination, recenter
│   ├── setcover.py         #   RWLS
│   ├── setcover_py.py      #   setcover.py without numba
│   ├── order.py            #   blocks, paths inside blocks, 7 candidate orders
│   ├── refine.py           #   center refinement (shift FOV centers to cut move time)
│   ├── assign.py           #   heavy reassign
│   ├── anneal.py           #   simulated annealing, polish
│   ├── polish_nb.py        #   polish with numba
│   ├── release.py          #   release
│   ├── release_tools.py    #   inspection timing used by release.py
│   ├── recover.py          #   re-split the components of two nearby FOVs
│   ├── ct_sim.py           #   CT simulator (same computation as the checker)
│   ├── acceleration.py     #   ct_sim.py with numba
│   ├── verify.py           #   contest rule check and repair
│   ├── parallel.py         #   runs the 5 covers in separate processes
│   └── atomic_io.py        #   output writing (no broken file if interrupted)
├── contest/                # official contest material (unchanged)
│   ├── 1_generate_data.py  #   data generator
│   ├── 2_run.py            #   runner and scoring (--eval: re-score saved outputs)
│   ├── 3_view.py           #   viewer (FOV layout and core timeline)
│   ├── utils/              #   checker.py
│   ├── simulation_data/    #   practice data, 96 boards
│   ├── evaluation_data/    #   evaluation data, 96 boards, with ARGOS and baseline outputs per board
│   ├── evaluation_results/ #   official evaluation results
│   ├── ky_solver/          #   Koh Young baseline solver (compiled)
│   ├── greedy_solver/      #   official greedy example
│   ├── my_solver/          #   official solver template
│   ├── *.html              #   contest rules, data description, guide, tutorial, Q&A
│   ├── SOURCE.md           #   where each file in this folder comes from
│   └── …                   #   other official files
├── README.md               # this file
├── README.ko.md            # Korean README
├── requirements.txt        # package versions
├── LICENSE                 # CC BY-NC-SA 4.0
└── NOTICE.md               # contest material sources, parts re-implemented from the checker
```

## Run

Python 3.12, package versions in `requirements.txt`.

```bash
pip install -r requirements.txt
cp -r ARGOS contest/
cd contest
python 2_run.py --solver ARGOS --data evaluation_data --eval     # re-score the contest outputs only
```

Solving again overwrites the contest outputs in `evaluation_data/<board>/ARGOS/`. Re-score them first if you want to keep them.

```bash
python 2_run.py --solver ARGOS --data evaluation_data            # solve and score
```

## License

[CC BY-NC-SA 4.0](LICENSE). Give credit, no commercial use, and share changes under the same license.

- `contest/` is the original contest material provided by Koh Young Technology. Sources are in [contest/SOURCE.md](contest/SOURCE.md).
- `ct_sim.py`, `acceleration.py`, `release_tools.py`, `geometry.py` and `verify.py` independently re-implement computations of the contest checker, so the whole repository uses the same license. Details are in [NOTICE.md](NOTICE.md).
- This is not official code of Koh Young Technology or of the contest host, ICROS (Institute of Control, Robotics and Systems).
