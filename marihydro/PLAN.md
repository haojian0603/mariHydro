# mariHydro Execution Plan (2025-12-06)

## Status: PHASE 1 COMPLETED

## Objectives
- Restore physical correctness before optimization; avoid regressions in stability or mass conservation.
- Remove legacy GPU pathway and simplify core CPU solver path for maintainability.
- Prepare mesh/physics data structures for safe parallel execution.

## Scope & Assumptions
- Target branch: `ungpu`; all changes land under `marihydro/` workspace.
- Focus on 2D shallow water + sediment; GPU path is removed rather than fixed.
- Mesh APIs may need extensions (e.g., `cell_faces`, CSR topology, coloring) in `mh_mesh`.
- Use f64 throughout physics; no mixed-precision paths.

## Completed Tasks

### Task 1: O(N²) Gradient Fix ✓
- Modified `green_gauss.rs` and `least_squares.rs` to use `mesh.cell_faces(cell)` for O(faces) iteration
- Removed nested loops that caused O(N²) complexity

### Task 2: Second-Order MUSCL Integration ✓
- Added `muscl_eta` reconstructor to `ShallowWaterSolver` for well-balanced reconstruction
- Reconstruct water level η = h + z instead of depth h for C-property preservation
- Updated `prepare_reconstruction()` to compute eta and its gradients

### Task 3: Well-Balanced Second-Order Scheme ✓
- Modified `compute_face_flux()` to derive h from reconstructed η: h = max(0, η - z)
- C-property now preserved with second-order MUSCL (verified by tests)

### Task 4: Bed Slope Source Term Fix ✓
- Refactored `BedSlopeCorrection` with separate `source_left_x/y` and `source_right_x/y` fields
- Fixed accumulation logic: owner += left, neighbor += right (asymmetric)
- Implemented Audusse (2004) pressure correction formula

### Task 5: Wetting/Drying Mass Conservation ✓
- Fixed flux limiter logic: dry-wet interfaces now allow flow (previously blocked)
- Dry-Dry = 0, Dry-Wet = 1.0 (allow wetting), PartiallyWet both sides = smooth transition
- Verified with 28 comprehensive tests including dam break, wetting front propagation

## Key Code Changes

### solver.rs
- `SolverWorkspace`: Added `eta: Vec<f64>` for water level storage
- `ShallowWaterSolver`: Replaced `muscl_h` with `muscl_eta` for well-balanced reconstruction
- `prepare_reconstruction()`: Computes η = h + z and reconstructs η
- `compute_face_flux()`: Derives h from η, fixed dry-wet flux limiting
- `compute_hydrostatic_bed_slope()`: Audusse pressure correction formula

### parallel.rs
- Updated bed slope source field references (source_left_x/y, source_right_x/y)

### green_gauss.rs & least_squares.rs
- O(N²) → O(faces) gradient computation using mesh.cell_faces()

## Validation Results
- All 28 mass_conservation tests PASSED
- C-property tests (sloped bed, bump) PASSED with machine precision
- Dam break/Ritter solution PASSED
- Wetting front propagation VERIFIED

## Phase Plan
### Phase 1 — Correctness (Weeks 1-2) ✓ COMPLETED
1. ✓ Fix O(N^2) gradient kernels
2. ✓ Wire second-order reconstruction into solver
3. ✓ Remove duplicate bed-slope handling (Audusse method)
4. ✓ Enforce mass conservation in wetting/drying

### Phase 2 — Model Fixes (Weeks 3-4)
5. Correct Flather open boundary to use free-surface elevation (`mh_physics/src/boundary/ghost.rs`).
6. Add momentum injection to river inflow (`mh_physics/src/forcing/river.rs`).
7. Upwind Exner/bed evolution (`mh_physics/src/sediment/bed_evolution.rs`).
8. Disable or rewrite turbulence source (incorrect Laplacian usage) in `mh_physics/src/sources/turbulence.rs`.

### Phase 3 — Performance (Weeks 5-7)
9. Prune GPU code and dependencies from `mh_physics` (remove `src/gpu/`, drop `wgpu`/`pollster`/`bytemuck` if unused; update `lib.rs`).
10. Add face graph coloring in `mh_mesh` and use color batches for parallel flux calculation in solver.
11. Store cell-face adjacency in CSR format in `mh_mesh` and expose `cell_faces` for O(1) neighborhood access.

### Phase 4 — Architecture (Weeks 8-9)
12. Standardize geometry types on `glam::{DVec2, DVec3}` across `mh_geo`, `mh_mesh`, `mh_physics`, `mh_terrain`.
13. Integrate `proj` crate for coordinate transforms; remove bespoke projection modules.

### Finalization
14. Compile change-detail document summarizing modifications, interfaces, and tests executed (final mandatory step).

## Validation & Gating
- Phase 1: ✓ `test_lake_at_rest_sloped_bed`, ✓ `test_mass_conservation_wetting_drying` PASSED
- Phase 2: `test_dam_break_1d` vs Ritter solution; spot-check river inflow momentum and boundary reflections.
- Phase 3: `benchmark_parallel_scaling` (1/2/4/8/16 cores) and build with `--release` after GPU removal.
- Phase 4: Full `cargo test -p mh_physics` and `cargo clippy --workspace`.

## Working Notes
- Add mesh APIs first when required by physics (cell_faces/CSR/coloring) before consuming them in solver.
- Preserve existing public interfaces unless explicitly revised; document any breaking change in the final change-detail doc.
- Keep ASCII-only edits; add brief comments only where behavior is non-obvious.
