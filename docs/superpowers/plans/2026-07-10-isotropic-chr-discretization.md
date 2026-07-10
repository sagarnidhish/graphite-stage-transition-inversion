# Isotropic CHR Discretization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the grid-biased graphite CHR operator and boundary reaction weights with conservative isotropy-controlled equivalents, then reauthorize all downstream simulation evidence.

**Architecture:** Keep the existing masked JAX solver and public interfaces. Replace its four-edge graph Laplacian with an eight-edge weighted graph Laplacian, and replace exposed-face boundary length with analytic-circle angular Voronoi weights. Validate numerical invariants before running claim-bearing simulations.

**Tech Stack:** Python 3.12.13, JAX 0.10.2, NumPy 2.5.1, pytest 9.0.3, Matplotlib 3.11.0.

## Global Constraints

- Preserve pairwise conservation and Euclidean symmetry required by conjugate gradient.
- Use the repaired operator consistently everywhere `masked_laplacian` is called.
- Keep fixed color limits and unsmoothed fields in scientific figures.
- Do not reuse old cohort, inversion, or backend-gate claims after source changes.

---

### Task 1: Isotropic Operator Contract

**Files:**
- Modify: `tests/test_operators.py`
- Modify: `src/graphite_stage_transition/operators.py`

**Interfaces:**
- Consumes: `masked_laplacian(field: jax.Array, grid: Grid) -> jax.Array`
- Produces: the same signature with conservative axial and diagonal fluxes

- [ ] Add tests for the full-grid nine-point impulse stencil and improved angular Fourier-symbol error.
- [ ] Run the new tests and confirm they fail against the four-neighbour operator.
- [ ] Implement pairwise axial weights `2/3` and diagonal weights `1/6`.
- [ ] Run operator, solver, and gradient tests and confirm they pass.
- [ ] Commit the operator change.

### Task 2: Analytic Circular Boundary Weights

**Files:**
- Modify: `tests/test_geometry.py`
- Modify: `src/graphite_stage_transition/geometry.py`

**Interfaces:**
- Consumes: `make_circle_grid(config: GridConfig) -> Grid`
- Produces: `Grid.boundary_weight` as analytic arc length per reactive cell

- [ ] Add tests for exact circumference, interior zero, nonnegativity, and 90-degree equivariance.
- [ ] Run the new tests and confirm exposed-face weights fail the circumference test.
- [ ] Implement polar-angle Voronoi arc assignment on the axial boundary-cell set.
- [ ] Run geometry, reaction, observable, mass, and solver tests.
- [ ] Commit the boundary change.

### Task 3: Controlled Isotropy and Rotation Gates

**Files:**
- Modify: `tests/test_verification.py`
- Modify: `src/graphite_stage_transition/verification.py`
- Modify: `scripts/verify_isotropy.py`

**Interfaces:**
- Produces: controlled radial-transition and rotation-equivariance reports

- [ ] Add a failing controlled radial-transition test and a 90-degree trajectory-equivariance test.
- [ ] Add report dataclasses and validation functions without changing existing gate semantics.
- [ ] Run focused verification tests, then the full suite.
- [ ] Commit the new gates.

### Task 4: Forward and Refinement Validation

**Files:**
- Modify: `configs/transition.toml` only if stability evidence requires a smaller `dt`
- Generate: `outputs/transition_forward_isotropic_v1/`
- Modify: `STATUS.md`

**Interfaces:**
- Consumes: repaired solver and existing transition protocol
- Produces: trajectory, mass/full-cycle/refinement/isotropy JSON, montage, kymograph, diagnostics, movie

- [ ] Run the controlled radial transition at 48, 64, and 96 pixels.
- [ ] Require isotropy, conservation, relaxation, full-cycle, and refinement gates to pass.
- [ ] Run the seeded morphology transition and render fixed-scale diagnostics.
- [ ] Record failures honestly and stop downstream generation if any mandatory gate fails.
- [ ] Commit configuration or documentation changes.

### Task 4A: Radial Finite-Volume Reference Backend

**Files:**
- Create: `src/graphite_stage_transition/radial.py`
- Create: `tests/test_radial.py`
- Create: `scripts/run_radial_forward.py`

**Interfaces:**
- Produces: conservative radial geometry, volume-weighted Laplacian, transformed
  symmetric CG solve, matching free energy, and differentiable Cartesian raster

- [ ] Add failing geometry, conservation, self-adjointness, and energy-gradient tests.
- [ ] Implement radial finite-volume geometry and operators.
- [ ] Add failing equilibrium and mass/current integration tests.
- [ ] Implement the stabilized volume-weighted CHR step and simulation scan.
- [ ] Add failing rasterization and sampled-circle baseline tests.
- [ ] Implement fixed radial-to-Cartesian interpolation.
- [ ] Run focused and full tests and commit.

### Task 4B: Radial Full-Cycle and Resolution Gate

**Files:**
- Create: `configs/radial_transition.toml`
- Generate: `outputs/radial_transition_isotropic_v1/`
- Modify: `STATUS.md`

- [ ] Choose radial resolution from the analytical 10-90 interface width,
  requiring at least four cells.
- [ ] Run the stage-2 to stage-1 to stage-2 full cycle.
- [ ] Require exact mass/current balance, finite bounds, relaxation, and radial
  refinement against a finer reference.
- [ ] Rasterize and render the 2D movie with fixed concentration limits.
- [ ] Calibrate the Cartesian angular diagnostic against exact sampled radial
  fronts rather than applying an uncorrected threshold.

### Task 5: Backend Reauthorization and Cohort Decision

**Files:**
- Generate: new CPU/GPU probe artifacts under `outputs/`
- Modify: `STATUS.md`
- Modify: `docs/methods_report.md`

**Interfaces:**
- Consumes: new source fingerprint and passing local solver gates
- Produces: fingerprint-bound CPU/GPU authorization for the repaired operator

- [ ] Generate the new canonical execution fingerprint and CPU probe.
- [ ] Publish the exact source commit used by the GPU bootstrap.
- [ ] Run the GPU probe and compare observable, objective, and gradient evidence.
- [ ] Rebuild the synthetic cohort only if the backend gate passes.
- [ ] Run the full test suite and record final claim boundaries.
- [ ] Commit final status and methods updates.
