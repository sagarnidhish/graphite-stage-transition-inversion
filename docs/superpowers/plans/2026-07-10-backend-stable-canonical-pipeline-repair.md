# Backend-Stable Canonical Pipeline Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every production change and superpowers:verification-before-completion before each commit.

**Goal:** Repair the boundary-driven graphite CHR benchmark, reduce differentiable execution cost without changing its recurrence, and make its scientific objective and canonical execution reproducible enough for a new simulation-only cohort.

**Architecture:** Keep the existing CHR solver, configuration, and benchmark boundaries. Correct geometry at the source, make saved-output collection an internal solver concern, add a separate differentiable observable layer consumed by inversion and identifiability, and put deterministic process/provenance handling around the existing benchmark task API.

**Tech Stack:** Python 3.12, JAX/JAXLIB 0.10.2 in float64, NumPy, SciPy, pytest, TOML, multiprocessing with spawn.

## Global Constraints

- Add a failing test before each production change and observe the intended failure.
- Do not preserve the old cohort, timestep, or recovery results as positive evidence.
- Do not change the per-step `semi_implicit_step` recurrence in the sparse-output task.
- Validation and locked-test inversions remain disabled until all development gates pass.
- Keep all random seeds derived from stable task identity, never list position.
- Commit each completed task independently and update `STATUS.md` when scientific status changes.

## Task 1: Correct Boundary Face Counting

**Files:**
- Modify: `src/graphite_stage_transition/geometry.py`
- Test: `tests/test_geometry.py`
- Test: `tests/test_reaction.py`

- [ ] Add failing tests for exact 3 x 3 exposed-face counts and canonical-circle boundary facts.
- [ ] Add a failing reaction test proving exchange weight and reaction rate are zero on interior cells.
- [ ] Run `PYTHONNOUSERSITE=1 .conda/bin/pytest -q tests/test_geometry.py tests/test_reaction.py` and confirm the new tests fail for the volumetric-source defect.
- [ ] Cast padded Boolean neighbor masks to an integer dtype before summing.
- [ ] Rerun the focused tests and the full suite.
- [ ] Commit as `fix: localize reaction to particle boundary`.

## Task 2: Implement Sparse Saved-Output Collection

**Files:**
- Modify: `src/graphite_stage_transition/solver.py`
- Modify: `src/graphite_stage_transition/protocols.py`
- Test: `tests/test_solver.py`
- Test: `tests/test_protocols.py`
- Test: `tests/test_inversion.py`

- [ ] Retain a test-only dense reference and add failing tests for exact saved-state/final-state equality.
- [ ] Add irregular-save alignment, diagnostics-tolerance, and tiny objective/gradient equivalence tests.
- [ ] Validate save indices as int32, unique, strictly increasing, starting at zero and ending at the step count.
- [ ] Replace dense scan output with `ys=None` and fixed-size save buffers in the scan carry.
- [ ] Store only on requested steps using a precomputed step-to-slot map and `jax.lax.cond`.
- [ ] Run focused solver/protocol/inversion tests, then the full suite.
- [ ] Commit as `perf: store only requested CHR states`.

## Task 3: Add Physics-Facing Observables

**Files:**
- Create: `src/graphite_stage_transition/observables.py`
- Modify: `src/graphite_stage_transition/__init__.py`
- Modify: `src/graphite_stage_transition/inversion.py`
- Modify: `src/graphite_stage_transition/identifiability.py`
- Create: `tests/test_observables.py`
- Modify: `tests/test_inversion.py`
- Modify: `tests/test_identifiability.py`

- [ ] Add failing tests for uniform fields, symmetry invariance, Parseval scaling, and equal-mean core/shell discrimination.
- [ ] Implement fixed equal-area radial weights, radial spectral-band masks, and corrected boundary weights.
- [ ] Implement radial profile, structure power, boundary excess, and weighted residual-vector APIs.
- [ ] Add failing tests for identical-movie zero loss and centered finite-difference agreement.
- [ ] Replace primary pixel/mass loss with frozen `0.50/0.35/0.15` observable weights plus the existing bounds penalty.
- [ ] Keep pixel MSE and mass mismatch as named zero-weight diagnostics.
- [ ] Route identifiability through the same residual vector used by inversion.
- [ ] Run focused tests, then the full suite.
- [ ] Commit as `feat: fit morphology observables`.

## Task 4: Add Deterministic Canonical CPU Execution

**Files:**
- Create: `.python-version`
- Create: `requirements/canonical-cpu.txt`
- Create: `src/graphite_stage_transition/execution.py`
- Modify: `src/graphite_stage_transition/benchmark.py`
- Modify: `scripts/run_benchmark.py`
- Modify: `scripts/build_kgpu_payload.py`
- Create: `tests/test_execution.py`
- Modify: `tests/test_benchmark.py`

- [ ] Add failing tests for task-ID seed stability, plan fingerprint sensitivity, marker validation, and deterministic resume order.
- [ ] Pin direct canonical dependency versions and record the Python version.
- [ ] Implement SHA-256 task seeds and hashes for source, manifest, config, dependencies, observable schema, optimizer, and seed policy.
- [ ] Validate result markers against execution fingerprint, seed, optimizer controls, and result checksum.
- [ ] Add a spawn-based CPU runner with two workers, six assigned cores per worker, pre-import affinity, and single-threaded BLAS defaults.
- [ ] Make public/private payload seed policy use the same task-ID derivation.
- [ ] Run focused tests, the full suite, and a deterministic two-task dry run.
- [ ] Commit as `feat: add canonical CPU benchmark runner`.

## Task 5: Gate Backend Use And Scientific Claims

**Files:**
- Create: `src/graphite_stage_transition/backend_gate.py`
- Create: `scripts/run_backend_probe.py`
- Create: `scripts/compare_backend_probes.py`
- Create: `configs/backend_gate.toml`
- Create: `tests/test_backend_gate.py`
- Modify: `scripts/run_benchmark.py`

- [ ] Add failing tests for observable RMS/objective/gradient thresholds and gate fingerprint matching.
- [ ] Implement probe serialization and comparison for observable blocks, primary objective, and gradients.
- [ ] Require a matching passed gate for claim-eligible CHR multistart runs.
- [ ] Allow development-only diagnostic execution with `claim_eligible=false`; reject validation/test splits.
- [ ] Run focused tests and the full suite.
- [ ] Commit as `feat: gate claim-eligible benchmark runs`.

## Task 6: Revalidate And Regenerate

**Files:**
- Modify: `configs/canonical.toml`
- Modify: `STATUS.md`
- Modify: `docs/methods_report.md`
- Modify as needed: `scripts/verify_forward.py`
- Generate under: `outputs/`

- [ ] Run staged forward gates on the repaired boundary model: signs, current, mass, relaxation, replay, timestep, grid refinement, and full transition.
- [ ] Freeze a newly validated timestep only if all forward gates pass.
- [ ] Run sparse/dense equivalence and tiny observable-gradient gates in the pinned CPU environment.
- [ ] Run a bounded two-case CPU/free-GPU observable backend probe.
- [ ] If and only if the backend gate passes, regenerate the 16-case one-replicate development cohort with a new revision and hashes.
- [ ] Run tiny recovery and bounded development-only inversion; keep validation/test locked.
- [ ] Update `STATUS.md` and methods documentation with commands, hashes, timings, pass/fail evidence, and guarded claims.
- [ ] Run the complete test suite from a clean process and review `git diff --check`.
- [ ] Publish the repaired source only after a final code review and verification pass.

