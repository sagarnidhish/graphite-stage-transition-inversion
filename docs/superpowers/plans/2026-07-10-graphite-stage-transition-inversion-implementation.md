# Graphite Stage Transition Inversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. In this thread, use inline `superpowers:executing-plans`; subagents require an explicit user request.

**Goal:** Build and verify a differentiable 2D Cahn-Hilliard-reaction benchmark that generates graphite stage 2 <-> stage 1 concentration movies, recovers identifiable parameter groups, quantifies non-identifiability, compares baselines, and produces reproducible scientific outputs.

**Architecture:** Use a masked cell-centered finite-volume discretization on idealized circular particles. Advance the stiff Cahn-Hilliard operator with a semi-implicit matrix-free conjugate-gradient solve inside `jax.lax.scan`, and impose a differentiable galvanostatic Butler-Volmer boundary source whose summed reaction exactly matches applied current. Keep simulation, observation corruption, inversion, diagnostics, baselines, and reporting as separate modules with typed data contracts.

**Tech Stack:** Python 3.12, NumPy, SciPy, JAX, Matplotlib, ImageIO, pytest, TOML configuration, NPZ/JSON/CSV artifacts, `kgpu` Tesla P100 execution.

## Global Constraints

- Keep all first-phase claims simulation-only and nondimensional.
- Stage 2 is `c = 0.5`; stage 1 is `c = 1.0`.
- The direct concentration field is the only first-phase observation; no iSCAT renderer is included.
- Comparable concentration frames use fixed limits `[0.5, 1.0]`.
- Positive inverse parameters are represented in log space.
- Score identifiable dimensionless groups before raw `(M, A, kappa, k0)` parameters.
- Parameter cases, not frames, define development, validation, and locked test splits.
- Do not run the benchmark ensemble until conservation, relaxation, convergence, gradient, determinism, and tiny-recovery gates pass.
- Preserve every random seed, configuration, command, code revision, and failure status.
- Use `kgpu` for full differentiable benchmark execution after local CPU smoke tests pass.

---

## File Map

- `pyproject.toml`: package metadata and dependency floors.
- `.gitignore`: excludes environments, caches, large generated artifacts, and videos.
- `configs/canonical.toml`: canonical forward and inverse smoke configuration.
- `src/graphite_stage_transition/config.py`: validated configuration dataclasses and TOML loader.
- `src/graphite_stage_transition/geometry.py`: circular masks, boundary weights, coordinates, and radial bins.
- `src/graphite_stage_transition/operators.py`: masked finite-volume gradient/Laplacian primitives.
- `src/graphite_stage_transition/free_energy.py`: two-well free energy and chemical terms.
- `src/graphite_stage_transition/reaction.py`: differentiable galvanostatic reaction law.
- `src/graphite_stage_transition/protocols.py`: piecewise-constant current protocol definitions.
- `src/graphite_stage_transition/solver.py`: semi-implicit CHR stepping and trajectory diagnostics.
- `src/graphite_stage_transition/verification.py`: mass, energy, convergence, and determinism gates.
- `src/graphite_stage_transition/visualization.py`: approved concentration movie, montage, and kymograph.
- `src/graphite_stage_transition/synthetic.py`: Latin-hypercube cases, corruption, split, and manifests.
- `src/graphite_stage_transition/inversion.py`: transformed parameters, differentiable loss, and multistart fitting.
- `src/graphite_stage_transition/baselines.py`: random search, mean-only, Fickian, and sharp-interface controls.
- `src/graphite_stage_transition/identifiability.py`: Hessian/Fisher, SVD, correlations, and profile likelihoods.
- `src/graphite_stage_transition/benchmark.py`: resumable benchmark and ablation orchestration.
- `src/graphite_stage_transition/gpu_payload.py`: self-contained Kaggle payload construction and result retrieval.
- `src/graphite_stage_transition/reporting.py`: aggregate tables, figures, and methods summary.
- `scripts/*.py`: thin command-line entry points for verified workflows.
- `tests/*.py`: focused unit, integration, gradient, and recovery tests.

---

### Task 1: Package Scaffold And Validated Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `configs/canonical.toml`
- Create: `src/graphite_stage_transition/__init__.py`
- Create: `src/graphite_stage_transition/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `GridConfig`, `ProtocolConfig`, `SolverConfig`, `ModelConfig`, `InversionConfig`, `ProjectConfig`, and `load_config(path: Path) -> ProjectConfig`.

- [ ] **Step 1: Write failing configuration tests**

```python
from pathlib import Path
import pytest
from graphite_stage_transition.config import load_config


def test_canonical_config_loads():
    cfg = load_config(Path("configs/canonical.toml"))
    assert cfg.grid.nx == 48
    assert cfg.model.stage2 == 0.5
    assert cfg.model.stage1 == 1.0
    assert cfg.protocol.currents == (0.02, 0.0, -0.02, 0.0)


def test_invalid_stage_order_fails(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[model]\nstage2=1.0\nstage1=0.5\n")
    with pytest.raises(ValueError, match="stage2 < stage1"):
        load_config(path)
```

- [ ] **Step 2: Run the tests and verify import/config failures**

Run: `.venv/bin/pytest tests/test_config.py -v`

Expected: FAIL because the package and configuration loader do not exist.

- [ ] **Step 3: Add package metadata and canonical TOML**

Use these dependency floors in `pyproject.toml`:

```toml
[project]
name = "graphite-stage-transition"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "jax>=0.4.38",
  "numpy>=2.0",
  "scipy>=1.13",
  "matplotlib>=3.8",
  "imageio>=2.34",
  "imageio-ffmpeg>=0.5",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Set the canonical smoke grid to `48 x 48`, domain length `1.0`, radius `0.4`,
four protocol segments `(0.02, 0.0, -0.02, 0.0)`, 32 saved frames per
segment, semi-implicit timestep `2e-3`, and deterministic seed `7`.

- [ ] **Step 4: Implement frozen validated dataclasses**

```python
@dataclass(frozen=True)
class ModelConfig:
    mobility: float
    barrier: float
    kappa: float
    reaction_rate: float
    stage2: float = 0.5
    stage1: float = 1.0

    def __post_init__(self):
        if not self.stage2 < self.stage1:
            raise ValueError("stage2 < stage1 is required")
        for name in ("mobility", "barrier", "kappa", "reaction_rate"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
```

Parse TOML with `tomllib`; reject missing required sections and nonpositive grid,
solver, or physical values.

- [ ] **Step 5: Create the environment and run configuration tests**

Run: `python3 -m venv .venv`

Run: `.venv/bin/pip install -e '.[dev]'`

Run: `.venv/bin/pytest tests/test_config.py -v`

Expected: 2 passed.

- [ ] **Step 6: Commit scaffold**

```bash
git add pyproject.toml .gitignore README.md configs src tests/test_config.py
git commit -m "build: scaffold graphite inversion package"
```

---

### Task 2: Geometry And Conservative Masked Operators

**Files:**
- Create: `src/graphite_stage_transition/geometry.py`
- Create: `src/graphite_stage_transition/operators.py`
- Test: `tests/test_geometry.py`
- Test: `tests/test_operators.py`

**Interfaces:**
- Produces: `Grid`, `make_circle_grid(config) -> Grid`, `masked_laplacian(field, grid)`, `boundary_cell_weights(grid)`, and `radial_bin_indices(grid, bins)`.
- `Grid` contains `x`, `y`, `mask`, `boundary_weight`, `dx`, `cell_area`, `radius`, and `active_count`.

- [ ] **Step 1: Write geometry tests**

```python
def test_circle_grid_is_centered_and_has_boundary():
    grid = make_circle_grid(GridConfig(nx=64, ny=64, length=1.0, radius=0.4))
    assert grid.mask.shape == (64, 64)
    assert grid.boundary_weight.shape == grid.mask.shape
    assert int(grid.mask.sum()) > 0
    assert float(grid.boundary_weight.sum()) > 0.0
    assert abs(float((grid.x * grid.mask).sum())) < 1e-12
```

- [ ] **Step 2: Write conservative-operator tests**

```python
def test_masked_laplacian_annihilates_constant():
    grid = make_circle_grid(GridConfig(nx=48, ny=48, length=1.0, radius=0.4))
    field = jnp.where(grid.mask, 0.73, 0.0)
    lap = masked_laplacian(field, grid)
    np.testing.assert_allclose(np.asarray(lap[grid.mask]), 0.0, atol=1e-12)


def test_masked_laplacian_sums_to_zero():
    grid = make_circle_grid(GridConfig(nx=48, ny=48, length=1.0, radius=0.4))
    key = jax.random.key(2)
    field = jax.random.normal(key, (48, 48))
    lap = masked_laplacian(field, grid)
    assert abs(float(lap.sum())) < 1e-10
```

- [ ] **Step 3: Run tests and confirm missing-interface failures**

Run: `.venv/bin/pytest tests/test_geometry.py tests/test_operators.py -v`

Expected: FAIL because geometry and operators are undefined.

- [ ] **Step 4: Implement face-pair finite-volume Laplacian**

Compute flux differences only across pairs where both adjacent cells are active:

```python
def masked_laplacian(field, grid):
    u = jnp.where(grid.mask, field, 0.0)
    fx = (u[1:, :] - u[:-1, :]) / grid.dx
    fy = (u[:, 1:] - u[:, :-1]) / grid.dx
    fx = fx * (grid.mask[1:, :] & grid.mask[:-1, :])
    fy = fy * (grid.mask[:, 1:] & grid.mask[:, :-1])
    out = jnp.zeros_like(u)
    out = out.at[:-1, :].add(fx / grid.dx)
    out = out.at[1:, :].add(-fx / grid.dx)
    out = out.at[:, :-1].add(fy / grid.dx)
    out = out.at[:, 1:].add(-fy / grid.dx)
    return jnp.where(grid.mask, out, 0.0)
```

Count missing active-to-inactive cardinal faces per active cell for
`boundary_weight`; normalize only when distributing a requested total current.

- [ ] **Step 5: Run operator tests**

Run: `.venv/bin/pytest tests/test_geometry.py tests/test_operators.py -v`

Expected: all pass; constant and global-sum errors below specified tolerances.

- [ ] **Step 6: Commit geometry and operators**

```bash
git add src/graphite_stage_transition/geometry.py src/graphite_stage_transition/operators.py tests/test_geometry.py tests/test_operators.py
git commit -m "feat: add conservative masked grid operators"
```

---

### Task 3: Effective Free Energy And Galvanostatic Reaction

**Files:**
- Create: `src/graphite_stage_transition/free_energy.py`
- Create: `src/graphite_stage_transition/reaction.py`
- Test: `tests/test_free_energy.py`
- Test: `tests/test_reaction.py`

**Interfaces:**
- Produces: `homogeneous_free_energy(c, barrier, c2, c1)`, `homogeneous_mu(...)`, `total_free_energy(c, grid, params)`, and `galvanostatic_reaction(c_boundary, mu_boundary, weights, target_current, k0) -> ReactionState`.
- `ReactionState` contains `rate`, `overpotential`, `summed_current`, and `exchange_weight`.

- [ ] **Step 1: Test free-energy minima and derivative**

```python
def test_double_well_has_stage_minima():
    c = jnp.array([0.5, 0.75, 1.0])
    f = homogeneous_free_energy(c, barrier=2.0, c2=0.5, c1=1.0)
    mu = homogeneous_mu(c, barrier=2.0, c2=0.5, c1=1.0)
    assert f[1] > f[0]
    assert f[1] > f[2]
    np.testing.assert_allclose(np.asarray(mu[[0, 2]]), 0.0, atol=1e-12)
```

- [ ] **Step 2: Test exact current matching and reversal**

```python
@pytest.mark.parametrize("target", [0.03, 0.0, -0.03])
def test_galvanostatic_reaction_matches_target(target):
    c = jnp.array([0.55, 0.70, 0.90])
    mu = jnp.array([-0.2, 0.0, 0.15])
    weights = jnp.array([1.0, 2.0, 1.0])
    state = galvanostatic_reaction(c, mu, weights, target, k0=0.4)
    np.testing.assert_allclose(float(state.summed_current), target, rtol=1e-7, atol=1e-9)
    assert np.sign(float(state.rate.sum())) == np.sign(target)
```

- [ ] **Step 3: Run focused tests and verify failure**

Run: `.venv/bin/pytest tests/test_free_energy.py tests/test_reaction.py -v`

Expected: FAIL because functions do not exist.

- [ ] **Step 4: Implement scaled quartic free energy**

```python
def homogeneous_free_energy(c, barrier, c2=0.5, c1=1.0):
    width = c1 - c2
    q = (c - c2) / width
    return 16.0 * barrier * q**2 * (1.0 - q) ** 2
```

Implement `homogeneous_mu` analytically and verify it independently with
`jax.grad` in the test.

- [ ] **Step 5: Implement analytic galvanostatic overpotential**

For `rate_i = a_i sinh((eta - mu_i)/2)`, define
`P=sum(a_i exp(-mu_i/2))`, `Q=sum(a_i exp(mu_i/2))`, and solve
`P*y^2 - 2*I*y - Q = 0` for positive `y=exp(eta/2)`. Use
`eta=2*log((I + sqrt(I^2 + P*Q))/P)` and evaluate the rates. Normalize the
discrete rate by boundary measure so `sum(rate_i * weight_i) == I` to floating
precision.

- [ ] **Step 6: Run tests and commit**

Run: `.venv/bin/pytest tests/test_free_energy.py tests/test_reaction.py -v`

Expected: all pass, including autodiff derivative comparison.

```bash
git add src/graphite_stage_transition/free_energy.py src/graphite_stage_transition/reaction.py tests/test_free_energy.py tests/test_reaction.py
git commit -m "feat: add phase thermodynamics and boundary reaction"
```

---

### Task 4: Protocols And Semi-Implicit Differentiable CHR Solver

**Files:**
- Create: `src/graphite_stage_transition/protocols.py`
- Create: `src/graphite_stage_transition/solver.py`
- Test: `tests/test_protocols.py`
- Test: `tests/test_solver.py`

**Interfaces:**
- Produces: `Protocol`, `build_protocol(config)`, `CHRParameters`, `SimulationResult`, `semi_implicit_step`, and `simulate(grid, protocol, params, solver, seed) -> SimulationResult`.
- `SimulationResult` contains `concentration`, `times`, `currents`, `mass`, `free_energy`, `overpotential`, `cg_residual`, and `metadata`.

- [ ] **Step 1: Test exact protocol segment construction**

```python
def test_protocol_has_requested_current_and_rest_segments():
    canonical = load_config(Path("configs/canonical.toml"))
    p = build_protocol(canonical.protocol, canonical.solver.dt)
    assert p.current[0] > 0.0
    assert np.any(np.asarray(p.current) == 0.0)
    assert p.current[-1] == 0.0
    assert p.save_indices[0] == 0
    assert p.save_indices[-1] == len(p.current)
```

- [ ] **Step 2: Write zero-current and current-direction solver tests**

```python
def test_uniform_equilibrium_is_stationary():
    grid, equilibrium_params, solver = make_small_test_system(initial=0.5)
    zero_protocol = make_constant_protocol(current=0.0, steps=8, dt=solver.dt)
    result = simulate(grid, zero_protocol, equilibrium_params, solver, seed=0)
    np.testing.assert_allclose(result.concentration[-1], result.concentration[0], atol=2e-6)


def test_current_changes_mass_with_correct_sign():
    grid, params, solver = make_small_test_system(initial=0.75)
    positive_protocol = make_constant_protocol(current=0.01, steps=8, dt=solver.dt)
    negative_protocol = make_constant_protocol(current=-0.01, steps=8, dt=solver.dt)
    lith = simulate(grid, positive_protocol, params, solver, seed=1)
    de = simulate(grid, negative_protocol, params, solver, seed=1)
    assert lith.mass[-1] > lith.mass[0]
    assert de.mass[-1] < de.mass[0]
```

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `.venv/bin/pytest tests/test_protocols.py tests/test_solver.py -v`

Expected: FAIL because protocol and solver interfaces are absent.

- [ ] **Step 4: Implement matrix-free semi-implicit step**

For fixed mobility, split the fourth-order term:

```text
(I + dt * M * kappa * L^2) c_next
    = c + dt * M * L(mu_hom(c)) + dt * boundary_source.
```

Use `jax.scipy.sparse.linalg.cg` with a matrix-free function applying
`v + dt*M*kappa*L(L(v))`. Keep inactive cells zero and initialize CG from the
previous concentration. Return the explicit residual norm for diagnostics.

- [ ] **Step 5: Implement checkpointed scan and saved-frame extraction**

Use `jax.lax.scan(jax.checkpoint(step), ...)` over fixed protocol steps. Store
only requested save indices in the public result. Apply seeded perturbations
inside the mask, remove their masked mean, and clip only the initial state; do
not clip time steps because clipping would hide conservation errors.

- [ ] **Step 6: Run solver tests and a canonical smoke**

Run: `.venv/bin/pytest tests/test_protocols.py tests/test_solver.py -v`

Run: `.venv/bin/python scripts/run_forward.py --config configs/canonical.toml --out outputs/smoke_forward`

Expected: tests pass; NPZ and JSON summary are created; all fields are finite.

- [ ] **Step 7: Commit solver**

```bash
git add src/graphite_stage_transition/protocols.py src/graphite_stage_transition/solver.py scripts/run_forward.py tests/test_protocols.py tests/test_solver.py
git commit -m "feat: add differentiable semi-implicit CHR solver"
```

---

### Task 5: Forward Verification Gates

**Files:**
- Create: `src/graphite_stage_transition/verification.py`
- Create: `scripts/verify_forward.py`
- Test: `tests/test_verification.py`

**Interfaces:**
- Produces: `VerificationReport`, `verify_mass_balance`, `verify_relaxation`, `verify_refinement`, `verify_determinism`, and `run_verification_suite(config) -> VerificationReport`.

- [ ] **Step 1: Write verification-report tests**

```python
def test_mass_report_fails_bad_trajectory():
    report = verify_mass_balance(times, currents, mass_with_drift, tolerance=1e-5)
    assert not report.passed
    assert report.relative_error > 1e-5


def test_canonical_verification_passes():
    report = run_verification_suite(load_config(Path("configs/canonical.toml")))
    assert report.mass_balance.passed
    assert report.relaxation.passed
    assert report.determinism.passed
```

- [ ] **Step 2: Run tests and verify missing implementation**

Run: `.venv/bin/pytest tests/test_verification.py -v`

Expected: FAIL because verification functions are undefined.

- [ ] **Step 3: Implement quantitative gates**

Mass balance compares `mass(t)-mass(0)` against the time integral of total
boundary current. Relaxation requires nonincreasing free energy within numerical
tolerance under zero current. Refinement compares front radius at matched saved
times on `48`, `64`, and `96` grids and reports displacement in coarse pixels.
Determinism requires exact array equality for repeated runs with the same seed.

- [ ] **Step 4: Run and persist verification evidence**

Run: `.venv/bin/python scripts/verify_forward.py --config configs/canonical.toml --out outputs/verification`

Expected: `verification.json` reports all mandatory gates true; otherwise stop
and diagnose before Task 6.

- [ ] **Step 5: Commit verification**

```bash
git add src/graphite_stage_transition/verification.py scripts/verify_forward.py tests/test_verification.py
git commit -m "test: add forward physics verification gates"
```

---

### Task 6: Approved Concentration Visualization

**Files:**
- Create: `src/graphite_stage_transition/visualization.py`
- Create: `scripts/render_forward.py`
- Test: `tests/test_visualization.py`

**Interfaces:**
- Produces: `radial_kymograph`, `render_montage`, `render_diagnostics`, and `render_movie`.

- [ ] **Step 1: Test fixed scales and kymograph dimensions**

Add a `make_visual_test_result()` helper in the test module that creates a
three-frame, `24 x 24` masked concentration sequence without running the solver.

```python
def test_radial_kymograph_shape():
    result, grid = make_visual_test_result()
    kymo = radial_kymograph(result.concentration, grid, bins=24)
    assert kymo.shape == (result.concentration.shape[0], 24)


def test_movie_uses_fixed_stage_limits(monkeypatch, tmp_path):
    result, grid = make_visual_test_result()
    calls = []
    monkeypatch.setattr("matplotlib.axes.Axes.imshow", lambda self, x, **kw: calls.append(kw))
    render_movie(result, grid, tmp_path / "movie.mp4")
    assert all(call["vmin"] == 0.5 and call["vmax"] == 1.0 for call in calls)
```

- [ ] **Step 2: Implement publication-safe renderers**

Use a perceptually uniform sequential colormap, fixed `[0.5, 1.0]` limits,
masked background, time/current/mean-concentration labels, a shared scale bar,
and a separate radial-time kymograph. Do not render atom-like dots or flux arrows.

- [ ] **Step 3: Run tests and render canonical artifacts**

Run: `.venv/bin/pytest tests/test_visualization.py -v`

Run: `.venv/bin/python scripts/render_forward.py --input outputs/smoke_forward/trajectory.npz --out outputs/smoke_forward/rendered`

Expected: deterministic PNG montage, PNG kymograph, MP4, and plotting-data NPZ.

- [ ] **Step 4: Commit visualization**

```bash
git add src/graphite_stage_transition/visualization.py scripts/render_forward.py tests/test_visualization.py
git commit -m "feat: add concentration-field visual diagnostics"
```

---

### Task 7: Synthetic Cases, Noise, Splits, And Manifests

**Files:**
- Create: `src/graphite_stage_transition/synthetic.py`
- Create: `scripts/generate_benchmark.py`
- Test: `tests/test_synthetic.py`

**Interfaces:**
- Produces: `ParameterCase`, `sample_cases`, `assign_case_splits`, `corrupt_concentration`, `generate_case`, and `write_manifest`.

- [ ] **Step 1: Test deterministic sampling and leakage-safe splits**

Define `bounds` in the test as four log-spaced intervals for mobility, barrier,
gradient penalty, and reaction rate. Construct three replicate rows per case
before calling `assign_case_splits`.

```python
def test_lhs_cases_are_deterministic():
    a = sample_cases(64, bounds, seed=20260710)
    b = sample_cases(64, bounds, seed=20260710)
    assert a == b
    assert len({case.case_id for case in a}) == 64


def test_parameter_case_stays_in_one_split():
    rows = assign_case_splits(cases, seed=19)
    grouped = defaultdict(set)
    for row in rows:
        grouped[row.case_id].add(row.split)
    assert all(len(value) == 1 for value in grouped.values())
```

- [ ] **Step 2: Test noise scale and clipping**

```python
def test_noise_is_scaled_to_clean_field_std():
    clean = np.linspace(0.5, 1.0, 400, dtype=float).reshape(4, 10, 10)
    noisy = corrupt_concentration(clean, noise_fraction=0.1, seed=4)
    delta = noisy - clean
    assert np.isclose(delta.std(), 0.1 * clean.std(), rtol=0.15)
    assert noisy.min() >= 0.5
    assert noisy.max() <= 1.0
```

- [ ] **Step 3: Implement Latin hypercube and versioned manifest**

Use `scipy.stats.qmc.LatinHypercube`, transform positive bounds in log space,
generate 64 cases x 3 replicate seeds, and attach 0%, 5%, 10%, and 20% noise
variants plus temporal-subsampling metadata. Hash the canonicalized case JSON to
form stable case IDs.

- [ ] **Step 4: Run tests and generate a four-case smoke benchmark**

Run: `.venv/bin/pytest tests/test_synthetic.py -v`

Run: `.venv/bin/python scripts/generate_benchmark.py --config configs/canonical.toml --cases 4 --replicates 1 --out outputs/benchmark_smoke`

Expected: manifest JSON/CSV, four clean NPZ files, corruption variants, no split
overlap, and no nonfinite values.

- [ ] **Step 5: Commit synthetic pipeline**

```bash
git add src/graphite_stage_transition/synthetic.py scripts/generate_benchmark.py tests/test_synthetic.py
git commit -m "feat: add synthetic benchmark generator"
```

---

### Task 8: Differentiable Loss, Gradient Checks, And Multistart Inversion

**Files:**
- Create: `src/graphite_stage_transition/inversion.py`
- Create: `scripts/run_inversion.py`
- Test: `tests/test_inversion.py`

**Interfaces:**
- Produces: `ParameterTransform`, `InverseProblem`, `loss_components`, `fit_single_start`, `fit_multistart`, and `FitResult`.
- `FitResult` contains transformed/raw estimates, identifiable groups, losses, gradient norm, steps, forward solves, status, runtime, and initialization.

- [ ] **Step 1: Test transform round-trip and positivity**

Add `make_tiny_inverse_problem()` in the test module. It uses a `20 x 20` grid,
eight time steps, and an exactly generated clean trajectory, and returns the
problem, truth, truth groups, a near-truth start, and a displaced start.

```python
def test_parameter_transform_round_trip():
    theta = CHRParameters(mobility=0.7, barrier=1.8, kappa=0.002, reaction_rate=0.3)
    z = transform.to_unconstrained(theta)
    recovered = transform.from_unconstrained(z)
    np.testing.assert_allclose(recovered.as_array(), theta.as_array(), rtol=1e-12)
    assert np.all(recovered.as_array() > 0.0)
```

- [ ] **Step 2: Test autodiff gradient against centered finite differences**

```python
def test_loss_gradient_matches_finite_difference():
    problem, _, _, near_truth, _ = make_tiny_inverse_problem()
    z0 = transform.to_unconstrained(near_truth)
    value, grad = jax.value_and_grad(problem.loss)(z0)
    fd = centered_finite_difference(problem.loss, z0, step=1e-4)
    np.testing.assert_allclose(np.asarray(grad), fd, rtol=2e-2, atol=2e-4)
```

- [ ] **Step 3: Test tiny near-truth recovery**

```python
def test_tiny_clean_recovery_reduces_group_error():
    problem, _, truth_groups, near_truth, displaced = make_tiny_inverse_problem()
    result = fit_multistart(problem, starts=[near_truth, displaced], maxiter=40)
    assert result.best.loss < problem.loss(displaced)
    assert relative_group_error(result.best.groups, truth_groups).max() < 0.05
```

- [ ] **Step 4: Implement normalized masked movie loss**

Simulate only requested observation frames, subtract observations inside the
mask, divide squared error by active pixels x frames, and add separately logged
mass and weak bound penalties. Use `jax.value_and_grad` and wrap it for
`scipy.optimize.minimize(method="L-BFGS-B", jac=True)`.

- [ ] **Step 5: Implement deterministic multistart fitting**

Generate starts in transformed space from a recorded seed, always include the
configured central start, run each start independently, retain all results, and
select the lowest finite objective. Never discard failed starts from reported
success-rate denominators.

- [ ] **Step 6: Run inversion gates**

Run: `.venv/bin/pytest tests/test_inversion.py -v`

Run: `.venv/bin/python scripts/run_inversion.py --manifest outputs/benchmark_smoke/manifest.json --case-index 0 --starts 3 --out outputs/inversion_smoke`

Expected: gradient test passes, tiny recovery meets 5% group error, and every
start produces a recorded status.

- [ ] **Step 7: Commit inversion**

```bash
git add src/graphite_stage_transition/inversion.py scripts/run_inversion.py tests/test_inversion.py
git commit -m "feat: add differentiable multistart inversion"
```

---

### Task 9: Baselines And Controlled Ablations

**Files:**
- Create: `src/graphite_stage_transition/baselines.py`
- Create: `scripts/run_baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Produces: `fit_random_search`, `mean_only_loss`, `simulate_fickian`, `simulate_sharp_interface`, and `BaselineResult`.

- [ ] **Step 1: Test equal-budget random search and mean-only loss**

Construct `problem` with `make_tiny_inverse_problem()`. Define `movie_a` and
`movie_b` as two `3 x 8 x 8` movies with identical masked means but different
left/right spatial patterns, and use an all-true `8 x 8` mask.

```python
def test_random_search_respects_forward_budget():
    result = fit_random_search(problem, bounds, budget=17, seed=5)
    assert result.forward_solves == 17


def test_mean_only_discards_spatial_information():
    assert mean_only_loss(movie_a, movie_b, mask) == pytest.approx(0.0)
    assert spatial_loss(movie_a, movie_b, mask) > 0.0
```

- [ ] **Step 2: Implement Fickian and sharp-interface controls**

The Fickian control uses `dc/dt = D*L(c) + boundary_source` with the same grid,
protocol, save times, and current bookkeeping. The sharp-interface control uses a
radial front with conserved phase fractions, estimates front position from total
mass, and renders the same stage values on the grid. Both return the same public
trajectory fields required by comparison code.

- [ ] **Step 3: Run baseline tests and smoke comparison**

Run: `.venv/bin/pytest tests/test_baselines.py -v`

Run: `.venv/bin/python scripts/run_baselines.py --manifest outputs/benchmark_smoke/manifest.json --case-index 0 --budget 20 --out outputs/baseline_smoke`

Expected: JSON/CSV records equal budgets and spatial/mean-only, Fickian,
sharp-interface, and random-search outcomes.

- [ ] **Step 4: Commit baselines**

```bash
git add src/graphite_stage_transition/baselines.py scripts/run_baselines.py tests/test_baselines.py
git commit -m "feat: add inversion baselines and ablations"
```

---

### Task 10: Identifiability Diagnostics

**Files:**
- Create: `src/graphite_stage_transition/identifiability.py`
- Create: `scripts/run_identifiability.py`
- Test: `tests/test_identifiability.py`

**Interfaces:**
- Produces: `local_hessian`, `residual_jacobian`, `fisher_spectrum`, `parameter_correlation`, `profile_likelihood`, and `IdentifiabilityReport`.

- [ ] **Step 1: Test SVD and degeneracy detection on analytic residuals**

```python
def test_rank_deficiency_is_reported():
    jac = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    report = fisher_spectrum(jac, relative_cutoff=1e-8)
    assert report.rank == 1
    assert report.condition_number == np.inf
```

- [ ] **Step 2: Test profile likelihood fixes the requested coordinate**

Use a two-parameter quadratic test problem with optimum `[0.0, 0.0]`; its
`profile_loss(fixed_index, fixed_value, free_values)` method returns the exact
quadratic objective so the test does not depend on the CHR solver.

```python
def test_profile_likelihood_holds_parameter_fixed():
    profile = profile_likelihood(problem, optimum, index=1, grid=np.array([-0.2, 0.0, 0.2]))
    np.testing.assert_allclose(profile.fixed_values, [-0.2, 0.0, 0.2])
    assert profile.losses[1] <= profile.losses[[0, 2]].min()
```

- [ ] **Step 3: Implement local and profile diagnostics**

Use `jax.jacrev` on a reduced residual vector, compute `J.T @ J`, SVD, effective
rank, condition number, covariance pseudoinverse, and correlation matrix. For
each profile coordinate, fix one transformed parameter and reoptimize all others
from the best fit. Save every profile optimization status.

- [ ] **Step 4: Run tests and smoke report**

Run: `.venv/bin/pytest tests/test_identifiability.py -v`

Run: `.venv/bin/python scripts/run_identifiability.py --fit outputs/inversion_smoke/best_fit.json --manifest outputs/benchmark_smoke/manifest.json --case-index 0 --out outputs/identifiability_smoke`

Expected: finite diagnostics or explicit rank-deficient status; no silent matrix
inverse failure.

- [ ] **Step 5: Commit identifiability**

```bash
git add src/graphite_stage_transition/identifiability.py scripts/run_identifiability.py tests/test_identifiability.py
git commit -m "feat: add parameter identifiability diagnostics"
```

---

### Task 11: Resumable Benchmark And GPU Execution

**Files:**
- Create: `src/graphite_stage_transition/benchmark.py`
- Create: `src/graphite_stage_transition/gpu_payload.py`
- Create: `scripts/run_benchmark.py`
- Create: `scripts/build_kgpu_payload.py`
- Create: `scripts/retrieve_kgpu_results.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `BenchmarkTask`, `build_task_table`, `run_task`, `resume_benchmark`, and `aggregate_task_status`.
- Produces: `build_gpu_payload(project_root, manifest, output_script)` and `retrieve_gpu_results(kernel_output_dir, destination)`.

- [ ] **Step 1: Test task IDs, resume, and failed-task accounting**

```python
def test_resume_skips_only_verified_success(tmp_path):
    table = build_task_table(smoke_manifest, methods=("chr", "fickian"))
    write_success_marker(tmp_path, table[0], checksum="valid")
    pending = resume_benchmark(table, tmp_path)
    assert table[0] not in pending
    assert len(pending) == len(table) - 1
```

- [ ] **Step 2: Implement atomic per-task artifacts**

Write results to a temporary task directory, validate required JSON/NPZ/CSV
contents, then rename atomically and create a checksum-bearing success marker.
Keep failures as JSON with traceback, seed, parameters, and command. Never treat
file existence alone as completion.

- [ ] **Step 3: Run local smoke benchmark**

Run: `.venv/bin/pytest tests/test_benchmark.py -v`

Run: `.venv/bin/python scripts/run_benchmark.py --manifest outputs/benchmark_smoke/manifest.json --max-cases 2 --starts 2 --out outputs/benchmark_run_smoke`

Expected: all task states are success or explicit failure; rerunning skips only
verified successes.

- [ ] **Step 4: Build a self-contained GPU payload**

`kgpu` submits only the text of one Python file. `build_gpu_payload` must create
that file by embedding a base64-encoded tar archive containing `src/`, the
canonical configuration, the smoke manifest, and selected case NPZ files. The
payload extracts itself under `/kaggle/working/graphite_benchmark`, runs the
benchmark, verifies output JSON/NPZ files, and writes
`/kaggle/working/kgpu_graphite_results.tar.gz` plus a compact stdout summary.

Run: `.venv/bin/python scripts/build_kgpu_payload.py --manifest outputs/benchmark_smoke/manifest.json --max-cases 2 --starts 2 --out outputs/kgpu_payload_smoke.py`

Expected: the payload is a standalone Python file and its embedded archive
contains a manifest, selected cases, source package, and configuration.

- [ ] **Step 5: Submit GPU scaling probe and retrieve artifacts**

Run: `kgpu run outputs/kgpu_payload_smoke.py`

Run: `.venv/bin/python scripts/retrieve_kgpu_results.py --kernel-output /home/ns2038/kaggle_runner/.kernel/output --out outputs/kgpu_smoke_results`

Expected: JAX reports Tesla P100, two cases complete, and GPU results match CPU
smoke group estimates within numerical tolerance.

- [ ] **Step 6: Run staged benchmark**

Execute in stages: clean development subset, noisy development subset, locked
validation, then locked test only after configurations are frozen. Start with 16
cases; expand to 64 cases x 3 replicates only if wall time and convergence are
acceptable. Record any reduction from the planned ensemble as a limitation.

- [ ] **Step 7: Commit orchestration before final results**

```bash
git add src/graphite_stage_transition/benchmark.py src/graphite_stage_transition/gpu_payload.py scripts/run_benchmark.py scripts/build_kgpu_payload.py scripts/retrieve_kgpu_results.py tests/test_benchmark.py
git commit -m "feat: add resumable CPU and GPU benchmark runner"
```

---

### Task 12: Aggregate Results, Figures, And Scientific Report

**Files:**
- Create: `src/graphite_stage_transition/reporting.py`
- Create: `scripts/build_report.py`
- Create: `tests/test_reporting.py`
- Modify: `README.md`
- Create: `docs/methods_report.md`

**Interfaces:**
- Produces: `aggregate_results`, `build_recovery_figure`, `build_noise_figure`, `build_identifiability_figure`, `build_ablation_figure`, `build_failure_figure`, and `write_methods_report`.

- [ ] **Step 1: Test aggregation includes failures and locked splits**

```python
def test_aggregation_keeps_failed_runs(tmp_path):
    rows = aggregate_results(tmp_path)
    expected_tasks = sum(1 for _ in tmp_path.glob("tasks/*/task.json"))
    assert len(rows) == expected_tasks
    assert "status" in rows.dtype.names
    assert set(rows["split"]) >= {"development", "validation", "test"}
```

- [ ] **Step 2: Implement predeclared metrics**

Compute per-group relative error, median and 90th percentile by noise/split,
multistart basin agreement, forward-solve budget, held-out trajectory RMSE,
conservation error, Fickian/CHR comparison, and unresolved-target flags. Evaluate
the locked test thresholds exactly once and preserve the generated timestamp and
Git revision.

- [ ] **Step 3: Build reproducible scientific figures**

Generate:

1. forward verification and canonical concentration montage;
2. radial-time stage-transition kymograph;
3. ground-truth versus recovered identifiable groups with all cases;
4. error distributions versus noise and temporal subsampling;
5. profile likelihood, correlation, and Fisher/SVD diagnostics;
6. spatial versus mean-only and full-cycle versus lithiation-only ablations;
7. CHR versus Fickian/sharp-interface held-out predictions;
8. selected success and failure residual maps using a zero-centered diverging scale.

Every figure writes a CSV or NPZ sidecar with plotted values and a caption noting
split, sample count, seeds, noise, normalization, and simulation-only status.

- [ ] **Step 4: Write methods report with guarded conclusions**

The report must contain objective, equations, nondimensionalization, verification,
dataset and split, optimizer and starts, baselines, predeclared criteria, results,
identifiability, failures, limitations, reproducibility commands, and deferred
iSCAT work. Separate successful recovery, non-identifiable targets, and failed
runs. Do not translate nondimensional estimates into real graphite constants.

- [ ] **Step 5: Run final verification**

Run: `.venv/bin/pytest -q`

Run: `.venv/bin/python scripts/build_report.py --benchmark outputs/benchmark_final --out outputs/final_report`

Run: `git diff --check`

Expected: all tests pass; all required figures, sidecars, summary tables, movie,
and report exist; repository-tracked files have no whitespace errors.

- [ ] **Step 6: Commit documentation and report code**

```bash
git add src/graphite_stage_transition/reporting.py scripts/build_report.py tests/test_reporting.py README.md docs/methods_report.md
git commit -m "docs: publish graphite inversion benchmark results"
```

---

## Completion Review

Before declaring the goal complete:

1. Run the full test suite from a clean environment.
2. Re-run the canonical forward simulation and compare checksums/metrics.
3. Confirm the GPU benchmark output was actually retrieved and aggregated.
4. Check that failure rows remain in denominators.
5. Check locked-test evaluation was not used to tune settings.
6. Inspect every figure for fixed scales, units, labels, sample counts, collisions,
   and simulation-only caveats.
7. Review the methods report against the design specification section by section.
8. Run `git status --short` and leave only intentional generated outputs ignored.
9. Use `superpowers:requesting-code-review` and address verified findings.
10. Use `superpowers:verification-before-completion` before updating the durable
    goal to complete.
