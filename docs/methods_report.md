# Graphite Stage 2 to Stage 1 Simulation Benchmark

## Claim Boundary

This is a **simulation-only** effective-scalar Cahn-Hilliard-reaction (CHR)
study. It does not analyze the experimental iSCAT movie, calibrate an optical
observation model, resolve individual lithium ions, or estimate real graphite
material constants.

## Objective and Model

The benchmark tests which parameter combinations can be recovered from synthetic
two-dimensional concentration movies spanning stage 2 (`c=0.5`) to stage 1
(`c=1.0`) and back.

The free energy is

```text
F = integral [f(c; A) + kappa |grad c|^2 / 2] dV,
mu = df/dc - kappa laplacian(c),
dc/dt = div(M grad(mu)).
```

A galvanostatic boundary reaction enforces the requested total current. The
positive raw parameters `(M, A, kappa, k0)` are fitted in log space. The primary
dimensionless groups are

```text
epsilon^2 = kappa / (A L^2)
tau_D     = L^2 / (M A)
Da        = k0 L / (M A).
```

## Numerical Verification

The cell-centered masked finite-volume Laplacian cancels internal fluxes exactly.
The semi-implicit stabilized CHR step treats the stiff gradient and linear
stabilization terms implicitly with matrix-free conjugate gradients. The current
implementation passes these quantitative smoke gates:

- symmetric-cycle mass closure: `6.6e-14` on the canonical smoke;
- imposed-current agreement: `1.7e-15` maximum absolute error;
- verification-suite mass relative error: `7.1e-11`;
- zero-current relaxation: no detected free-energy increase;
- deterministic replay: exact array equality;
- `48/64/96` refinement: `0.134` coarse-pixel maximum front displacement;
- autodiff gradient agreement with centered finite differences;
- full transition mean filling: `0.5 -> 1.0 -> 0.5`;
- fail-closed rejection of finite but physically runaway trajectories.

At the production `46,400` steps, iterative linear-solver errors initially
accumulated to a symmetric-cycle mass drift of `5.3e-6`. Projecting the exact
constant conserved mode after each solve reduces closure to `5.9e-15` absolute
(`2.36e-14` relative to transferred charge) without clipping the field. Synthetic
generation now fails closed on mass balance and summed-current error as well as
field bounds and finiteness.

The four-case smoke initially suggested `dt=0.001` was sufficient, but a staged
16-case run invalidated that setting: a low-mobility, low-`kappa` case left the
physical range while remaining finite. `dt=0.0005` passed that case but failed a
second sampled corner during rapid spinodal growth. The second case completes at
`dt=0.000125` with concentration range `0.463` to `1.036`. The declared ensemble
therefore uses `dt=0.000125`; the complete staged ensemble must pass before this
is called a validated parameter domain.

## Synthetic Dataset and Splits

The declared full design is 64 log-space Latin-hypercube cases with three seeded
replicates, four noise fractions (`0`, `0.05`, `0.10`, `0.20` of clean-field
standard deviation), and temporal subsampling factors (`1`, `2`, `4`). Parameter
cases are assigned to development, validation, or locked test before replicate
expansion, preventing frame or replicate leakage.

The current generated evidence includes a complete 16-case, one-replicate staged
benchmark. All cases traverse both stages, return to stage 2, retain finite
diagnostics, and stay inside the declared overshoot gate. Across the cohort the
concentration range is `0.463` to `1.039`; maximum mass relative error is
`2.64e-14`, maximum summed-current error is `4.04e-15`, and maximum CG residual
is `1.22e-6`. The case split contains 10 development, 3 validation, and 3
locked-test-labelled cases with no overlap.
Only solver stability was checked across all cases; validation and test movies
have not been used to select inverse settings or make recovery claims. This is
still pipeline verification, not a statistically powered recovery result.

## Inversion and Controls

The inverse objective combines normalized masked movie mismatch, mass mismatch,
and a weak physical-bound penalty. Optimization uses L-BFGS-B in log space and
records every multistart, evaluation count, status, runtime, loss component, and
gradient norm. A tiny exact-data gate recovers all three dimensionless groups
within 5% and verifies the differentiated gradient numerically.

Controls include equal-budget random CHR search, implicit Fickian diffusion, a
conserved sharp stage-1-core/stage-2-shell interface, and a mean-only ablation.
On one clean smoke case, preliminary normalized spatial losses are:

| Method | Normalized spatial loss |
|---|---:|
| random CHR, 20 evaluations | 0.0896 |
| fitted implicit Fickian | 0.0928 |
| conserved sharp interface | 0.1662 |

Fickian and sharp-interface mean-only losses are approximately `2.8e-14`
because imposed current determines total mass. Their spatial losses demonstrate
that morphology, not mean filling, supplies model-discrimination information.

## Preliminary Identifiability

A one-iteration full `48x48`, 5,800-step local CPU probe took 116 seconds and
stopped at its declared iteration limit. It must not be interpreted as a
converged parameter estimate. At that point, a 128-residual Jacobian is rank 4
but has Fisher condition number `1.14e5`. The largest absolute parameter
correlations are approximately `0.978` (`M` with `k0`) and `0.940` (`A` with
`kappa`). These values motivate multistart fits and profile likelihoods; they are
not final identifiability claims.

## Reproducibility and Execution State

The repository contains checksum-verified resumable task orchestration. Success
requires matching task metadata, result content, and SHA-256 marker; failed jobs
remain explicit records. The self-contained P100 payload embeds source, config,
selected manifests, and arrays with an archive hash.

GPU submission is currently paused pending explicit approval to upload repository
source and synthetic arrays to Kaggle as an external service. No experimental
movie or measured battery data is included in that payload. Until converged GPU
multistart results are retrieved, no locked-test recovery threshold is evaluated.

## Limitations and Deferred Work

- The scalar field is an effective stage order parameter, not a graphite gallery model.
- Geometry is an idealized circular two-dimensional particle.
- Direct concentration is observed; iSCAT optics, drift, and nuisance scales are absent.
- The 16-case, one-replicate staged set is too small for final recovery statistics.
- Preliminary local inversion and curvature results are not converged.
- No nondimensional estimate should be translated into a real material constant.

The next scientific phase should add a validated differentiable iSCAT observation
operator and compare simulated observables with the measured movie only after the
simulation-only recovery and identifiability gates pass.
