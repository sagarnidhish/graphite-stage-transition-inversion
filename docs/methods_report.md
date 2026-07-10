# Graphite Stage 2 to Stage 1 Simulation Benchmark

## Claim Boundary

This is a simulation-only effective-scalar Cahn-Hilliard-reaction (CHR) study.
It does not analyze the experimental iSCAT movie, calibrate an optical
observation model, resolve individual lithium ions, or estimate real graphite
material constants.

## Objective and Model

The benchmark tests which morphology observables distinguish synthetic
two-dimensional concentration movies spanning stage 2 (`c=0.5`) to stage 1
(`c=1.0`) and back. The free energy is

```text
F = integral [f(c; A) + kappa |grad c|^2 / 2] dV,
mu = df/dc - kappa laplacian(c),
dc/dt = div(M grad(mu)).
```

A galvanostatic reaction enforces total current on exposed particle faces. The
positive raw parameters `(M, A, kappa, k0)` are fitted in log space. The
dimensionless groups are `epsilon^2 = kappa/(A L^2)`,
`tau_D = L^2/(M A)`, and `Da = k0 L/(M A)`.

## Numerical Verification

The original masked Cartesian solver was found to be unsuitable for
claim-bearing morphology. Its phase interface was subpixel at 48-96 pixels and
its staircase boundary introduced a fourfold source harmonic. A conservative
nine-point operator and analytic arc weights reduced smooth-stencil error but
did not remove the cross-shaped transition. Controlled angular RMS remained
`0.492`, `0.452`, and `0.422` at 48, 64, and 96 pixels.

The circular reference therefore uses concentric annular finite volumes. Cell
areas weight mass, free energy, and the variational chemical potential. Internal
face fluxes cancel pairwise, boundary arc length is exactly `2*pi*R`, and a
square-root-volume similarity transform makes the semi-implicit linear system
symmetric for conjugate gradients. The 1D radial state is mapped to a 2D
Cartesian movie by fixed differentiable radial interpolation.

The production reference uses 192 radial cells and a 384-cell refinement run.
The analytic 10-90 interface spans `7.22` production cells. It passes:

- mean filling `0.5 -> 1.0 -> 0.5`;
- concentration range `[0.49734, 1.00266]`;
- mass relative error `3.12e-14`;
- front refinement displacement `0.0366` production pixels;
- maximum CG residual `3.58e-8`;
- raster angular RMS `0.0107` and maximum deviation `0.0225`.

The earlier Cartesian verification below is retained only as pipeline history,
not current morphology validation.

The cell-centered masked finite-volume Laplacian cancels internal fluxes. The
reaction uses integer exposed-face counts; interior cells have zero reaction and
exchange weight. The semi-implicit stabilized CHR step uses matrix-free
conjugate gradients. The repaired pipeline passes:

- verification mass relative error `2.91e-14`;
- exact deterministic replay and zero detected relaxation increase;
- `48/64/96` refinement displacement `0.168` coarse pixels against a one-pixel gate;
- full-cycle mean filling `0.5 -> 1.0 -> 0.5`;
- full-cycle active concentration range `[0.4679, 1.0315]`;
- maximum current mismatch `8.33e-17` and CG residual `6.99e-7`.

The declared timestep is `dt=0.000125`. All 16 repaired one-replicate cases
complete with finite diagnostics and concentration range `[0.4653, 1.0347]`.
This is pipeline stability evidence, not a validated graphite parameter domain.

## Synthetic Dataset

The declared full design is 64 log-space Latin-hypercube cases with three seeded
replicates, four noise fractions, and temporal subsampling factors. Cases are
assigned to development, validation, or locked test before replicate expansion.

The repaired public manifest records a 16-case, one-replicate cohort: 10
development, 3 validation, and 3 locked-test-labelled cases. Maximum clean-case
mass relative error is `2.72e-14`; maximum current error is `3.61e-16`. The
manifest is public, while generated concentration arrays are reproducible
outputs rather than Git-tracked data.

## Inversion and Controls

The primary inverse objective combines equal-area radial filling, pooled
structure power, and boundary excess with weights `0.50/0.35/0.15`, plus a small
bounds penalty. Raw pixel MSE and mass mismatch remain diagnostics with zero
primary weight. Optimization uses L-BFGS-B in log space and records every
multistart, evaluation count, status, loss component, and gradient norm.
Nonfinite starts fail closed and cannot be selected as successful fits.

Controls include equal-budget random CHR search, implicit Fickian diffusion, a
conserved sharp interface, and a mean-only ablation. Their existing smoke
outputs are retained as controls, not as repaired boundary-kinetics evidence.

## Identifiability

The uncapped identifiability residual is exactly the full inversion residual,
including the bounds term. Reduced Jacobians use deterministic stratified
morphology/bounds sampling so zero in-bounds penalty rows cannot dominate. The
short eight-step test recovers the stronger groups while leaving Damkohler weakly
identified; no converged 16-case recovery or profile-likelihood claim is frozen.

## Reproducibility and Backend Gate

The canonical environment is Python 3.12.13 with exact direct dependency pins.
Task seeds derive from stable task identity. Spawned workers are affinity-bound,
success markers bind source/config/manifest/environment/optimizer hashes, and
failed tasks exit nonzero.

The backend gate recomputes probe evidence from the actual CPU and GPU probe
artifacts. It requires matching source/config/manifest fingerprints, exact target
definition, two clean development cases, observable RMS, objective range/CV, and
gradient direction/norm thresholds. The final gate passed with maximum
observable-block RMS `2.91e-10`, objective range `4.94e-12`, gradient cosine
`1.0`, and maximum gradient norm disagreement `1.12e-7`. The P100 route used
public URLs only and regenerated an analytic charge-consistent target remotely;
no measured data or local concentration arrays were uploaded.

The radial reference was separately probed on canonical CPU and Kaggle P100
from public commit `98ab642`. Maximum concentration difference was `7.21e-11`,
concentration RMS `7.48e-12`, objective difference `9.33e-14`, gradient cosine
`1.0`, and gradient norm disagreement `7.64e-10`; all radial thresholds passed.

## Limitations and Deferred Work

- The scalar field is an effective stage order parameter, not a graphite gallery model.
- The current reference is homogeneous and rotationally symmetric. Its 2D movie
  is rasterized from a 1D radial state and cannot express non-radial mechanisms.
- The Cartesian cohort and inversion outputs predate the radial backend and are
  diagnostic only until regenerated through the new solver.
- Direct concentration is observed; iSCAT optics, drift, and nuisance scales are absent.
- The 16-case one-replicate set is too small for final recovery statistics.
- No nondimensional estimate should be translated into a real material constant.
- Validation and locked-test inversion remain disabled until a recovery budget is frozen;
  the backend reproducibility gate itself has passed.

The next scientific phase should add a validated differentiable iSCAT observation
operator only after simulation-only recovery and identifiability gates pass.
