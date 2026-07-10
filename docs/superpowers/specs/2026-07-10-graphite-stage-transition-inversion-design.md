# Graphite Stage 2 to Stage 1 Inversion Benchmark

Date: 2026-07-10

## 1. Objective

Build a simulation-only, reproducible benchmark that tests which physical
parameters can be recovered from synthetic two-dimensional concentration movies
of the graphite stage 2 <-> stage 1 transition.

The first project phase deliberately excludes the iSCAT observation model. The
inverse problem sees the simulated concentration field directly. A later project
may insert a differentiable transport-to-iSCAT observation operator after the
direct-field inversion has passed its verification and identifiability gates.

## 2. Scientific Claim Boundary

The intended methods claim is:

> Given synthetic stage 2 <-> stage 1 concentration movies on idealized 2D
> particles, determine which thermodynamic, transport, and reaction parameter
> combinations can be recovered by differentiating through a
> Cahn-Hilliard-reaction simulator.

The project will not claim that it has measured real graphite parameters,
validated a real battery particle, resolved individual lithium ions, or
established a calibrated relationship between iSCAT intensity and lithium
concentration.

## 3. Modeling Hierarchy

### 3.1 Primary model: effective scalar CHR

Use a scalar lithium filling field `c(x, y, t)` and represent stage 2 and stage
1 by the two minima of an effective homogeneous free-energy density. In graphite
notation, stage 2 is LiC12 (`c = 0.5`) and stage 1 is LiC6 (`c = 1.0`).

The free energy is

```text
F[c] = integral_Omega (f(c; A) + 0.5 * kappa * |grad c|^2) dV
```

and the variational chemical potential is

```text
mu = df/dc - kappa * laplacian(c).
```

Bulk evolution follows conserved Cahn-Hilliard dynamics:

```text
dc/dt = div(M * grad(mu)).
```

Lithium enters or leaves only through the particle boundary. The normal flux is
coupled to a thermodynamically consistent Butler-Volmer-type reaction law:

```text
-n dot (M * grad(mu)) = R(c, mu, eta; k0).
```

The initial implementation will use the simplest reaction law that preserves
the correct equilibrium condition and insertion/extraction signs. More complex
graphite-specific cooperative kinetics are outside the first implementation.

### 3.2 Baseline model: sharp moving boundary

Implement a lower-complexity stage-1-core/stage-2-shell moving-front baseline.
It will be used to test whether diffuse-interface complexity is necessary to
explain a generated trajectory. It is not the primary inversion model because
nucleation and interface width are prescribed rather than predicted.

### 3.3 Later extension: graphite two-layer CHR

A two-layer model with coupled gallery concentrations is explicitly deferred.
It is more faithful to graphite staging but introduces additional parameters and
latent fields that will become difficult to identify once an optical observation
operator is added.

## 4. Geometry And Protocol

### 4.1 Geometry

The first benchmark uses idealized circular 2D particles. The numerical domain,
mask, and physical length scale are explicit configuration values. Ellipses and
other smooth idealized shapes are reserved for the geometry-robustness phase.

### 4.2 Electrochemical protocol

Each canonical trajectory contains:

1. stage 2 initial state;
2. constant-current lithiation;
3. zero-current rest;
4. stage 1 state;
5. equal-magnitude reverse current for delithiation;
6. final zero-current rest toward stage 2.

Small, seeded perturbations may initiate nucleation. Every perturbation seed and
amplitude must be saved. Current sign, boundary-flux sign, and concentration
change must be tested together.

## 5. Parameters And Identifiability

The nominal raw parameters are:

- `M`: mobility or effective diffusivity factor;
- `A`: homogeneous free-energy barrier or interaction strength;
- `kappa`: gradient penalty controlling diffuse-interface structure;
- `k0`: boundary reaction-rate scale.

The primary recovery targets are identifiable dimensionless combinations rather
than an assumed set of independently recoverable dimensional constants. With
particle length scale `L`, important combinations include

```text
epsilon^2 = kappa / (A * L^2)
tau_D     = L^2 / (M * A)
Da        ~ k0 * L / (M * A)
```

The exact reaction nondimensionalization will define the final Damkohler group.
The benchmark will:

1. recover independent dimensionless groups as its primary task;
2. attempt raw `(M, A, kappa, k0)` recovery as a stress test;
3. use profile likelihoods, local curvature, singular values, and multistart
   solutions to identify degeneracies;
4. report non-identifiability as a scientific result rather than forcing a
   unique estimate.

All positive quantities are optimized in log space.

## 6. Synthetic Benchmark Dataset

Generate 64 ground-truth parameter combinations with Latin hypercube sampling
over predeclared, numerically stable nondimensional ranges. For each combination,
run three seeded nucleation/noise replicates, producing 192 underlying clean
trajectories.

For each trajectory, save:

- concentration movie;
- time and applied-current arrays;
- particle mask and coordinate grid;
- exact raw parameters and dimensionless groups;
- free-energy history;
- total-mass and boundary-flux histories;
- solver settings, grid, timestep, seed, and code revision.

Create direct-field observation variants with additive noise at 0%, 5%, 10%,
and 20% of the clean field standard deviation. Clip observations to the physical
concentration interval. Also create documented temporal-subsampling variants.

Parameter cases, not individual frames, are divided into development,
validation, and locked test subsets. Solver choices, optimizer settings, and
stopping criteria must not be tuned on the locked test cases.

## 7. Inversion

Implement the forward solver in JAX and differentiate through its time stepping.
The primary objective is

```text
loss = movie_mismatch
     + lambda_mass * mass_balance_penalty
     + lambda_prior * weak_bound_penalty.
```

The movie mismatch is computed only inside the particle mask and normalized so
that grid size and frame count do not change its scale. Strong priors are not
allowed in the primary recovery benchmark because they could create apparently
accurate estimates without informative movies.

Use multiple optimization starts for every reported recovery. Record initial
parameters, accepted steps, loss components, gradient norms, termination status,
wall time, and forward-solve count.

## 8. Baselines And Ablations

The benchmark includes:

- gradient-based inversion versus random or grid search at equal forward-solve
  budget;
- full spatial movie versus particle-mean concentration only;
- full lithiation-rest-delithiation cycle versus lithiation only;
- CHR inversion versus the sharp-interface baseline;
- correct CHR model versus deliberately misspecified Fickian diffusion;
- fixed `kappa` versus jointly fitted `kappa`;
- full temporal sampling versus reduced frame rates.

At least one held-out prediction test must fit a restricted part of a trajectory
and predict later or reverse-direction frames.

## 9. Verification Gates

No benchmark ensemble may run until the forward model passes:

1. analytical free-energy derivative checks;
2. discrete gradient, divergence, and Laplacian checks;
3. boundary-flux sign tests;
4. relative mass-balance error below `1e-5`;
5. decreasing free energy during an unforced relaxation test;
6. timestep refinement;
7. grid refinement with front-position change below one coarse-grid pixel;
8. finite-difference versus automatic-differentiation gradient checks;
9. deterministic reproduction from a saved seed;
10. tiny synthetic recovery from a near-truth and a displaced initialization.

Large ensemble runs use GPU compute only after local small-grid tests pass.

## 10. Predeclared Success Criteria

- Clean synthetic data: median relative error below 5% for each identifiable
  dimensionless group.
- 10% noise: median relative error below 15% for each identifiable group.
- At least 90% of clean cases converge to the same basin across multiple starts.
- Forward trajectories reconstructed from recovered parameters satisfy the same
  conservation checks as the ground truth.
- Fits using spatial movies outperform fits using only particle-average
  concentration on held-out trajectory prediction.
- CHR fits outperform deliberately misspecified Fickian fits where the generated
  data contain a moving two-phase boundary.
- Any target failing identifiability tests is labeled unresolved and excluded
  from positive recovery claims.

Thresholds are fixed before locked-test evaluation. If they prove unrealistic,
the original result remains recorded and any revised threshold creates a new
benchmark version.

## 11. Visualization

Use the approved concentration-field visualization:

- fixed color limits for every comparable frame;
- stage 2 labeled at `c = 0.5` and stage 1 at `c = 1.0`;
- synchronized current, time, and mean concentration;
- a separate radial-time kymograph showing phase-front propagation;
- small-multiple frame sequences for static figures;
- no artificial individual-ion dots and no claim of atomistic resolution.

Recovery figures include ground-truth versus estimate plots, complete seed-level
error distributions, error versus noise, profile likelihoods, parameter
correlation maps, singular-value diagnostics, ablation comparisons, and selected
failure cases. Signed spatial residuals use a diverging scale centered at zero.

## 12. Repository Architecture

```text
graphite_stage_transition_inversion/
|-- configs/
|-- src/graphite_stage_transition/
|   |-- free_energy.py
|   |-- geometry.py
|   |-- chr_solver.py
|   |-- protocols.py
|   |-- synthetic_data.py
|   |-- inversion.py
|   `-- identifiability.py
|-- scripts/
|-- tests/
|-- figures/
|-- outputs/
|-- docs/
`-- README.md
```

Modules must expose narrow, testable interfaces. Generated datasets and large
figures are excluded from Git; compact manifests, summaries, and plotting data
remain versioned when useful.

## 13. Reproducible Outputs

The completed first phase provides:

- verified forward solver and test suite;
- canonical stage 2 -> stage 1 -> stage 2 movie;
- radial-time kymograph and frame montage;
- versioned synthetic benchmark manifest;
- parameter-recovery and identifiability tables;
- baseline and ablation results;
- publication-style figures with regenerating commands;
- a methods report stating successful, failed, and non-identifiable targets;
- exact environment, configuration, seed, and command records.

## 14. Key Risks And Responses

- **Raw parameters are structurally confounded.** Score dimensionless groups
  first and use profile likelihoods before claiming individual recovery.
- **Diffuse interface is under-resolved.** Enforce minimum points across the
  interface and grid-convergence gates.
- **Boundary reaction and mobility trade off.** Include rest and reverse-current
  segments and compare multiple protocol rates in later ablations.
- **Optimization succeeds only near truth.** Require broad multistart tests and
  equal-budget derivative-free baselines.
- **Synthetic noise is too convenient.** Keep the first benchmark simple but
  predeclare later correlated and structured-noise extensions.
- **Effective model is mistaken for complete graphite physics.** Keep the claim
  boundary visible in the README, captions, and report.

## 15. Deferred Work

- real-particle masks;
- direct fitting of the available experimental movie;
- differentiable iSCAT signal formation;
- optical calibration and microscope nuisance parameters;
- graphite two-layer or multilayer staging;
- elasticity, defects, cracks, grain boundaries, and full-cell coupling;
- calibrated dimensional graphite parameters.

## 16. Scientific Basis

- Guo et al., "Li Intercalation into Graphite: Direct Optical Imaging and
  Cahn-Hilliard Reaction Dynamics," J. Phys. Chem. Lett. 7, 2151-2156 (2016),
  https://doi.org/10.1021/acs.jpclett.6b00625
- Zeng and Bazant, "Phase Separation Dynamics in Isotropic Ion-Intercalation
  Particles," SIAM J. Appl. Math. 74, 980-1004 (2014),
  https://arxiv.org/abs/1309.4543
- Cohen et al., "Differentiable Learning and Control of Free-Energy-Driven
  Pattern Dynamics," Phys. Rev. Research 8, 023344 (2026),
  https://doi.org/10.1103/b8kc-vpwq
- Cordoba, Chandesris, and Plapp, "Spinodal decomposition and domain coarsening
  in a multi-layer Cahn-Hilliard model for lithium intercalation in graphite,"
  https://arxiv.org/abs/2401.13108
