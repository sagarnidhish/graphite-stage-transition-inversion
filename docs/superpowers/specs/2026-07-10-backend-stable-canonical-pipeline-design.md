# Backend-Stable Canonical Pipeline Repair

Date: 2026-07-10

## Objective

Repair the graphite stage 2 <-> stage 1 simulation benchmark so that it uses a
real boundary-localized reaction, avoids materializing unsaved time steps, fits
physics-facing morphology observables instead of exact domain pixels, and runs
reproducibly on a pinned local CPU environment.

The repaired pipeline remains simulation-only. It does not introduce the iSCAT
observation model or measured data.

## Confirmed Invalidation

`_boundary_face_count` currently adds Boolean JAX arrays. Boolean addition is
logical, so every active cell receives a nonzero boundary weight. On the
canonical 48 x 48 circle, all 1,160 active cells are marked as reactive and the
weight sum is 72.5 instead of a discrete perimeter near 2.5.

The existing synthetic cohort therefore represents a volumetric source, not the
declared boundary-driven CHR model. Existing boundary-kinetics, Damkohler,
parameter-recovery, and identifiability evidence is invalidated. The cohort must
not be reused after the repair.

## Binding Decisions

1. Correct exposed-face counting before any other scientific change.
2. Preserve the exact per-step state recurrence while changing output storage.
3. Use local CPU as the canonical production backend.
4. Pin Python 3.12.13, JAX/JAXLIB 0.10.2, NumPy 2.5.1, and SciPy 1.18.0.
5. Use two spawned CPU workers with six assigned cores each by default.
6. Make physics observables the primary inversion target.
7. Keep raw pixel MSE and mass mismatch as diagnostics with zero primary weight.
8. Use development cases only until solver, observable, gradient, and backend
   gates pass. Validation and locked-test inversions remain forbidden.
9. Run no additional paid GPU scaling until a bounded backend gate passes.

## Boundary Repair

Cast every neighbor mask to an integer before summation. Exposed-face count must
be an integer in `[0, 4]`; interior cells must be zero; boundary weight is
`face_count * dx`.

Required canonical facts:

- A full 3 x 3 mask has face counts
  `[[2,1,2],[1,0,1],[2,1,2]]`.
- The 48 x 48, radius-0.4 circle has 108 boundary cells and 152 exposed faces.
- Its weights are exactly `{0, dx, 2*dx}`.
- Reaction rate and exchange weight are zero on interior cells.

After the fix, rerun sign, current, mass, relaxation, deterministic replay,
timestep, grid-refinement, and full-transition gates. The validated timestep may
change; do not preserve `dt=0.000125` merely for compatibility.

## Sparse Output Execution

Keep the same `lax.scan` sequence and call `semi_implicit_step` exactly once for
every protocol step. Change the scan to return `ys=None` and carry fixed-size
buffers for requested saved states.

Build an integer `step_save_slot` of length `N`: `-1` for unsaved steps and slot
`k` at step `save_indices[k]-1`. Initialize slot zero with the initial state and
zero step diagnostics. At each step, use `lax.cond(slot >= 0)` and indexed updates
to store concentration, mass, energy, overpotential, summed current, and CG
residual. Saved buffers never feed back into concentration.

`SimulationResult` remains unchanged. Save indices must be int32, unique,
strictly increasing, start at zero, end at `N`, and lie in `[0, N]`.

The new solver must match a retained test-only dense reference on CPU float64:

- exact concentration and final-state equality;
- exact state/current/time alignment for irregular saves;
- unchanged diagnostics within `1e-12` absolute and relative tolerance;
- objective and gradient relative error at most `1e-10` on tiny problems;
- all existing conservation and deterministic gates unchanged.

Sparse output is not assumed to solve reverse-mode memory by itself. Block-level
checkpointing is deferred unless measured gradient memory remains limiting.

## Physics Observable Objective

Normalize filling as

```text
u = (c - stage2) / (stage1 - stage2).
```

Use three differentiable, rotation-invariant blocks.

### Equal-Area Radial Profile

Use eight annuli defined by `floor(8 * (r/R)^2)`. Store normalized fixed weights
for each bin and calculate mean filling per frame and bin.

### Radially Pooled Structure Power

Demean `u` inside the mask, zero it outside, calculate `fft2`, and pool normalized
power into six radial frequency bands with edges
`[0, 1, 2, 3, 4.5, 6.5, infinity]`. Exclude the zero mode. Scale pooled power so
the band sum equals four times active-region filling variance.

### Boundary Excess

Using corrected exposed-face weights, calculate boundary-weighted mean filling
minus particle mean filling for every frame.

The frozen primary objective is

```text
Lprimary = 0.50 * Lradial
         + 0.35 * Lstructure
         + 0.15 * Lboundary
         + 1e-4 * Lbounds.
```

Pixel MSE and mass mismatch remain in `LossComponents` as diagnostics but have
zero coefficient in `total`. The residual vector uses square-root block weights
so its mean squared norm matches the primary objective and identifiability uses
the same residual definition.

Public data records:

```python
class ObservableGeometry(NamedTuple):
    mask: jax.Array
    radial_weights: jax.Array
    boundary_weights: jax.Array
    spectral_band_mask: jax.Array
    active_count: int

class PhysicsObservables(NamedTuple):
    radial_profile: jax.Array
    structure_power: jax.Array
    boundary_excess: jax.Array
```

Public functions are `make_observable_geometry(grid)`,
`physics_observables(movie, geometry, stage2, stage1)`, and
`observable_residual_vector(predicted, observed)` with the return records shown
above.

Required observable gates:

- uniform fields give constant radial profiles and zero structure/boundary terms;
- rotation, reflection, and transpose preserve every primary observable;
- pooled structure power satisfies the variance identity;
- equal-mean core/shell reversals remain distinguishable;
- identical movies have zero primary loss;
- observable autodiff agrees with centered finite differences;
- cross-backend observable-block RMS discrepancy is at most 0.02;
- primary-objective range is at most 0.005 and coefficient of variation at most
  5% across canonical CPU and one free GPU check;
- gradient cosine similarity is at least 0.99 and norm disagreement at most 10%,
  unless both norms are below `1e-6`.

## Canonical Execution

Add a canonical requirements file with exact direct dependency versions and an
environment-provenance record. Every benchmark root persists a deterministic
plan and fingerprint containing source, manifest, config, dependency-file,
observable-schema, optimizer, and seed-policy hashes.

Derive task seed from SHA-256 of `(base_seed, task_id)`, never queue position or
manifest index. A result marker is reusable only when task identity, execution
fingerprint, seed, starts, max iterations, and result checksum match.

The CPU runner uses spawn, not fork. Default to two workers and six cores per
worker, leaving two host cores free. Worker setup fixes CPU affinity and BLAS
thread counts before importing JAX. Results remain atomically written and resume
in deterministic plan order.

CHR multistart execution requires a matching passed backend gate. A diagnostic
mode may run development cases without the gate but stamps outputs
`claim_eligible=false`; it rejects validation and test splits.

## Regeneration And Claims

After the boundary repair and solver gates pass:

1. revalidate timestep on staged development cases;
2. regenerate the 16-case one-replicate cohort with a new revision and hashes;
3. run the observable backend gate on two development cases;
4. run tiny recovery and development-only bounded inversion;
5. freeze optimizer and thresholds;
6. evaluate validation and locked test only after explicit gate passage.

No earlier synthetic inversion, Fisher, baseline, or Damkohler result may be
carried into the repaired benchmark as positive evidence.
