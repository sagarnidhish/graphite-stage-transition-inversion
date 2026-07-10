# Project Status

Updated: 2026-07-10

## Current State

The repaired simulation-only graphite stage 2 <-> stage 1 pipeline is
implemented through forward CHR physics, sparse differentiable execution,
physics-facing observables, synthetic cohort generation, guarded inversion and
identifiability, baselines, visualizations, and reproducible execution.

The original cohort and all earlier boundary-kinetics/Damkohler claims were
invalidated. The cause was Boolean neighbor addition in `_boundary_face_count`,
which marked all 1,160 active cells reactive instead of the 108 boundary cells.
The repair now gives 108 boundary cells, 152 exposed faces, and zero interior
reaction/exchange weight on the canonical 48 x 48 circle.

## Verified Evidence

- Full test suite: 130 collected tests pass in the pinned Python 3.12.13
  environment.
- Repaired forward verification at `dt=0.000125`: mass relative error
  `2.91e-14`, exact deterministic replay, zero detected relaxation increase,
  and `0.168` coarse-pixel refinement displacement against a one-pixel gate.
- Repaired full cycle: mean filling `0.5 -> 1.0 -> 0.5`; active concentration
  range `[0.4679, 1.0315]`; maximum current mismatch `8.33e-17`; maximum CG
  residual `6.99e-7`.
- Repaired cohort: 16 cases, one replicate, 64 records, split 10 development /
  3 validation / 3 locked-test-labelled; concentration range across clean
  cases `[0.4653, 1.0347]`, mass relative error below `2.72e-14`, current error
  below `3.61e-16`.
- Sparse solver tests prove requested-save alignment, carry-only scan output,
  diagnostics agreement at `1e-12`, and tiny objective/gradient agreement at
  `1e-10` relative tolerance.
- The morphology objective uses equal-area radial filling, pooled structure
  power, and boundary excess with weights `0.50/0.35/0.15`. Raw pixel and mass
  mismatch remain diagnostics with zero optimization weight.
- The exact uncapped inversion residual reproduces the complete objective,
  including the bounds term. Capped Fisher Jacobians use deterministic
  stratified morphology/bounds sampling and are labelled reduced approximations.
- Nonfinite optimizer evaluations and coordinates fail closed and cannot be
  selected as successful multistart fits.

## Backend Gate

The final fingerprint-bound CPU/P100 probe passed against public commit
`800ba6d` and the exact manifest in `benchmarks/stage16_boundary_v2/`. The
fingerprint is `1c156ec7bb7495c87a7f2b19e93d686684ad50f9ae2268c278752ee2cd9031ed`.
The gate required exactly one canonical CPU and one GPU probe, matching probe
hashes, target definition, case count, source/config/manifest hashes, and all
observable/objective/gradient thresholds. Authorization recomputed the gate
from both probe files and passed.

Final cross-backend metrics: maximum observable-block RMS `2.91e-10`, maximum
objective range `4.94e-12`, gradient cosine `1.0`, and maximum gradient norm
disagreement `1.12e-7`. The CPU cases took `160.0` and `178.4` seconds; the
P100 cases took `248.6` and `282.1` seconds.

The P100 run used public URLs only; it regenerated the analytic,
charge-consistent reference target remotely and did not upload local source or
concentration arrays.

## Claim Boundary

This remains synthetic effective-scalar CHR evidence. It is not an iSCAT
observation model, does not track individual lithium ions, and does not estimate
real graphite material constants. The square-grid/circular-mask discretization
also produces visible lattice-oriented morphology, which is retained as a model
limitation rather than interpreted as graphite crystallographic evidence.

## Next Steps

1. Complete and archive the final CPU/P100 observable backend gate.
2. Run only bounded, diagnostic development inversion until a practical budget
   is frozen; keep validation and locked-test fits gated.
3. Add an iSCAT observation operator after the simulation-only objective and
   identifiability behavior are understood.
