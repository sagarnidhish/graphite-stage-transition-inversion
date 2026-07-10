# Project Status

Updated: 2026-07-10

## Current State

The claim-bearing circular forward model is now a volume-weighted radial
finite-volume CHR solver rasterized into a two-dimensional movie. It replaces
the grid-locked Cartesian transition for the homogeneous circular-particle
baseline. Synthetic cohort and inversion outputs from the Cartesian backend are
retained only as pipeline diagnostics and are not current morphology evidence.

The original cohort and all earlier boundary-kinetics/Damkohler claims were
invalidated. The cause was Boolean neighbor addition in `_boundary_face_count`,
which marked all 1,160 active cells reactive instead of the 108 boundary cells.
The repair now gives 108 boundary cells, 152 exposed faces, and zero interior
reaction/exchange weight on the canonical 48 x 48 circle.

## Verified Evidence

- Full test suite: 149 collected tests pass in the pinned Python 3.12.13
  environment.
- Resolved radial full cycle: 192 production cells, 384-cell refinement
  reference, and `7.22` production cells across the analytic 10-90 interface.
  Mean filling is `0.5 -> 1.0 -> 0.5`, concentration remains in
  `[0.49734, 1.00266]`, and mass relative error is `3.12e-14`.
- Radial front refinement displacement is `0.0366` production pixels against a
  one-pixel gate. Maximum production CG residual is `3.58e-8`.
- The 192x192 raster passes the angular diagnostic with maximum normalized RMS
  `0.0107` and maximum angular deviation `0.0225`.
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

The Cartesian fingerprint-bound CPU/P100 probe passed against public commit
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

The radial reference probe was independently rerun on canonical CPU and Kaggle
P100 from public commit `98ab642`. It passed with concentration maximum absolute
difference `7.21e-11`, concentration RMS `7.48e-12`, objective difference
`9.33e-14`, gradient cosine `1.0`, and gradient norm disagreement `7.64e-10`.

## Claim Boundary

This remains synthetic effective-scalar CHR evidence. It is not an iSCAT
observation model, does not track individual lithium ions, and does not estimate
real graphite material constants. The radial reference assumes a homogeneous,
perfectly circular, rotationally symmetric particle. Its two-dimensional movie
is a rasterization of a one-dimensional radial state, so it cannot represent
non-radial nucleation, defects, facets, or heterogeneous reaction kinetics.

An explicit angular isotropy gate was added in `verification.py` and applied to
the repaired transition trajectory. With 12 radial bins and 16 angular sectors,
the trajectory scores maximum normalized angular RMS `0.497` and maximum angular
deviation `0.629`, failing the `0.05` / `0.10` thresholds. The diagnostic is
archived under `outputs/transition_forward_boundary_v2/rendered/isotropy/` and
should be treated as a discretization failure, not a physically interpretable
stage-front result. The nine-point/analytic-arc repair also failed at 48, 64,
and 96 pixels (`0.492`, `0.452`, `0.422` RMS), so it remains diagnostic only.

## Next Steps

1. Route synthetic generation and inversion through the radial backend for the
   homogeneous circular baseline, then regenerate the cohort.
2. Add a full 2D polar or agglomerated cut-cell backend before making any
   non-radial morphology claim.
3. Add an iSCAT observation operator after the simulation-only radial recovery
   identifiability behavior are understood.
