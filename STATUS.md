# Project Status

Updated: 2026-07-10

## Current State

The simulation-only graphite stage 2 to stage 1 benchmark is implemented through
forward physics, verification, visualization, synthetic generation, inversion,
baselines, identifiability, resumable execution, GPU payload construction, and
guarded reporting.

The project is not scientifically complete. Full multistart GPU recovery,
profile likelihoods at a converged optimum, staged ensemble expansion, and the
locked-test evaluation have not run.

Multistart scaling is now explicitly stopped by a failed backend-reproducibility
gate. See `docs/backend_reproducibility_report.md`.

## Verified Evidence

- Test suite: `53 passed` with `PYTHONNOUSERSITE=1 .conda/bin/pytest -q`.
- Canonical mass relative error: `7.1e-11`.
- Deterministic replay: exact equality.
- Zero-current relaxation: no detected energy increase.
- Grid refinement: `0.134` coarse pixels against a `1.0` pixel gate.
- Full transition: mean filling `0.5 -> 1.0 -> 0.5` at `dt=0.000125`.
- Full-transition symmetric mass closure after conserved-mode projection:
  `5.9e-15` absolute (`2.36e-14` relative to transferred charge).
- Sixteen-case staged benchmark: every case finite, physically bounded, and split-safe.
- Autodiff: agrees with centered finite differences.
- Full local inversion probe: finite but stopped after one iteration; 116 seconds.
- Preliminary local Fisher condition number: `1.14e5` at the unconverged probe.
- Baseline normalized losses on one clean smoke case: random CHR `0.0896`,
  Fickian `0.0928`, sharp interface `0.1662`.
- Public Kaggle P100 scaling probe: two clean development tasks completed with
  checksum-verified inputs and outputs in `2572.96` seconds (`42.9` minutes).
  Each task used one start and an intentionally nonconverged one-iteration
  L-BFGS-B budget, requiring four and six objective/gradient evaluations.
- The P100 probe exposed two redundant full simulations after every optimizer
  run. The final value, gradient, and loss components are now returned together
  and reused; the regression is covered by the 53-test suite. The identical
  repeat completed in `1787.24` seconds (`29.8` minutes), a `30.5%` speedup.
- Modal A100-SXM4-80GB completed the same cached two-task probe in `2994.72`
  seconds (`49.9` minutes), `1.68x` slower than the cached P100 repeat. Modal
  reported approximately `$2.16` gross resource cost before account credits.
- Exact-central forward losses failed backend agreement. For development cases
  `case_17909d05ebd0` and `case_3abdd11a7d1c`, CPU JAX 0.10.2 returned
  `0.22535` and `0.04726`; P100 JAX 0.7.2 returned `0.23658` and `0.11347`;
  A100 JAX 0.10.2 returned `0.08843` and `0.23067`. Identical source and data
  SHA-256 hashes were verified for every remote run.
- The earlier `dt=0.001` setting was invalidated by `case_4996769a95fe`.
  `dt=0.0005` then failed `case_ca2908ecd0c2`. That second case completes at
  `dt=0.000125` with concentration range `0.463` to `1.036`.
- The complete 16-case, one-replicate forward cohort passes at `dt=0.000125`:
  global range `0.463` to `1.039`, maximum mass relative error `2.64e-14`,
  maximum summed-current error `4.04e-15`, 10 development cases, 3 validation
  cases, 3 locked-test-labelled cases, and no case-level split overlap. No
  validation or test inversion was used for tuning.

## Active Limitation

The direct pixelwise inversion is both too expensive and backend sensitive.
Roundoff/reduction differences during spinodal growth are the leading mechanism
hypothesis, but field-level divergence localization is still required. The
current CPU/P100/A100 mismatch invalidates cross-backend parameter-recovery and
identifiability claims. No experimental iSCAT movie or measured battery data was
submitted; remote jobs downloaded only public, checksum-pinned source and
synthetic Stage-16 assets.

## Next Steps

1. Pin execution versions and localize the first CPU/GPU field divergence.
2. Design and validate backend-stable linear-solve and sparse-output execution.
3. Require physics-facing observable and objective agreement across backends.
4. Freeze a practical multistart iteration budget using development cases only.
5. Run converged development fits and profile likelihoods.
6. Evaluate validation, then locked test exactly once after settings are frozen.
7. Refresh figures and methods report with converged evidence.

## Claim Boundary

Current results are synthetic effective-scalar CHR evidence. They do not analyze
the experimental iSCAT movie, validate an optical observation model, or estimate
real graphite material constants.
