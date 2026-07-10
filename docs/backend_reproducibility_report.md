# Backend Reproducibility Report

Date: 2026-07-10

## Decision

The direct pixelwise CHR inversion is not cleared for multistart scaling.
Exact-central forward losses differ materially across CPU, P100, and A100
backends even though every run used identical source and synthetic-data assets.
Parameter-recovery, identifiability, validation, and locked-test runs remain
stopped until a backend-reproducibility gate passes.

## Controlled Inputs

- Source release: `v0.1.1-stage16`, commit `07acf24`.
- Source SHA-256:
  `d46beb09c18823c246748808934484efbcb1d98ac0b23abf9dc00503010550d7`.
- Synthetic cohort SHA-256:
  `c26be9fca5328cabd3b7c7c414ea366720e861de536f77a01f3fef830323e277`.
- Cases: clean development records 4 and 12.
- Parameters: canonical center `(0.2, 1.0, 0.0015, 0.25)`.
- Solver: float64, `dt=0.000125`, CG tolerance `1e-8`, at most 200 CG
  iterations, 46,400 protocol steps, and 157 saved frames.

## Exact-Central Forward Gate

The reported total is the normalized masked pixelwise movie loss plus the mass
penalty and the `1e-8`-weighted bound penalty. The movie term dominates every
row.

| Backend | JAX | Case | Total loss | Runtime (s) |
|---|---:|---|---:|---:|
| CPU | 0.10.2 | `case_17909d05ebd0` | 0.22534834 | 22.10 |
| P100 | 0.7.2 | `case_17909d05ebd0` | 0.23658265 | 51.98 |
| A100-SXM4-80GB | 0.10.2 | `case_17909d05ebd0` | 0.08843392 | 59.05 |
| CPU | 0.10.2 | `case_3abdd11a7d1c` | 0.04725578 | 21.19 |
| P100 | 0.7.2 | `case_3abdd11a7d1c` | 0.11347423 | 52.50 |
| A100-SXM4-80GB | 0.10.2 | `case_3abdd11a7d1c` | 0.23066735 | 57.39 |

CPU and A100 disagree despite using the same JAX version. The mismatch is
therefore not explained by the P100's older JAX version alone.

## Inversion Timing Evidence

The first P100 probe used the pre-cache source and completed two one-start,
one-iteration tasks in 2,572.96 seconds. Reusing the final value, gradient, and
loss components reduced the controlled P100 repeat to 1,787.24 seconds, a 30.5%
wall-time reduction. Both optimizers intentionally stopped at the one-iteration
limit and are not recovery evidence.

The same cached source took 2,994.72 seconds on Modal A100-SXM4-80GB. The A100
run completed both task artifacts, but it was 1.68 times slower than the cached
P100 repeat. Modal reported approximately $2.16 of gross resource cost across
the connectivity, image-build, device, CPU, and memory probes before account
credits.

## Interpretation

The measured failure is a backend-dependent phase morphology, not a mass or
current-closure failure. The most plausible mechanism is amplification of
floating-point and reduction-order differences during rapid spinodal growth,
combined with tolerance-terminated CG solves. This mechanism is an inference;
field-level divergence diagnostics are still required to localize the first
departing time step.

The `48 x 48` problem is also too small and sequential to use the A100
efficiently. A faster accelerator does not compensate for 46,400 serial steps,
small stencil kernels, repeated reductions, and materialization of every state
before selecting 157 output frames.

## Required Repair Gate

1. Pin and record JAX/JAXLIB versions for every execution backend.
2. Locate the first CPU/GPU trajectory divergence using saved field, phase
   fraction, structure-factor, mass, current, and CG-residual diagnostics.
3. Test a deterministic fixed-iteration or otherwise backend-stable linear
   solve policy without weakening the validated physical gates.
4. Avoid materializing unsaved trajectory states while preserving the exact
   time-stepping map and its gradients.
5. Require agreement of physics-facing observables and the inversion objective
   across CPU and GPU before resuming multistart development fits.
6. If pixelwise agreement remains chaotic, replace the primary fitting target
   with registered or morphology-statistical observables and retain raw
   pixelwise loss only as a fixed-backend diagnostic.

No validation or locked-test inversion should run before this gate passes.
