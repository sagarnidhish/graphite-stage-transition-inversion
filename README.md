# Graphite Stage Transition Inversion

Simulation-only benchmark for differentiable recovery of identifiable physical
parameter groups from synthetic 2D graphite stage 2 to stage 1 concentration
movies.

The first phase operates directly on simulated concentration fields. It does not
fit experimental iSCAT data or claim calibrated graphite parameters.

## Current Scope

The repository implements:

- a conservative differentiable 2D CHR solver on an idealized circular particle;
- an exact-charge stage 2 to stage 1 and reverse transition protocol;
- fixed-scale field movies, montages, diagnostics, and radial-time kymographs;
- leakage-safe Latin-hypercube synthetic cases with noise and subsampling variants;
- log-parameter multistart inversion with gradient verification;
- Fickian, sharp-interface, random-search, and mean-only controls;
- Fisher/SVD, correlation, and profile-likelihood diagnostics;
- resumable checksum-verified CPU and single-file `kgpu` benchmark runners.

## Reproduce Smoke Gates

Use the project-local environment from the repository root:

```bash
PYTHONNOUSERSITE=1 .conda/bin/pytest -q
PYTHONNOUSERSITE=1 .conda/bin/python scripts/verify_forward.py \
  --config configs/canonical.toml --out outputs/verification
PYTHONNOUSERSITE=1 .conda/bin/python scripts/run_forward.py \
  --config configs/transition.toml --out outputs/transition_forward
PYTHONNOUSERSITE=1 .conda/bin/python scripts/generate_benchmark.py \
  --config configs/transition.toml --cases 4 --replicates 1 \
  --out outputs/benchmark_smoke
```

The declared 64-case, three-replicate study is intentionally gated on stable
runtime and converged multistart recovery. Current smoke artifacts are not a
statistically powered result and the locked test split has not been evaluated.
