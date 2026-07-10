# Stage-16 Boundary-Model Manifest

This directory records the exact 16-case, one-replicate cohort generated after
the exposed-face boundary fix. The manifest and split assignments are public so
backend probes can use identical case parameters without resampling under a
different NumPy/SciPy environment.

The concentration arrays remain generated outputs and are not stored in Git.
Regenerate them with:

```bash
python scripts/generate_benchmark.py \
  --config configs/transition.toml \
  --cases 16 --replicates 1 --seed 20260710 \
  --out outputs/benchmark_stage16_boundary_v2_dt000125
```

The backend gate uses the checked-in manifest with `--analytic-targets`; it does
not require or reuse concentration arrays from the invalidated volumetric-source
cohort.
