#!/usr/bin/env python3
"""Emit a deterministic radial-solver CPU/GPU reproducibility probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import jax
import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.config import SolverConfig
from graphite_stage_transition.protocols import make_constant_protocol
from graphite_stage_transition.radial import make_radial_grid, simulate_radial
from graphite_stage_transition.solver import CHRParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grid = make_radial_grid(64, 0.4)
    protocol = make_constant_protocol(0.005, steps=80, dt=0.00025, save_every=20)
    solver = SolverConfig(0.00025, 1e-9, 400, 0.0, 11)
    base = CHRParameters(0.08, 0.7, 0.0018, 0.22, 0.5, 1.0)
    initial = 0.5 + 0.5 / (1.0 + jnp.exp(-(grid.centers - 0.23) / 0.025))
    target = 0.72 + 0.03 * jnp.cos(jnp.pi * grid.centers / grid.radius)

    def objective(vector):
        parameters = base._replace(mobility=vector[0], kappa=vector[1])
        result = simulate_radial(
            grid, protocol, parameters, solver, initial_concentration=initial
        )
        residual = result.concentration[-1] - target
        return jnp.sum(grid.volumes * residual**2) / grid.total_area

    vector = jnp.asarray([base.mobility, base.kappa], dtype=jnp.float64)
    value, gradient = jax.value_and_grad(objective)(vector)
    result = simulate_radial(grid, protocol, base, solver, initial_concentration=initial)
    payload = {
        "schema_version": 1,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "objective": float(value),
        "gradient": np.asarray(gradient).tolist(),
        "concentration": np.asarray(result.concentration).tolist(),
        "mass": np.asarray(result.mass).tolist(),
        "free_energy": np.asarray(result.free_energy).tolist(),
        "summed_current": np.asarray(result.summed_current).tolist(),
        "cg_residual": np.asarray(result.cg_residual).tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("backend", "device", "objective", "gradient")}, sort_keys=True))


if __name__ == "__main__":
    main()
