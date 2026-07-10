#!/usr/bin/env python3
"""Run random-search, mean-only, Fickian, and sharp-interface comparisons."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.baselines import (
    fit_random_search,
    mean_only_loss,
    simulate_fickian,
    simulate_sharp_interface,
    spatial_loss,
)
from graphite_stage_transition.config import load_config
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.inversion import InverseProblem, ParameterTransform
from graphite_stage_transition.protocols import build_protocol
from graphite_stage_transition.solver import CHRParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="ascii"))
    record = manifest["records"][args.case_index]
    config = load_config(Path(manifest["metadata"]["config_path"]))
    archive = np.load(args.manifest.parent / record["observation_path"])
    observations = jnp.asarray(archive["concentration"])
    grid = make_circle_grid(config.grid)
    protocol = build_protocol(config.protocol, config.solver.dt)
    names = ("mobility", "barrier", "kappa", "reaction_rate")
    declared_bounds = manifest["metadata"]["bounds"]
    transform = ParameterTransform(
        tuple(declared_bounds[name][0] for name in names),
        tuple(declared_bounds[name][1] for name in names),
        config.model.stage2,
        config.model.stage1,
    )
    solver = replace(config.solver, perturbation_amplitude=0.0, seed=int(record["seed"]))
    problem = InverseProblem(
        grid,
        protocol,
        solver,
        observations,
        observations[0],
        transform,
        config.inversion.mass_penalty,
        config.inversion.bound_penalty,
    )
    random_result = fit_random_search(problem, declared_bounds, args.budget, args.seed)

    width_squared = (config.model.stage1 - config.model.stage2) ** 2
    diffusivity_candidates = np.geomspace(0.002, 2.0, args.budget)
    fickian_records = []
    for diffusivity in diffusivity_candidates:
        trajectory = simulate_fickian(grid, protocol, diffusivity, solver, observations[0])
        raw_spatial_loss = spatial_loss(trajectory.concentration, observations, grid.mask)
        fickian_records.append(
            {
                "diffusivity": float(diffusivity),
                "spatial_loss": raw_spatial_loss,
                "normalized_spatial_loss": raw_spatial_loss / width_squared,
                "mean_only_loss": mean_only_loss(trajectory.concentration, observations, grid.mask),
            }
        )
    best_fickian = min(fickian_records, key=lambda row: row["normalized_spatial_loss"])
    initial_mean = float(np.asarray(observations[0])[:, :][np.asarray(grid.mask)].mean())
    sharp = simulate_sharp_interface(
        grid,
        protocol,
        initial_mean,
        config.model.stage2,
        config.model.stage1,
    )
    sharp_spatial_loss = spatial_loss(sharp.concentration, observations, grid.mask)
    sharp_record = {
        "spatial_loss": sharp_spatial_loss,
        "normalized_spatial_loss": sharp_spatial_loss / width_squared,
        "mean_only_loss": mean_only_loss(sharp.concentration, observations, grid.mask),
        "forward_solves": 1,
    }
    payload = {
        "record": record,
        "random_search": asdict(random_result),
        "fickian": {
            "best": best_fickian,
            "forward_solves": args.budget,
            "all": fickian_records,
        },
        "sharp_interface": sharp_record,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baselines.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
