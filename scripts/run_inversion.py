#!/usr/bin/env python3
"""Run deterministic multistart inversion for one benchmark observation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.config import load_config
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.inversion import (
    InverseProblem,
    ParameterTransform,
    fit_multistart,
    generate_starts,
)
from graphite_stage_transition.protocols import build_protocol
from graphite_stage_transition.solver import CHRParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="ascii"))
    record = manifest["records"][args.case_index]
    config_path = Path(manifest["metadata"]["config_path"])
    config = load_config(config_path)
    observation_path = args.manifest.parent / record["observation_path"]
    archive = np.load(observation_path)
    observations = jnp.asarray(archive["concentration"])
    grid = make_circle_grid(config.grid)
    protocol = build_protocol(config.protocol, config.solver.dt)
    if not np.array_equal(np.asarray(protocol.times[protocol.save_indices]), archive["times"]):
        raise ValueError("observation times do not match configured protocol")
    bounds = manifest["metadata"]["bounds"]
    names = ("mobility", "barrier", "kappa", "reaction_rate")
    transform = ParameterTransform(
        lower=tuple(bounds[name][0] for name in names),
        upper=tuple(bounds[name][1] for name in names),
        stage2=config.model.stage2,
        stage1=config.model.stage1,
    )
    central = CHRParameters(
        config.model.mobility,
        config.model.barrier,
        config.model.kappa,
        config.model.reaction_rate,
        config.model.stage2,
        config.model.stage1,
    )
    solver = replace(config.solver, perturbation_amplitude=0.0, seed=int(record["seed"]))
    initial = observations[0]
    problem = InverseProblem(
        grid,
        protocol,
        solver,
        observations,
        initial,
        transform,
        config.inversion.mass_penalty,
        config.inversion.bound_penalty,
    )
    starts = generate_starts(transform, central, args.starts, args.seed)
    result = fit_multistart(
        problem,
        starts,
        maxiter=args.maxiter or config.inversion.max_iterations,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "record": record,
        "best": asdict(result.best),
        "starts": [asdict(start) for start in result.starts],
        "settings": {"start_count": args.starts, "seed": args.seed},
    }
    (args.out / "fit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    (args.out / "best_fit.json").write_text(
        json.dumps(asdict(result.best), indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
