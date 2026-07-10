#!/usr/bin/env python3
"""Compute local and profile identifiability diagnostics for one fitted case."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.config import load_config
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.identifiability import (
    build_identifiability_report,
    profile_likelihood,
    residual_jacobian,
)
from graphite_stage_transition.inversion import InverseProblem, ParameterTransform
from graphite_stage_transition.protocols import build_protocol
from graphite_stage_transition.solver import CHRParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--max-residuals", type=int, default=512)
    parser.add_argument("--profile-points", type=int, default=3)
    parser.add_argument("--profile-half-width", type=float, default=0.15)
    parser.add_argument("--profile-maxiter", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fit_payload = json.loads(args.fit.read_text(encoding="ascii"))
    fit = fit_payload.get("best", fit_payload)
    manifest = json.loads(args.manifest.read_text(encoding="ascii"))
    record = manifest["records"][args.case_index]
    config = load_config(Path(manifest["metadata"]["config_path"]))
    archive = np.load(args.manifest.parent / record["observation_path"])
    observations = jnp.asarray(archive["concentration"])
    grid = make_circle_grid(config.grid)
    protocol = build_protocol(config.protocol, config.solver.dt)
    names = ("mobility", "barrier", "kappa", "reaction_rate")
    bounds = manifest["metadata"]["bounds"]
    transform = ParameterTransform(
        tuple(bounds[name][0] for name in names),
        tuple(bounds[name][1] for name in names),
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
    parameters = CHRParameters(
        *(fit["parameters"][name] for name in names),
        config.model.stage2,
        config.model.stage1,
    )
    optimum = np.asarray(transform.to_unconstrained(parameters))
    jacobian = residual_jacobian(problem, optimum, args.max_residuals)
    profiles = []
    if args.profile_points > 0:
        for index in range(4):
            fixed_grid = np.linspace(
                optimum[index] - args.profile_half_width,
                optimum[index] + args.profile_half_width,
                args.profile_points,
            )
            profiles.append(
                profile_likelihood(
                    problem,
                    optimum,
                    index,
                    fixed_grid,
                    maxiter=args.profile_maxiter,
                )
            )
    report = build_identifiability_report(jacobian, tuple(profiles))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "identifiability.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="ascii",
    )
    np.savez_compressed(
        args.out / "identifiability_arrays.npz",
        jacobian=jacobian,
        fisher=report.fisher_matrix,
        covariance=report.covariance,
        correlation=report.correlation,
    )


if __name__ == "__main__":
    main()
