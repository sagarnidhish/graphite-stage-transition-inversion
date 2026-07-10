#!/usr/bin/env python3
"""Run and save one configured CHR trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from graphite_stage_transition.config import load_config
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.protocols import build_protocol
from graphite_stage_transition.solver import CHRParameters, simulate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    grid = make_circle_grid(config.grid)
    protocol = build_protocol(config.protocol, config.solver.dt)
    parameters = CHRParameters(
        config.model.mobility,
        config.model.barrier,
        config.model.kappa,
        config.model.reaction_rate,
        config.model.stage2,
        config.model.stage1,
    )
    result = simulate(grid, protocol, parameters, config.solver)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "trajectory.npz",
        concentration=np.asarray(result.concentration),
        times=np.asarray(result.times),
        currents=np.asarray(result.currents),
        mass=np.asarray(result.mass),
        free_energy=np.asarray(result.free_energy),
        overpotential=np.asarray(result.overpotential),
        summed_current=np.asarray(result.summed_current),
        cg_residual=np.asarray(result.cg_residual),
        mask=np.asarray(grid.mask),
        x=np.asarray(grid.x),
        y=np.asarray(grid.y),
    )
    summary = {
        "config": asdict(config),
        "metadata": result.metadata,
        "mass_change": float(result.mass[-1] - result.mass[0]),
        "max_cg_residual": float(np.max(np.asarray(result.cg_residual))),
        "finite": bool(np.all(np.isfinite(np.asarray(result.concentration)))),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
