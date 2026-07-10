#!/usr/bin/env python3
"""Run the volume-weighted radial CHR reference and save a 2D raster movie."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from graphite_stage_transition.config import load_config
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.protocols import build_protocol
from graphite_stage_transition.radial import (
    diffuse_interface_width_10_90,
    make_radial_grid,
    rasterize_radial_result,
    simulate_radial,
    verify_radial_full_cycle,
    verify_radial_refinement,
)
from graphite_stage_transition.solver import CHRParameters
from graphite_stage_transition.verification import verify_mass_balance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refinement-cells", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    radial_grid = make_radial_grid(config.grid.nx, config.grid.radius)
    cartesian_grid = make_circle_grid(config.grid)
    protocol = build_protocol(config.protocol, config.solver.dt)
    parameters = CHRParameters(
        config.model.mobility,
        config.model.barrier,
        config.model.kappa,
        config.model.reaction_rate,
        config.model.stage2,
        config.model.stage1,
    )
    radial_result = simulate_radial(radial_grid, protocol, parameters, config.solver)
    raster_result = rasterize_radial_result(radial_result, radial_grid, cartesian_grid)
    mass_gate = verify_mass_balance(
        radial_result.times, radial_result.currents, radial_result.mass
    )
    cycle_gate = verify_radial_full_cycle(
        radial_result.concentration,
        radial_grid,
        parameters.stage2,
        parameters.stage1,
    )
    interface_width = diffuse_interface_width_10_90(parameters)
    interface_cells = interface_width / radial_grid.dr
    refinement_gate = None
    if args.refinement_cells is not None:
        if args.refinement_cells <= radial_grid.cells:
            raise ValueError("refinement cells must exceed the production radial cells")
        fine_grid = make_radial_grid(args.refinement_cells, config.grid.radius)
        fine_result = simulate_radial(fine_grid, protocol, parameters, config.solver)
        refinement_gate = verify_radial_refinement(
            {
                radial_grid.cells: (radial_result, radial_grid),
                fine_grid.cells: (fine_result, fine_grid),
            }
        )
    passed = bool(
        mass_gate.passed
        and cycle_gate.passed
        and (refinement_gate is None or refinement_gate.passed)
        and interface_cells >= 4.0
        and np.all(np.isfinite(np.asarray(radial_result.concentration)))
    )
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "trajectory.npz",
        concentration=np.asarray(raster_result.concentration),
        radial_concentration=np.asarray(radial_result.concentration),
        radial_centers=np.asarray(radial_grid.centers),
        radial_volumes=np.asarray(radial_grid.volumes),
        times=np.asarray(radial_result.times),
        currents=np.asarray(radial_result.currents),
        mass=np.asarray(radial_result.mass),
        free_energy=np.asarray(radial_result.free_energy),
        overpotential=np.asarray(radial_result.overpotential),
        summed_current=np.asarray(radial_result.summed_current),
        cg_residual=np.asarray(radial_result.cg_residual),
        mask=np.asarray(cartesian_grid.mask),
        x=np.asarray(cartesian_grid.x),
        y=np.asarray(cartesian_grid.y),
    )
    summary = {
        "passed": passed,
        "config": asdict(config),
        "metadata": radial_result.metadata,
        "interface_width_10_90": interface_width,
        "interface_cells": interface_cells,
        "mass_balance": asdict(mass_gate),
        "full_cycle": asdict(cycle_gate),
        "refinement": None if refinement_gate is None else asdict(refinement_gate),
        "maximum_cg_residual": float(np.max(np.asarray(radial_result.cg_residual))),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
