#!/usr/bin/env python3
"""Render fixed-scale diagnostics from a saved forward trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from graphite_stage_transition.geometry import Grid
from graphite_stage_transition.solver import SimulationResult
from graphite_stage_transition.visualization import (
    radial_kymograph,
    render_diagnostics,
    render_kymograph,
    render_montage,
    render_movie,
)


def _load(input_path: Path) -> tuple[SimulationResult, Grid]:
    archive = np.load(input_path)
    x = archive["x"]
    y = archive["y"]
    mask = archive["mask"].astype(bool)
    dx = float(np.median(np.diff(x[:, 0])))
    radius = float(np.sqrt(x[mask] ** 2 + y[mask] ** 2).max() + 0.5 * dx)
    grid = Grid(
        x=x,
        y=y,
        mask=mask,
        boundary_weight=np.zeros_like(x),
        dx=dx,
        cell_area=dx**2,
        radius=radius,
        active_count=int(mask.sum()),
    )
    result = SimulationResult(
        concentration=archive["concentration"],
        times=archive["times"],
        currents=archive["currents"],
        mass=archive["mass"],
        free_energy=archive["free_energy"],
        overpotential=archive["overpotential"],
        summed_current=archive["summed_current"],
        cg_residual=archive["cg_residual"],
        metadata={},
    )
    return result, grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result, grid = _load(args.input)
    render_montage(result, grid, args.out / "concentration_montage.png")
    kymograph = render_kymograph(result, grid, args.out / "radial_kymograph.png")
    render_diagnostics(result, grid, args.out / "forward_diagnostics.png")
    render_movie(result, grid, args.out / "concentration_movie.mp4")
    np.savez_compressed(
        args.out / "plotting_data.npz",
        times=np.asarray(result.times),
        currents=np.asarray(result.currents),
        mean_concentration=np.asarray(result.concentration)[:, np.asarray(grid.mask)].mean(axis=1),
        radial_kymograph=kymograph,
    )


if __name__ == "__main__":
    main()
