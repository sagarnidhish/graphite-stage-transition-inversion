"""Fixed-scale visual diagnostics for concentration-field simulations."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import imageio.v2 as imageio

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "graphite-matplotlib"))
import matplotlib.pyplot as plt
import numpy as np

from .geometry import Grid, radial_bin_indices
from .solver import SimulationResult


STAGE2_CONCENTRATION = 0.5
STAGE1_CONCENTRATION = 1.0


def _stage_colormap():
    return plt.colormaps["cividis"].with_extremes(bad="#f2f2f2")


def _extent(grid: Grid) -> tuple[float, float, float, float]:
    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    half_cell = 0.5 * grid.dx
    return (
        float(x.min() - half_cell),
        float(x.max() + half_cell),
        float(y.min() - half_cell),
        float(y.max() + half_cell),
    )


def _masked_frame(frame, grid: Grid):
    return np.ma.array(np.asarray(frame), mask=~np.asarray(grid.mask))


def _draw_field(axis, frame, grid: Grid, stage2: float, stage1: float):
    image = axis.imshow(
        _masked_frame(frame, grid),
        origin="lower",
        extent=_extent(grid),
        cmap=_stage_colormap(),
        vmin=stage2,
        vmax=stage1,
        interpolation="nearest",
    )
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    return image


def _masked_mean(concentration, grid: Grid) -> np.ndarray:
    values = np.asarray(concentration)
    return values[:, np.asarray(grid.mask)].mean(axis=1)


def radial_kymograph(concentration, grid: Grid, bins: int = 48) -> np.ndarray:
    """Compute radial-bin mean concentration for every saved movie frame."""

    values = np.asarray(concentration)
    if values.ndim != 3 or values.shape[1:] != np.asarray(grid.mask).shape:
        raise ValueError("concentration must have shape (time, nx, ny)")
    indices = radial_bin_indices(grid, bins)
    kymograph = np.full((values.shape[0], bins), np.nan, dtype=np.float64)
    populated = np.zeros((bins,), dtype=bool)
    for radial_bin in range(bins):
        selected = indices == radial_bin
        if np.any(selected):
            populated[radial_bin] = True
            kymograph[:, radial_bin] = values[:, selected].mean(axis=1)
    if not np.any(populated):
        raise ValueError("particle mask has no populated radial bins")
    bin_positions = np.arange(bins)
    for frame in range(values.shape[0]):
        kymograph[frame] = np.interp(
            bin_positions,
            bin_positions[populated],
            kymograph[frame, populated],
        )
    return kymograph


def _add_scale_bar(axis, grid: Grid) -> None:
    length = 0.25 * (2.0 * grid.radius)
    x0 = -0.75 * grid.radius
    y0 = -0.78 * grid.radius
    axis.plot([x0, x0 + length], [y0, y0], color="white", linewidth=3)
    axis.text(
        x0 + 0.5 * length,
        y0 + 0.04 * grid.radius,
        f"{length:.2g} L",
        color="white",
        ha="center",
        va="bottom",
        fontsize=8,
    )


def render_montage(
    result: SimulationResult,
    grid: Grid,
    output: Path,
    stage2: float = STAGE2_CONCENTRATION,
    stage1: float = STAGE1_CONCENTRATION,
    panels: int = 6,
) -> None:
    """Render evenly spaced concentration frames with one fixed color scale."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(result.concentration.shape[0])
    indices = np.unique(np.rint(np.linspace(0, frame_count - 1, min(panels, frame_count))).astype(int))
    figure, axes = plt.subplots(
        1,
        len(indices),
        figsize=(2.25 * len(indices), 2.5),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    means = _masked_mean(result.concentration, grid)
    for axis, index in zip(axes[0], indices):
        image = _draw_field(axis, result.concentration[index], grid, stage2, stage1)
        axis.set_title(
            f"t={float(result.times[index]):.3g}\n"
            f"I={float(result.currents[index]):+.3g}, mean={means[index]:.3f}",
            fontsize=8,
        )
    _add_scale_bar(axes[0, 0], grid)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("Lithium filling, c")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render_kymograph(
    result: SimulationResult,
    grid: Grid,
    output: Path,
    bins: int = 64,
    stage2: float = STAGE2_CONCENTRATION,
    stage1: float = STAGE1_CONCENTRATION,
) -> np.ndarray:
    """Render radial filling versus physical time."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kymograph = radial_kymograph(result.concentration, grid, bins)
    figure, axis = plt.subplots(figsize=(6.4, 3.5))
    image = axis.imshow(
        kymograph.T,
        origin="lower",
        aspect="auto",
        extent=(float(result.times[0]), float(result.times[-1]), 0.0, grid.radius),
        cmap=_stage_colormap(),
        vmin=stage2,
        vmax=stage1,
        interpolation="nearest",
    )
    axis.set_xlabel("Time")
    axis.set_ylabel("Radius")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Lithium filling, c")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return kymograph


def render_diagnostics(result: SimulationResult, grid: Grid, output: Path) -> None:
    """Render scalar forward-solver diagnostics without decorative panels."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(result.times)
    means = _masked_mean(result.concentration, grid)
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), sharex=True)
    axes[0, 0].plot(times, means, color="#0072B2", linewidth=1.8)
    axes[0, 0].set_ylabel("Mean filling")
    axes[0, 1].plot(times, np.asarray(result.currents), color="#D55E00", linewidth=1.5)
    axes[0, 1].set_ylabel("Applied current")
    axes[1, 0].plot(times, np.asarray(result.free_energy), color="#009E73", linewidth=1.5)
    axes[1, 0].set_ylabel("Free energy")
    raw_residual = np.asarray(result.cg_residual)
    positive_residual = raw_residual[raw_residual > 0.0]
    residual_floor = (
        0.5 * float(positive_residual.min()) if positive_residual.size else 1e-16
    )
    residual = np.maximum(raw_residual, residual_floor)
    axes[1, 1].semilogy(times, residual, color="#CC79A7", linewidth=1.2)
    axes[1, 1].set_ylabel("Linear residual")
    for axis in axes[1]:
        axis.set_xlabel("Time")
    for axis in axes.ravel():
        axis.grid(color="#d8d8d8", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render_movie(
    result: SimulationResult,
    grid: Grid,
    output: Path,
    stage2: float = STAGE2_CONCENTRATION,
    stage1: float = STAGE1_CONCENTRATION,
    fps: int = 12,
) -> None:
    """Render an MP4 whose concentration scale remains fixed across time."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    means = _masked_mean(result.concentration, grid)
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
    try:
        for index, frame in enumerate(np.asarray(result.concentration)):
            figure, axis = plt.subplots(figsize=(4.8, 4.4), dpi=120)
            image = _draw_field(axis, frame, grid, stage2, stage1)
            _add_scale_bar(axis, grid)
            axis.set_title(
                f"t={float(result.times[index]):.3g}   "
                f"I={float(result.currents[index]):+.3g}   mean c={means[index]:.3f}",
                fontsize=10,
            )
            colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.03)
            colorbar.set_label("Lithium filling, c")
            figure.tight_layout()
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.append_data(rgba[..., :3].copy())
            plt.close(figure)
    finally:
        writer.close()
