"""Idealized particle geometries and spatial indexing helpers."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import GridConfig


class Grid(NamedTuple):
    """Cell-centered Cartesian grid containing a masked particle."""

    x: jax.Array
    y: jax.Array
    mask: jax.Array
    boundary_weight: jax.Array
    dx: float
    cell_area: float
    radius: float
    active_count: int


def _boundary_face_count(mask: jax.Array) -> jax.Array:
    padded = jnp.pad(mask, 1, constant_values=False)
    neighbor_count = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return jnp.where(mask, 4 - neighbor_count, 0)


def make_circle_grid(config: GridConfig) -> Grid:
    """Create a centered circular particle on a cell-centered square grid."""

    if config.nx != config.ny:
        raise ValueError("the first implementation requires a square grid")
    dx = config.length / config.nx
    x_axis = jnp.linspace(-config.length / 2 + dx / 2, config.length / 2 - dx / 2, config.nx)
    y_axis = jnp.linspace(-config.length / 2 + dx / 2, config.length / 2 - dx / 2, config.ny)
    x, y = jnp.meshgrid(x_axis, y_axis, indexing="ij")
    mask = x**2 + y**2 <= config.radius**2
    face_count = _boundary_face_count(mask)
    boundary_weight = face_count.astype(jnp.float64) * dx
    return Grid(
        x=x,
        y=y,
        mask=mask,
        boundary_weight=boundary_weight,
        dx=float(dx),
        cell_area=float(dx**2),
        radius=float(config.radius),
        active_count=int(mask.sum()),
    )


def radial_bin_indices(grid: Grid, bins: int) -> np.ndarray:
    """Return radial-bin IDs for active cells and -1 outside the particle."""

    if bins < 1:
        raise ValueError("bins must be positive")
    radius = np.sqrt(np.asarray(grid.x) ** 2 + np.asarray(grid.y) ** 2)
    edges = np.linspace(0.0, grid.radius, bins + 1)
    indices = np.digitize(radius, edges[1:-1], right=False).astype(np.int32)
    indices = np.clip(indices, 0, bins - 1)
    indices[~np.asarray(grid.mask)] = -1
    return indices

