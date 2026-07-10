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
    padded = jnp.pad(mask, 1, constant_values=False).astype(jnp.int32)
    neighbor_count = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return jnp.where(mask, 4 - neighbor_count, 0)


def _circle_boundary_arc_weights(
    mask: jax.Array,
    x: jax.Array,
    y: jax.Array,
    radius: float,
) -> jax.Array:
    """Assign each reactive cell its angular Voronoi arc on the circle."""
    boundary = np.asarray(_boundary_face_count(mask) > 0)
    flat_indices = np.flatnonzero(boundary)
    if flat_indices.size == 0:
        raise ValueError("circle grid must contain reactive boundary cells")
    x_flat = np.asarray(x).reshape(-1)[flat_indices]
    y_flat = np.asarray(y).reshape(-1)[flat_indices]
    angles = np.mod(np.arctan2(y_flat, x_flat), 2.0 * np.pi)
    order = np.argsort(angles)
    sorted_angles = angles[order]
    forward_gaps = np.diff(np.concatenate((sorted_angles, sorted_angles[:1] + 2.0 * np.pi)))
    sorted_weights = radius * 0.5 * (forward_gaps + np.roll(forward_gaps, 1))
    weights = np.zeros(mask.size, dtype=np.float64)
    weights[flat_indices[order]] = sorted_weights
    return jnp.asarray(weights.reshape(mask.shape), dtype=jnp.float64)


def make_circle_grid(config: GridConfig) -> Grid:
    """Create a centered circular particle on a cell-centered square grid."""

    if config.nx != config.ny:
        raise ValueError("the first implementation requires a square grid")
    dx = config.length / config.nx
    x_axis = jnp.linspace(-config.length / 2 + dx / 2, config.length / 2 - dx / 2, config.nx)
    y_axis = jnp.linspace(-config.length / 2 + dx / 2, config.length / 2 - dx / 2, config.ny)
    x, y = jnp.meshgrid(x_axis, y_axis, indexing="ij")
    mask = x**2 + y**2 <= config.radius**2
    boundary_weight = _circle_boundary_arc_weights(mask, x, y, config.radius)
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
