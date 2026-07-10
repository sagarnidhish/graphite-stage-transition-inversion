"""Differentiable morphology observables for stage-transition movies."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import Grid


RADIAL_BINS = 8
SPECTRAL_BAND_EDGES = (0.0, 1.0, 2.0, 3.0, 4.5, 6.5, np.inf)
RADIAL_WEIGHT = 0.50
STRUCTURE_WEIGHT = 0.35
BOUNDARY_WEIGHT = 0.15


class ObservableGeometry(NamedTuple):
    """Fixed spatial weights shared by every frame of one particle movie."""

    mask: jax.Array
    radial_weights: jax.Array
    boundary_weights: jax.Array
    spectral_band_mask: jax.Array
    active_count: int


class PhysicsObservables(NamedTuple):
    """Rotation-invariant morphology summaries for each movie frame."""

    radial_profile: jax.Array
    structure_power: jax.Array
    boundary_excess: jax.Array


def make_observable_geometry(grid: Grid) -> ObservableGeometry:
    """Precompute normalized radial, boundary, and Fourier pooling weights."""

    mask = jnp.asarray(grid.mask, dtype=bool)
    radius_squared = grid.x**2 + grid.y**2
    radial_indices = jnp.floor(RADIAL_BINS * radius_squared / grid.radius**2).astype(
        jnp.int32
    )
    radial_indices = jnp.clip(radial_indices, 0, RADIAL_BINS - 1)
    radial_weights = jnp.moveaxis(
        jax.nn.one_hot(radial_indices, RADIAL_BINS, dtype=jnp.float64), -1, 0
    )
    radial_weights = radial_weights * mask[None, ...]
    radial_counts = jnp.sum(radial_weights, axis=(1, 2))
    if np.any(np.asarray(radial_counts) == 0):
        raise ValueError("each equal-area radial bin must contain an active cell")
    radial_weights = radial_weights / radial_counts[:, None, None]

    boundary_weights = jnp.where(mask, grid.boundary_weight, 0.0)
    boundary_total = float(jnp.sum(boundary_weights))
    if boundary_total <= 0.0:
        raise ValueError("particle geometry must contain exposed boundary faces")
    boundary_weights = boundary_weights / boundary_total

    nx, ny = mask.shape
    frequency_x = jnp.fft.fftfreq(nx) * nx
    frequency_y = jnp.fft.fftfreq(ny) * ny
    frequency_radius = jnp.sqrt(
        frequency_x[:, None] ** 2 + frequency_y[None, :] ** 2
    )
    spectral_masks = []
    for lower, upper in zip(SPECTRAL_BAND_EDGES[:-1], SPECTRAL_BAND_EDGES[1:]):
        band = (frequency_radius >= lower) & (frequency_radius < upper)
        spectral_masks.append(band & (frequency_radius > 0.0))

    return ObservableGeometry(
        mask=mask,
        radial_weights=radial_weights,
        boundary_weights=boundary_weights,
        spectral_band_mask=jnp.stack(spectral_masks),
        active_count=int(grid.active_count),
    )


def physics_observables(
    movie,
    geometry: ObservableGeometry,
    stage2: float,
    stage1: float,
) -> PhysicsObservables:
    """Calculate radial filling, pooled power, and boundary excess per frame."""

    concentration = jnp.asarray(movie, dtype=jnp.float64)
    if concentration.ndim < 3 or concentration.shape[-2:] != geometry.mask.shape:
        raise ValueError("movie must end in the observable geometry's two spatial axes")
    if not stage2 < stage1:
        raise ValueError("stage2 < stage1 is required")

    filling = (concentration - stage2) / (stage1 - stage2)
    filling = jnp.where(geometry.mask, filling, 0.0)
    particle_mean = jnp.sum(filling, axis=(-2, -1)) / geometry.active_count

    radial_profile = jnp.einsum(
        "...ij,kij->...k", filling, geometry.radial_weights
    )
    demeaned = jnp.where(
        geometry.mask,
        filling - particle_mean[..., None, None],
        0.0,
    )
    transformed = jnp.fft.fft2(demeaned, axes=(-2, -1))
    pixel_count = geometry.mask.shape[0] * geometry.mask.shape[1]
    power = 4.0 * jnp.abs(transformed) ** 2 / (
        pixel_count * geometry.active_count
    )
    structure_power = jnp.einsum(
        "...ij,kij->...k", power, geometry.spectral_band_mask
    )

    boundary_mean = jnp.einsum(
        "...ij,ij->...", filling, geometry.boundary_weights
    )
    boundary_excess = boundary_mean - particle_mean
    return PhysicsObservables(radial_profile, structure_power, boundary_excess)


def observable_residual_vector(
    predicted: PhysicsObservables,
    observed: PhysicsObservables,
) -> jax.Array:
    """Return block-weighted residuals whose mean square is the primary loss."""

    differences = (
        jnp.ravel(predicted.radial_profile - observed.radial_profile),
        jnp.ravel(predicted.structure_power - observed.structure_power),
        jnp.ravel(predicted.boundary_excess - observed.boundary_excess),
    )
    sizes = tuple(difference.size for difference in differences)
    if any(size == 0 for size in sizes):
        raise ValueError("observable blocks must be nonempty")
    total_size = sum(sizes)
    weights = (RADIAL_WEIGHT, STRUCTURE_WEIGHT, BOUNDARY_WEIGHT)
    scaled = tuple(
        difference * jnp.sqrt(weight * total_size / size)
        for difference, weight, size in zip(differences, weights, sizes)
    )
    return jnp.concatenate(scaled)


__all__ = [
    "BOUNDARY_WEIGHT",
    "ObservableGeometry",
    "PhysicsObservables",
    "RADIAL_WEIGHT",
    "STRUCTURE_WEIGHT",
    "make_observable_geometry",
    "observable_residual_vector",
    "physics_observables",
]
