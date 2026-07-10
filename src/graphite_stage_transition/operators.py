"""Conservative masked finite-volume spatial operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import Grid


@jax.jit
def masked_laplacian(field: jax.Array, grid: Grid) -> jax.Array:
    """Apply a no-flux Laplacian using canceling active-cell face pairs."""

    values = jnp.where(grid.mask, field, 0.0)
    active_x = grid.mask[1:, :] & grid.mask[:-1, :]
    active_y = grid.mask[:, 1:] & grid.mask[:, :-1]
    flux_x = (values[1:, :] - values[:-1, :]) * active_x / grid.dx
    flux_y = (values[:, 1:] - values[:, :-1]) * active_y / grid.dx

    output = jnp.zeros_like(values)
    output = output.at[:-1, :].add(flux_x / grid.dx)
    output = output.at[1:, :].add(-flux_x / grid.dx)
    output = output.at[:, :-1].add(flux_y / grid.dx)
    output = output.at[:, 1:].add(-flux_y / grid.dx)
    return jnp.where(grid.mask, output, 0.0)
