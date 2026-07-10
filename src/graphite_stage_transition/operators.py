"""Conservative masked finite-volume spatial operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import Grid


@jax.jit
def masked_laplacian(field: jax.Array, grid: Grid) -> jax.Array:
    """Apply an isotropic no-flux Laplacian using canceling cell pairs."""

    values = jnp.where(grid.mask, field, 0.0)
    active_x = grid.mask[1:, :] & grid.mask[:-1, :]
    active_y = grid.mask[:, 1:] & grid.mask[:, :-1]
    active_down = grid.mask[1:, 1:] & grid.mask[:-1, :-1]
    active_up = grid.mask[1:, :-1] & grid.mask[:-1, 1:]
    inverse_dx_squared = 1.0 / grid.dx**2
    axial_weight = (2.0 / 3.0) * inverse_dx_squared
    diagonal_weight = (1.0 / 6.0) * inverse_dx_squared
    flux_x = (values[1:, :] - values[:-1, :]) * active_x * axial_weight
    flux_y = (values[:, 1:] - values[:, :-1]) * active_y * axial_weight
    flux_down = (
        (values[1:, 1:] - values[:-1, :-1]) * active_down * diagonal_weight
    )
    flux_up = (
        (values[1:, :-1] - values[:-1, 1:]) * active_up * diagonal_weight
    )

    output = jnp.zeros_like(values)
    output = output.at[:-1, :].add(flux_x)
    output = output.at[1:, :].add(-flux_x)
    output = output.at[:, :-1].add(flux_y)
    output = output.at[:, 1:].add(-flux_y)
    output = output.at[:-1, :-1].add(flux_down)
    output = output.at[1:, 1:].add(-flux_down)
    output = output.at[:-1, 1:].add(flux_up)
    output = output.at[1:, :-1].add(-flux_up)
    return jnp.where(grid.mask, output, 0.0)
