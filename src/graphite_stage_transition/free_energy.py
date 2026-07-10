"""Effective two-phase thermodynamics for graphite stage 2 and stage 1."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import Grid


def _scaled_concentration(concentration, stage2, stage1):
    return (concentration - stage2) / (stage1 - stage2)


@jax.jit
def homogeneous_free_energy(concentration, barrier, stage2=0.5, stage1=1.0):
    """Quartic free-energy density with minima at the two graphite stages."""

    phase = _scaled_concentration(concentration, stage2, stage1)
    return 16.0 * barrier * phase**2 * (1.0 - phase) ** 2


@jax.jit
def homogeneous_mu(concentration, barrier, stage2=0.5, stage1=1.0):
    """Analytical concentration derivative of the homogeneous free energy."""

    width = stage1 - stage2
    phase = _scaled_concentration(concentration, stage2, stage1)
    return 32.0 * barrier * phase * (1.0 - phase) * (1.0 - 2.0 * phase) / width


@jax.jit
def total_free_energy(
    concentration,
    grid: Grid,
    barrier,
    kappa,
    stage2=0.5,
    stage1=1.0,
):
    """Return discrete homogeneous plus gradient free energy."""

    field = jnp.where(grid.mask, concentration, 0.0)
    homogeneous = jnp.sum(
        jnp.where(
            grid.mask,
            homogeneous_free_energy(field, barrier, stage2, stage1),
            0.0,
        )
    ) * grid.cell_area

    active_x = grid.mask[1:, :] & grid.mask[:-1, :]
    active_y = grid.mask[:, 1:] & grid.mask[:, :-1]
    active_down = grid.mask[1:, 1:] & grid.mask[:-1, :-1]
    active_up = grid.mask[1:, :-1] & grid.mask[:-1, 1:]
    delta_x = (field[1:, :] - field[:-1, :]) * active_x
    delta_y = (field[:, 1:] - field[:, :-1]) * active_y
    delta_down = (field[1:, 1:] - field[:-1, :-1]) * active_down
    delta_up = (field[1:, :-1] - field[:-1, 1:]) * active_up
    gradient = 0.5 * kappa * (
        (2.0 / 3.0) * (jnp.sum(delta_x**2) + jnp.sum(delta_y**2))
        + (1.0 / 6.0) * (jnp.sum(delta_down**2) + jnp.sum(delta_up**2))
    )
    return homogeneous + gradient
