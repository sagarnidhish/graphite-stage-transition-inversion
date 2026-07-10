"""Thermodynamically driven galvanostatic boundary reaction."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ReactionState(NamedTuple):
    rate: jax.Array
    overpotential: jax.Array
    summed_current: jax.Array
    exchange_weight: jax.Array


@jax.jit
def galvanostatic_reaction(
    concentration,
    chemical_potential,
    boundary_weight,
    target_current,
    reaction_rate,
    stage2=0.5,
    stage1=1.0,
):
    """Find the overpotential whose boundary reaction matches total current."""

    phase = jnp.clip((concentration - stage2) / (stage1 - stage2), 0.0, 1.0)
    is_boundary = boundary_weight > 0.0
    exchange = reaction_rate * jnp.sqrt(phase * (1.0 - phase) + 1.0e-6)
    exchange = jnp.where(is_boundary, exchange, 0.0)
    weighted_exchange = boundary_weight * exchange

    positive_sum = jnp.sum(weighted_exchange * jnp.exp(-0.5 * chemical_potential))
    negative_sum = jnp.sum(weighted_exchange * jnp.exp(0.5 * chemical_potential))
    positive_sum = jnp.maximum(positive_sum, jnp.finfo(jnp.float64).tiny)
    negative_sum = jnp.maximum(negative_sum, jnp.finfo(jnp.float64).tiny)

    root = jnp.sqrt(target_current**2 + positive_sum * negative_sum)
    positive_root = (target_current + root) / positive_sum
    negative_root = negative_sum / (root - target_current)
    exp_half_eta = jnp.where(target_current >= 0.0, positive_root, negative_root)
    overpotential = 2.0 * jnp.log(exp_half_eta)

    rate = exchange * jnp.sinh(0.5 * (overpotential - chemical_potential))
    rate = jnp.where(is_boundary, rate, 0.0)
    summed_current = jnp.sum(boundary_weight * rate)
    return ReactionState(rate, overpotential, summed_current, exchange)
