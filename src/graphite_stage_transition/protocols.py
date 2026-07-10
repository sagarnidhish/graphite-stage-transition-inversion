"""Electrochemical current protocols aligned to solver time steps."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import ProtocolConfig


class Protocol(NamedTuple):
    current: jax.Array
    times: jax.Array
    save_indices: jax.Array
    dt: float


def make_constant_protocol(current: float, steps: int, dt: float, save_every: int = 1) -> Protocol:
    """Construct a constant-current protocol with state-aligned save indices."""

    if steps < 1 or dt <= 0.0 or save_every < 1:
        raise ValueError("steps, dt, and save_every must be positive")
    currents = jnp.full((steps,), float(current), dtype=jnp.float64)
    times = jnp.arange(steps + 1, dtype=jnp.float64) * dt
    saves = np.arange(0, steps + 1, save_every, dtype=np.int32)
    if saves[-1] != steps:
        saves = np.append(saves, steps)
    return Protocol(currents, times, jnp.asarray(saves), float(dt))


def build_protocol(config: ProtocolConfig, dt: float) -> Protocol:
    """Expand piecewise-constant segments onto a common fixed time step."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    current_parts: list[np.ndarray] = []
    save_indices = [0]
    offset = 0
    for current, duration in zip(config.currents, config.durations):
        raw_steps = duration / dt
        steps = int(round(raw_steps))
        if steps < 1 or not np.isclose(steps * dt, duration, rtol=0.0, atol=1e-12):
            raise ValueError("each protocol duration must be an integer multiple of dt")
        current_parts.append(np.full(steps, current, dtype=np.float64))
        segment_saves = np.rint(
            np.linspace(0, steps, config.frames_per_segment, endpoint=True)
        ).astype(np.int32)
        save_indices.extend((offset + segment_saves[1:]).tolist())
        offset += steps

    currents = np.concatenate(current_parts)
    saves = np.unique(np.asarray(save_indices, dtype=np.int32))
    if saves[-1] != len(currents):
        saves = np.append(saves, len(currents))
    times = np.arange(len(currents) + 1, dtype=np.float64) * dt
    return Protocol(jnp.asarray(currents), jnp.asarray(times), jnp.asarray(saves), float(dt))

