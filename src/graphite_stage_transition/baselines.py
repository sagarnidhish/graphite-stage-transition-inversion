"""Controlled inversion and forward-model baselines for CHR comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from .config import SolverConfig
from .geometry import Grid
from .operators import masked_laplacian
from .protocols import Protocol
from .solver import CHRParameters, SimulationResult, particle_mass


@dataclass(frozen=True)
class BaselineResult:
    method: str
    loss: float
    parameters: dict[str, float]
    forward_solves: int


def spatial_loss(first, second, mask) -> float:
    """Normalized pixelwise movie mismatch inside the particle."""

    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    if first_values.shape != second_values.shape or first_values.shape[-2:] != active.shape:
        raise ValueError("movie and mask dimensions do not agree")
    return float(np.mean((first_values[..., active] - second_values[..., active]) ** 2))


def mean_only_loss(first, second, mask) -> float:
    """Movie loss after deliberately discarding all spatial information."""

    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    if first_values.shape != second_values.shape or first_values.shape[-2:] != active.shape:
        raise ValueError("movie and mask dimensions do not agree")
    difference = first_values[..., active].mean(axis=-1) - second_values[..., active].mean(axis=-1)
    return float(np.mean(difference**2))


def fit_random_search(
    problem,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    budget: int = 20,
    seed: int = 0,
) -> BaselineResult:
    """Evaluate exactly ``budget`` seeded log-uniform CHR parameter draws."""

    if budget < 1:
        raise ValueError("budget must be positive")
    names = ("mobility", "barrier", "kappa", "reaction_rate")
    if bounds is None:
        lower = np.asarray(problem.transform.lower, dtype=np.float64)
        upper = np.asarray(problem.transform.upper, dtype=np.float64)
    else:
        lower = np.asarray([bounds[name][0] for name in names])
        upper = np.asarray([bounds[name][1] for name in names])
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(np.log(lower), np.log(upper), size=(budget, 4))
    objective = jax.jit(problem.loss)
    losses = np.asarray([float(objective(jnp.asarray(candidate))) for candidate in candidates])
    finite = np.flatnonzero(np.isfinite(losses))
    if not finite.size:
        raise RuntimeError("random-search baseline produced no finite losses")
    best_index = int(finite[np.argmin(losses[finite])])
    raw = np.exp(candidates[best_index])
    return BaselineResult(
        method="random_search_chr",
        loss=float(losses[best_index]),
        parameters={name: float(value) for name, value in zip(names, raw)},
        forward_solves=budget,
    )


def _assemble_result(
    concentrations,
    grid: Grid,
    protocol: Protocol,
    summed_current,
) -> SimulationResult:
    save = np.asarray(protocol.save_indices)
    all_concentrations = jnp.asarray(concentrations)
    saved = all_concentrations[save]
    mass = jax.vmap(lambda field: particle_mass(field, grid))(saved)
    state_currents = jnp.concatenate((protocol.current[:1], protocol.current))[save]
    saved_summed = jnp.asarray(summed_current)[save]
    zeros = jnp.zeros((len(save),), dtype=jnp.float64)
    return SimulationResult(
        concentration=saved,
        times=protocol.times[save],
        currents=state_currents,
        mass=mass,
        free_energy=zeros,
        overpotential=zeros,
        summed_current=saved_summed,
        cg_residual=zeros,
        metadata={"model": "baseline", "steps": int(protocol.current.size), "dt": protocol.dt},
    )


def simulate_fickian(
    grid: Grid,
    protocol: Protocol,
    diffusivity: float,
    solver: SolverConfig,
    initial,
) -> SimulationResult:
    """Run a linear diffusion control with the same imposed total current."""

    if diffusivity <= 0.0 or not np.isclose(protocol.dt, solver.dt):
        raise ValueError("diffusivity must be positive and protocol dt must match solver")
    initial_field = jnp.where(grid.mask, jnp.asarray(initial, dtype=jnp.float64), 0.0)
    boundary_distribution = grid.boundary_weight / jnp.sum(grid.boundary_weight)

    def step(concentration, current):
        source = current * boundary_distribution / grid.cell_area
        right_hand_side = jnp.where(
            grid.mask, concentration + solver.dt * source, 0.0
        )

        def linear_operator(value):
            active_value = jnp.where(grid.mask, value, 0.0)
            return jnp.where(
                grid.mask,
                active_value
                - solver.dt * diffusivity * masked_laplacian(active_value, grid),
                value,
            )

        next_concentration, _ = cg(
            linear_operator,
            right_hand_side,
            x0=concentration,
            tol=solver.cg_tolerance,
            maxiter=solver.cg_max_iterations,
        )
        next_concentration = jnp.where(grid.mask, next_concentration, 0.0)
        mass_mode_correction = (
            jnp.sum(jnp.where(grid.mask, right_hand_side - next_concentration, 0.0))
            / grid.active_count
        )
        next_concentration = jnp.where(
            grid.mask,
            next_concentration + mass_mode_correction,
            0.0,
        )
        return next_concentration, next_concentration

    _, history = jax.lax.scan(step, initial_field, protocol.current)
    all_concentrations = jnp.concatenate((initial_field[None, ...], history), axis=0)
    summed = jnp.concatenate((jnp.zeros((1,)), protocol.current))
    return _assemble_result(all_concentrations, grid, protocol, summed)


def simulate_sharp_interface(
    grid: Grid,
    protocol: Protocol,
    initial_mean: float,
    stage2: float = 0.5,
    stage1: float = 1.0,
) -> SimulationResult:
    """Construct a conserved stage-1 core whose area follows integrated charge."""

    if not stage2 <= initial_mean <= stage1:
        raise ValueError("initial_mean must lie between the stage values")
    active = np.asarray(grid.mask)
    active_flat = np.flatnonzero(active.ravel())
    radius = np.sqrt(np.asarray(grid.x) ** 2 + np.asarray(grid.y) ** 2).ravel()
    radial_order = active_flat[np.argsort(radius[active_flat], kind="stable")]
    cumulative_charge = np.concatenate(
        ([0.0], np.cumsum(np.asarray(protocol.current) * protocol.dt))
    )
    particle_area = grid.active_count * grid.cell_area
    means = initial_mean + cumulative_charge / particle_area
    if np.any(means < stage2 - 1e-12) or np.any(means > stage1 + 1e-12):
        raise ValueError("protocol drives the sharp-interface mean outside stage bounds")
    fields = np.zeros((len(means),) + active.shape, dtype=np.float64)
    width = stage1 - stage2
    for time_index, mean in enumerate(means):
        flat = fields[time_index].ravel()
        flat[active_flat] = stage2
        high_cell_units = np.clip((mean - stage2) / width * grid.active_count, 0.0, grid.active_count)
        full_cells = min(int(np.floor(high_cell_units)), grid.active_count)
        flat[radial_order[:full_cells]] = stage1
        if full_cells < grid.active_count:
            fraction = high_cell_units - full_cells
            flat[radial_order[full_cells]] = stage2 + fraction * width
    summed = np.concatenate(([0.0], np.asarray(protocol.current)))
    return _assemble_result(fields, grid, protocol, summed)
