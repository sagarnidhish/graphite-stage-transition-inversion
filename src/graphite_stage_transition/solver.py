"""Differentiable semi-implicit Cahn-Hilliard-reaction solver."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from .config import SolverConfig
from .free_energy import homogeneous_mu, total_free_energy
from .geometry import Grid
from .operators import masked_laplacian
from .protocols import Protocol
from .reaction import ReactionState, galvanostatic_reaction


class CHRParameters(NamedTuple):
    mobility: float
    barrier: float
    kappa: float
    reaction_rate: float
    stage2: float = 0.5
    stage1: float = 1.0

    def as_array(self) -> jax.Array:
        return jnp.asarray(
            (self.mobility, self.barrier, self.kappa, self.reaction_rate),
            dtype=jnp.float64,
        )


class StepDiagnostics(NamedTuple):
    overpotential: jax.Array
    summed_current: jax.Array
    cg_residual: jax.Array


class SimulationResult(NamedTuple):
    concentration: jax.Array
    times: jax.Array
    currents: jax.Array
    mass: jax.Array
    free_energy: jax.Array
    overpotential: jax.Array
    summed_current: jax.Array
    cg_residual: jax.Array
    metadata: dict[str, Any]


def particle_mass(concentration: jax.Array, grid: Grid) -> jax.Array:
    return jnp.sum(jnp.where(grid.mask, concentration, 0.0)) * grid.cell_area


def full_chemical_potential(
    concentration: jax.Array,
    grid: Grid,
    parameters: CHRParameters,
) -> jax.Array:
    homogeneous = homogeneous_mu(
        concentration,
        parameters.barrier,
        parameters.stage2,
        parameters.stage1,
    )
    return jnp.where(
        grid.mask,
        homogeneous - parameters.kappa * masked_laplacian(concentration, grid),
        0.0,
    )


def semi_implicit_step(
    concentration: jax.Array,
    target_current: jax.Array,
    grid: Grid,
    parameters: CHRParameters,
    solver: SolverConfig,
) -> tuple[jax.Array, StepDiagnostics]:
    """Advance one fixed step with the stiff biharmonic term implicit."""

    chemical = full_chemical_potential(concentration, grid, parameters)
    reaction: ReactionState = galvanostatic_reaction(
        concentration,
        chemical,
        grid.boundary_weight,
        target_current,
        parameters.reaction_rate,
        parameters.stage2,
        parameters.stage1,
    )
    source = reaction.rate * grid.boundary_weight / grid.cell_area
    # The maximum positive curvature of the scaled quartic occurs at its stage
    # minima. Adding and subtracting this linear term makes the fixed-step scheme
    # stable without changing the underlying chemical potential.
    width = parameters.stage1 - parameters.stage2
    stabilization = 32.0 * parameters.barrier / width**2
    explicit_chemical = (
        homogeneous_mu(
            concentration,
            parameters.barrier,
            parameters.stage2,
            parameters.stage1,
        )
        - stabilization * concentration
    )
    explicit_bulk = parameters.mobility * masked_laplacian(explicit_chemical, grid)
    right_hand_side = jnp.where(
        grid.mask,
        concentration + solver.dt * (explicit_bulk + source),
        0.0,
    )

    gradient_coefficient = solver.dt * parameters.mobility * parameters.kappa
    stabilization_coefficient = solver.dt * parameters.mobility * stabilization

    def linear_operator(value):
        active_value = jnp.where(grid.mask, value, 0.0)
        laplacian = masked_laplacian(active_value, grid)
        biharmonic = masked_laplacian(masked_laplacian(active_value, grid), grid)
        return jnp.where(
            grid.mask,
            active_value - stabilization_coefficient * laplacian + gradient_coefficient * biharmonic,
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
    # The masked Laplacian and biharmonic terms annihilate constants, so the
    # sum of the linear system is an exact conserved mode. Iterative CG error
    # can otherwise accumulate in that mode over tens of thousands of steps.
    mass_mode_correction = (
        jnp.sum(jnp.where(grid.mask, right_hand_side - next_concentration, 0.0))
        / grid.active_count
    )
    next_concentration = jnp.where(
        grid.mask,
        next_concentration + mass_mode_correction,
        0.0,
    )
    residual = jnp.linalg.norm(linear_operator(next_concentration) - right_hand_side)
    diagnostics = StepDiagnostics(reaction.overpotential, reaction.summed_current, residual)
    return next_concentration, diagnostics


def _prepare_initial_concentration(
    grid: Grid,
    parameters: CHRParameters,
    solver: SolverConfig,
    initial_concentration,
    seed: int,
) -> jax.Array:
    if initial_concentration is None:
        initial = jnp.where(grid.mask, parameters.stage2, 0.0)
    else:
        values = jnp.asarray(initial_concentration, dtype=jnp.float64)
        initial = jnp.where(grid.mask, values, 0.0)
    if solver.perturbation_amplitude == 0.0:
        return initial
    noise = jax.random.normal(jax.random.key(seed), grid.mask.shape)
    masked_noise = jnp.where(grid.mask, noise, 0.0)
    masked_mean = jnp.sum(masked_noise) / grid.active_count
    perturbation = jnp.where(grid.mask, masked_noise - masked_mean, 0.0)
    return initial + solver.perturbation_amplitude * perturbation


def simulate(
    grid: Grid,
    protocol: Protocol,
    parameters: CHRParameters,
    solver: SolverConfig,
    initial_concentration=None,
    seed: int | None = None,
) -> SimulationResult:
    """Run a complete fixed-step protocol and return requested saved states."""

    run_seed = solver.seed if seed is None else int(seed)
    initial = _prepare_initial_concentration(
        grid, parameters, solver, initial_concentration, run_seed
    )

    def scan_step(concentration, target_current):
        next_concentration, diagnostics = semi_implicit_step(
            concentration, target_current, grid, parameters, solver
        )
        mass = particle_mass(next_concentration, grid)
        energy = total_free_energy(
            next_concentration,
            grid,
            parameters.barrier,
            parameters.kappa,
            parameters.stage2,
            parameters.stage1,
        )
        outputs = (
            next_concentration,
            mass,
            energy,
            diagnostics.overpotential,
            diagnostics.summed_current,
            diagnostics.cg_residual,
        )
        return next_concentration, outputs

    _, history = jax.lax.scan(jax.checkpoint(scan_step), initial, protocol.current)
    concentrations, mass, energy, overpotential, summed_current, residual = history
    all_concentrations = jnp.concatenate((initial[None, ...], concentrations), axis=0)
    all_mass = jnp.concatenate((particle_mass(initial, grid)[None], mass))
    initial_energy = total_free_energy(
        initial,
        grid,
        parameters.barrier,
        parameters.kappa,
        parameters.stage2,
        parameters.stage1,
    )
    all_energy = jnp.concatenate((initial_energy[None], energy))
    all_overpotential = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), overpotential))
    all_summed_current = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), summed_current))
    all_residual = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), residual))
    state_currents = jnp.concatenate((protocol.current[:1], protocol.current))
    save = protocol.save_indices
    return SimulationResult(
        concentration=all_concentrations[save],
        times=protocol.times[save],
        currents=state_currents[save],
        mass=all_mass[save],
        free_energy=all_energy[save],
        overpotential=all_overpotential[save],
        summed_current=all_summed_current[save],
        cg_residual=all_residual[save],
        metadata={"seed": run_seed, "steps": int(protocol.current.size), "dt": solver.dt},
    )
