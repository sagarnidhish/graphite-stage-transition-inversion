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
from .protocols import Protocol, _step_save_slots
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

    save = protocol.save_indices
    save_count = int(save.size)
    save_slots = _step_save_slots(save, int(protocol.current.size))
    initial_mass = particle_mass(initial, grid)
    initial_energy = total_free_energy(
        initial,
        grid,
        parameters.barrier,
        parameters.kappa,
        parameters.stage2,
        parameters.stage1,
    )
    saved_concentration = jnp.zeros(
        (save_count, *initial.shape), dtype=initial.dtype
    ).at[0].set(initial)
    saved_mass = jnp.zeros((save_count,), dtype=jnp.float64).at[0].set(initial_mass)
    saved_energy = jnp.zeros((save_count,), dtype=jnp.float64).at[0].set(initial_energy)
    saved_overpotential = jnp.zeros((save_count,), dtype=jnp.float64)
    saved_summed_current = jnp.zeros((save_count,), dtype=jnp.float64)
    saved_residual = jnp.zeros((save_count,), dtype=jnp.float64)

    def scan_step(carry, step_inputs):
        (
            concentration,
            concentration_buffer,
            mass_buffer,
            energy_buffer,
            overpotential_buffer,
            summed_current_buffer,
            residual_buffer,
        ) = carry
        target_current, save_slot = step_inputs
        next_concentration, diagnostics = semi_implicit_step(
            concentration, target_current, grid, parameters, solver
        )

        def save_outputs(buffers):
            (
                concentrations,
                masses,
                energies,
                overpotentials,
                summed_currents,
                residuals,
            ) = buffers
            mass = particle_mass(next_concentration, grid)
            energy = total_free_energy(
                next_concentration,
                grid,
                parameters.barrier,
                parameters.kappa,
                parameters.stage2,
                parameters.stage1,
            )
            return (
                concentrations.at[save_slot].set(next_concentration),
                masses.at[save_slot].set(mass),
                energies.at[save_slot].set(energy),
                overpotentials.at[save_slot].set(diagnostics.overpotential),
                summed_currents.at[save_slot].set(diagnostics.summed_current),
                residuals.at[save_slot].set(diagnostics.cg_residual),
            )

        buffers = jax.lax.cond(
            save_slot >= 0,
            save_outputs,
            lambda current_buffers: current_buffers,
            (
                concentration_buffer,
                mass_buffer,
                energy_buffer,
                overpotential_buffer,
                summed_current_buffer,
                residual_buffer,
            ),
        )
        return (next_concentration, *buffers), None

    initial_carry = (
        initial,
        saved_concentration,
        saved_mass,
        saved_energy,
        saved_overpotential,
        saved_summed_current,
        saved_residual,
    )
    final_carry, _ = jax.lax.scan(
        jax.checkpoint(scan_step),
        initial_carry,
        (protocol.current, save_slots),
    )
    (
        _,
        saved_concentration,
        saved_mass,
        saved_energy,
        saved_overpotential,
        saved_summed_current,
        saved_residual,
    ) = final_carry
    state_currents = protocol.current[jnp.maximum(save - 1, 0)]
    return SimulationResult(
        concentration=saved_concentration,
        times=protocol.times[save],
        currents=state_currents,
        mass=saved_mass,
        free_energy=saved_energy,
        overpotential=saved_overpotential,
        summed_current=saved_summed_current,
        cg_residual=saved_residual,
        metadata={"seed": run_seed, "steps": int(protocol.current.size), "dt": solver.dt},
    )
