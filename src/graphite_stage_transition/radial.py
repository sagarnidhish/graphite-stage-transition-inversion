"""Volume-weighted radial finite-volume CHR reference solver."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from .config import SolverConfig
from .free_energy import homogeneous_free_energy, homogeneous_mu
from .geometry import Grid
from .protocols import Protocol, _step_save_slots
from .reaction import galvanostatic_reaction
from .solver import CHRParameters, SimulationResult, StepDiagnostics
from .verification import FullCycleGate, RefinementGate


def diffuse_interface_width_10_90(parameters: CHRParameters) -> float:
    """Return the planar equilibrium 10-90 interface width of the quartic model."""
    width = float(parameters.stage1 - parameters.stage2)
    if parameters.barrier <= 0.0 or parameters.kappa <= 0.0 or width <= 0.0:
        raise ValueError("barrier, kappa, and stage width must be positive")
    scale = np.sqrt(parameters.kappa * width**2 / (32.0 * parameters.barrier))
    return float(2.0 * np.log(9.0) * scale)


def verify_radial_full_cycle(
    concentration,
    grid: RadialGrid,
    stage2: float,
    stage1: float,
    endpoint_tolerance: float = 1e-8,
    bound_tolerance: float = 0.05,
) -> FullCycleGate:
    values = np.asarray(concentration, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != grid.cells or not stage2 < stage1:
        raise ValueError("invalid radial full-cycle inputs")
    finite = bool(np.all(np.isfinite(values)))
    if finite:
        volumes = np.asarray(grid.volumes)
        means = values @ volumes / grid.total_area
        initial_mean = float(means[0])
        maximum_mean = float(np.max(means))
        final_mean = float(means[-1])
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        passed = (
            abs(initial_mean - stage2) <= endpoint_tolerance
            and abs(maximum_mean - stage1) <= endpoint_tolerance
            and abs(final_mean - stage2) <= endpoint_tolerance
            and minimum >= stage2 - bound_tolerance
            and maximum <= stage1 + bound_tolerance
        )
    else:
        initial_mean = maximum_mean = final_mean = minimum = maximum = float("nan")
        passed = False
    return FullCycleGate(
        bool(passed), finite, initial_mean, maximum_mean, final_mean,
        minimum, maximum, float(endpoint_tolerance), float(bound_tolerance)
    )


def _radial_front_positions(concentration, grid: RadialGrid, threshold: float) -> np.ndarray:
    values = np.asarray(concentration, dtype=np.float64)
    centers = np.asarray(grid.centers)
    positions = np.full(values.shape[0], np.nan, dtype=np.float64)
    for frame_index, frame in enumerate(values):
        shifted = frame - threshold
        candidates = np.flatnonzero(shifted[:-1] * shifted[1:] <= 0.0)
        candidates = candidates[np.abs(np.diff(frame)[candidates]) > 1e-12]
        if candidates.size == 0:
            continue
        index = int(candidates[np.argmax(np.abs(np.diff(frame)[candidates]))])
        fraction = (threshold - frame[index]) / (frame[index + 1] - frame[index])
        positions[frame_index] = centers[index] + fraction * (centers[index + 1] - centers[index])
    return positions


def verify_radial_refinement(
    trajectories: dict[int, tuple[SimulationResult, RadialGrid]],
    threshold: float = 0.75,
    tolerance_pixels: float = 1.0,
) -> RefinementGate:
    sizes = tuple(sorted(trajectories))
    if len(sizes) < 2:
        raise ValueError("at least two radial refinements are required")
    finest_result, finest_grid = trajectories[sizes[-1]]
    finest_positions = _radial_front_positions(
        finest_result.concentration, finest_grid, threshold
    )
    displacements: list[float] = []
    for size in sizes[:-1]:
        result, grid = trajectories[size]
        if not np.array_equal(np.asarray(result.times), np.asarray(finest_result.times)):
            raise ValueError("radial refinement trajectories must share matched times")
        positions = _radial_front_positions(result.concentration, grid, threshold)
        matched = np.isfinite(positions) & np.isfinite(finest_positions)
        if not np.any(matched):
            raise ValueError("radial refinement requires at least one matched front")
        displacements.append(
            float(np.max(np.abs(positions[matched] - finest_positions[matched])) / grid.dr)
        )
    maximum = max(displacements)
    return RefinementGate(
        passed=bool(maximum <= tolerance_pixels),
        max_displacement_pixels=maximum,
        tolerance_pixels=float(tolerance_pixels),
        grid_sizes=sizes,
    )


class RadialGrid(NamedTuple):
    centers: jax.Array
    volumes: jax.Array
    face_conductance: jax.Array
    boundary_weight: jax.Array
    dr: float
    radius: float
    cells: int
    total_area: float


def make_radial_grid(cells: int, radius: float) -> RadialGrid:
    """Build concentric annular finite volumes for a two-dimensional disk."""
    if cells < 4 or radius <= 0.0:
        raise ValueError("cells must be at least four and radius must be positive")
    dr = float(radius / cells)
    faces = np.linspace(0.0, radius, cells + 1, dtype=np.float64)
    centers = 0.5 * (faces[:-1] + faces[1:])
    volumes = np.pi * (faces[1:] ** 2 - faces[:-1] ** 2)
    conductance = 2.0 * np.pi * faces[1:-1] / dr
    boundary = np.zeros(cells, dtype=np.float64)
    boundary[-1] = 2.0 * np.pi * radius
    return RadialGrid(
        centers=jnp.asarray(centers),
        volumes=jnp.asarray(volumes),
        face_conductance=jnp.asarray(conductance),
        boundary_weight=jnp.asarray(boundary),
        dr=dr,
        radius=float(radius),
        cells=int(cells),
        total_area=float(np.pi * radius**2),
    )


@jax.jit
def radial_stiffness(field: jax.Array, grid: RadialGrid) -> jax.Array:
    """Return integrated pairwise radial flux for each annular cell."""
    flux = grid.face_conductance * (field[1:] - field[:-1])
    output = jnp.zeros_like(field)
    output = output.at[:-1].add(flux)
    output = output.at[1:].add(-flux)
    return output


@jax.jit
def radial_laplacian(field: jax.Array, grid: RadialGrid) -> jax.Array:
    return radial_stiffness(field, grid) / grid.volumes


@jax.jit
def radial_total_free_energy(
    concentration: jax.Array,
    grid: RadialGrid,
    barrier: float,
    kappa: float,
    stage2: float = 0.5,
    stage1: float = 1.0,
) -> jax.Array:
    homogeneous = jnp.sum(
        grid.volumes
        * homogeneous_free_energy(concentration, barrier, stage2, stage1)
    )
    differences = concentration[1:] - concentration[:-1]
    gradient = 0.5 * kappa * jnp.sum(grid.face_conductance * differences**2)
    return homogeneous + gradient


@jax.jit
def radial_chemical_potential(
    concentration: jax.Array,
    grid: RadialGrid,
    barrier: float,
    kappa: float,
    stage2: float = 0.5,
    stage1: float = 1.0,
) -> jax.Array:
    return homogeneous_mu(concentration, barrier, stage2, stage1) - kappa * radial_laplacian(
        concentration, grid
    )


def radial_particle_mass(concentration: jax.Array, grid: RadialGrid) -> jax.Array:
    return jnp.sum(grid.volumes * concentration)


def radial_semi_implicit_step(
    concentration: jax.Array,
    target_current: jax.Array,
    grid: RadialGrid,
    parameters: CHRParameters,
    solver: SolverConfig,
) -> tuple[jax.Array, StepDiagnostics]:
    chemical = radial_chemical_potential(
        concentration,
        grid,
        parameters.barrier,
        parameters.kappa,
        parameters.stage2,
        parameters.stage1,
    )
    reaction = galvanostatic_reaction(
        concentration,
        chemical,
        grid.boundary_weight,
        target_current,
        parameters.reaction_rate,
        parameters.stage2,
        parameters.stage1,
    )
    source = reaction.rate * grid.boundary_weight / grid.volumes
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
    explicit_bulk = parameters.mobility * radial_laplacian(explicit_chemical, grid)
    right_hand_side = concentration + solver.dt * (explicit_bulk + source)
    sqrt_volume = jnp.sqrt(grid.volumes)

    def symmetric_laplacian(value):
        return sqrt_volume * radial_laplacian(value / sqrt_volume, grid)

    gradient_coefficient = solver.dt * parameters.mobility * parameters.kappa
    stabilization_coefficient = solver.dt * parameters.mobility * stabilization

    def linear_operator(value):
        laplacian = symmetric_laplacian(value)
        biharmonic = symmetric_laplacian(laplacian)
        return value - stabilization_coefficient * laplacian + gradient_coefficient * biharmonic

    transformed_rhs = sqrt_volume * right_hand_side
    transformed, _ = cg(
        linear_operator,
        transformed_rhs,
        x0=sqrt_volume * concentration,
        tol=solver.cg_tolerance,
        maxiter=solver.cg_max_iterations,
    )
    next_concentration = transformed / sqrt_volume
    correction = (
        jnp.sum(grid.volumes * (right_hand_side - next_concentration))
        / grid.total_area
    )
    next_concentration = next_concentration + correction
    residual = jnp.linalg.norm(linear_operator(sqrt_volume * next_concentration) - transformed_rhs)
    return next_concentration, StepDiagnostics(
        reaction.overpotential, reaction.summed_current, residual
    )


def _radial_initial(
    grid: RadialGrid,
    parameters: CHRParameters,
    solver: SolverConfig,
    initial_concentration,
    seed: int,
) -> jax.Array:
    if initial_concentration is None:
        initial = jnp.full((grid.cells,), parameters.stage2, dtype=jnp.float64)
    else:
        initial = jnp.asarray(initial_concentration, dtype=jnp.float64)
        if initial.shape != (grid.cells,):
            raise ValueError("radial initial concentration must match grid cells")
    if solver.perturbation_amplitude == 0.0:
        return initial
    noise = jax.random.normal(jax.random.key(seed), (grid.cells,))
    weighted_mean = jnp.sum(grid.volumes * noise) / grid.total_area
    return initial + solver.perturbation_amplitude * (noise - weighted_mean)


def simulate_radial(
    grid: RadialGrid,
    protocol: Protocol,
    parameters: CHRParameters,
    solver: SolverConfig,
    initial_concentration=None,
    seed: int | None = None,
) -> SimulationResult:
    """Run a differentiable radial CHR trajectory with sparse saved states."""
    run_seed = solver.seed if seed is None else int(seed)
    initial = _radial_initial(grid, parameters, solver, initial_concentration, run_seed)
    save = protocol.save_indices
    save_count = int(save.size)
    save_slots = _step_save_slots(save, int(protocol.current.size))
    saved_concentration = jnp.zeros((save_count, grid.cells), dtype=jnp.float64).at[0].set(initial)
    saved_mass = jnp.zeros((save_count,), dtype=jnp.float64).at[0].set(radial_particle_mass(initial, grid))
    saved_energy = jnp.zeros((save_count,), dtype=jnp.float64).at[0].set(
        radial_total_free_energy(initial, grid, parameters.barrier, parameters.kappa, parameters.stage2, parameters.stage1)
    )
    zeros = jnp.zeros((save_count,), dtype=jnp.float64)

    def scan_step(carry, inputs):
        concentration, concentrations, masses, energies, overpotentials, currents, residuals = carry
        target_current, save_slot = inputs
        next_concentration, diagnostics = radial_semi_implicit_step(
            concentration, target_current, grid, parameters, solver
        )

        def save_outputs(buffers):
            cs, ms, es, os, ins, rs = buffers
            return (
                cs.at[save_slot].set(next_concentration),
                ms.at[save_slot].set(radial_particle_mass(next_concentration, grid)),
                es.at[save_slot].set(radial_total_free_energy(next_concentration, grid, parameters.barrier, parameters.kappa, parameters.stage2, parameters.stage1)),
                os.at[save_slot].set(diagnostics.overpotential),
                ins.at[save_slot].set(diagnostics.summed_current),
                rs.at[save_slot].set(diagnostics.cg_residual),
            )

        buffers = jax.lax.cond(
            save_slot >= 0,
            save_outputs,
            lambda values: values,
            (concentrations, masses, energies, overpotentials, currents, residuals),
        )
        return (next_concentration, *buffers), None

    final, _ = jax.lax.scan(
        jax.checkpoint(scan_step),
        (initial, saved_concentration, saved_mass, saved_energy, zeros, zeros, zeros),
        (protocol.current, save_slots),
    )
    _, concentrations, masses, energies, overpotentials, currents, residuals = final
    state_currents = protocol.current[jnp.maximum(save - 1, 0)]
    return SimulationResult(
        concentration=concentrations,
        times=protocol.times[save],
        currents=state_currents,
        mass=masses,
        free_energy=energies,
        overpotential=overpotentials,
        summed_current=currents,
        cg_residual=residuals,
        metadata={"backend": "radial_finite_volume", "seed": run_seed, "steps": int(protocol.current.size), "dt": solver.dt},
    )


def rasterize_radial(
    concentration: jax.Array,
    radial_grid: RadialGrid,
    cartesian_grid: Grid,
) -> jax.Array:
    """Interpolate radial cell values onto a masked Cartesian image grid."""
    values = jnp.asarray(concentration, dtype=jnp.float64)
    radius = jnp.sqrt(cartesian_grid.x**2 + cartesian_grid.y**2)

    def rasterize_frame(frame):
        image = jnp.interp(radius.reshape(-1), radial_grid.centers, frame).reshape(radius.shape)
        return jnp.where(cartesian_grid.mask, image, 0.0)

    if values.ndim == 1:
        if values.shape != (radial_grid.cells,):
            raise ValueError("radial concentration must match radial grid cells")
        return rasterize_frame(values)
    if values.ndim == 2 and values.shape[1] == radial_grid.cells:
        return jax.vmap(rasterize_frame)(values)
    raise ValueError("radial concentration must have shape (cells,) or (time, cells)")


def rasterize_radial_result(
    result: SimulationResult,
    radial_grid: RadialGrid,
    cartesian_grid: Grid,
) -> SimulationResult:
    return result._replace(
        concentration=rasterize_radial(result.concentration, radial_grid, cartesian_grid),
        metadata={**result.metadata, "rasterized_shape": tuple(cartesian_grid.mask.shape)},
    )
