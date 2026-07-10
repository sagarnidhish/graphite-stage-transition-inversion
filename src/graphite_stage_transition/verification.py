"""Quantitative verification gates for the effective CHR forward model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from .config import GridConfig, ProjectConfig, SolverConfig
from .geometry import Grid, make_circle_grid
from .protocols import build_protocol, make_constant_protocol
from .solver import CHRParameters, SimulationResult, simulate


@dataclass(frozen=True)
class MassBalanceGate:
    passed: bool
    relative_error: float
    max_absolute_error: float
    tolerance: float


@dataclass(frozen=True)
class RelaxationGate:
    passed: bool
    max_energy_increase: float
    tolerance: float


@dataclass(frozen=True)
class RefinementGate:
    passed: bool
    max_displacement_pixels: float
    tolerance_pixels: float
    grid_sizes: tuple[int, ...]


@dataclass(frozen=True)
class IsotropyGate:
    passed: bool
    maximum_angular_rms: float
    maximum_angular_deviation: float
    tolerance: float
    radial_bins: int
    angular_sectors: int


@dataclass(frozen=True)
class RotationEquivarianceGate:
    passed: bool
    maximum_absolute_difference: float
    tolerance: float
    quarter_turns: int


@dataclass(frozen=True)
class DeterminismGate:
    passed: bool
    max_absolute_difference: float


@dataclass(frozen=True)
class FullCycleGate:
    passed: bool
    finite: bool
    initial_mean: float
    maximum_mean: float
    final_mean: float
    minimum_concentration: float
    maximum_concentration: float
    endpoint_tolerance: float
    bound_tolerance: float


@dataclass(frozen=True)
class VerificationReport:
    mass_balance: MassBalanceGate
    relaxation: RelaxationGate
    refinement: RefinementGate
    determinism: DeterminismGate

    @property
    def passed(self) -> bool:
        return all(
            gate.passed
            for gate in (
                self.mass_balance,
                self.relaxation,
                self.refinement,
                self.determinism,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passed"] = self.passed
        return data


def verify_mass_balance(
    times,
    currents,
    mass,
    tolerance: float = 1e-7,
) -> MassBalanceGate:
    """Compare particle mass change with state-aligned current integration.

    Solver state ``k`` contains the result after applying ``currents[k]`` over
    the preceding interval, so the integral uses right endpoint values.
    """

    time_values = np.asarray(times, dtype=np.float64)
    current_values = np.asarray(currents, dtype=np.float64)
    mass_values = np.asarray(mass, dtype=np.float64)
    if not (time_values.ndim == current_values.ndim == mass_values.ndim == 1):
        raise ValueError("times, currents, and mass must be one-dimensional")
    if not (len(time_values) == len(current_values) == len(mass_values)):
        raise ValueError("times, currents, and mass must have equal lengths")
    if len(time_values) < 2 or np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must contain at least two increasing values")

    expected = np.concatenate(
        ([0.0], np.cumsum(current_values[1:] * np.diff(time_values)))
    )
    measured = mass_values - mass_values[0]
    absolute_error = np.abs(measured - expected)
    scale = max(float(np.max(np.abs(expected))), float(np.max(np.abs(measured))), 1e-15)
    relative_error = float(np.max(absolute_error) / scale)
    return MassBalanceGate(
        passed=bool(relative_error <= tolerance),
        relative_error=relative_error,
        max_absolute_error=float(np.max(absolute_error)),
        tolerance=float(tolerance),
    )


def verify_relaxation(
    free_energy,
    tolerance: float = 1e-9,
) -> RelaxationGate:
    """Require free energy not to increase during a zero-current trajectory."""

    values = np.asarray(free_energy, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("free_energy must be a finite one-dimensional trajectory")
    maximum_increase = float(max(0.0, np.max(np.diff(values))))
    return RelaxationGate(
        passed=bool(maximum_increase <= tolerance),
        max_energy_increase=maximum_increase,
        tolerance=float(tolerance),
    )


def verify_determinism(first, second) -> DeterminismGate:
    """Require bitwise-identical concentration arrays for a repeated run."""

    first_values = np.asarray(first)
    second_values = np.asarray(second)
    if first_values.shape != second_values.shape:
        return DeterminismGate(False, float("inf"))
    difference = np.abs(first_values - second_values)
    maximum = float(np.max(difference)) if difference.size else 0.0
    return DeterminismGate(bool(np.array_equal(first_values, second_values)), maximum)


def verify_full_cycle_transition(
    concentration,
    mask,
    stage2: float,
    stage1: float,
    endpoint_tolerance: float = 1e-8,
    bound_tolerance: float = 0.05,
) -> FullCycleGate:
    """Require stage-2 start, stage-1 reach, stage-2 return, and bounded fields."""

    values = np.asarray(concentration, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    if (
        values.ndim != 3
        or values.shape[1:] != active.shape
        or values.shape[0] < 2
        or not np.any(active)
        or not stage2 < stage1
    ):
        raise ValueError("invalid full-cycle transition inputs")
    active_values = values[:, active]
    finite = bool(np.all(np.isfinite(active_values)))
    if finite:
        means = np.mean(active_values, axis=1)
        initial_mean = float(means[0])
        maximum_mean = float(np.max(means))
        final_mean = float(means[-1])
        minimum = float(np.min(active_values))
        maximum = float(np.max(active_values))
        passed = (
            abs(initial_mean - stage2) <= endpoint_tolerance
            and abs(maximum_mean - stage1) <= endpoint_tolerance
            and abs(final_mean - stage2) <= endpoint_tolerance
            and minimum >= stage2 - bound_tolerance
            and maximum <= stage1 + bound_tolerance
        )
    else:
        initial_mean = maximum_mean = final_mean = float("nan")
        minimum = maximum = float("nan")
        passed = False
    return FullCycleGate(
        bool(passed),
        finite,
        initial_mean,
        maximum_mean,
        final_mean,
        minimum,
        maximum,
        float(endpoint_tolerance),
        float(bound_tolerance),
    )


def _front_radius(concentration: np.ndarray, grid: Grid, threshold: float) -> np.ndarray:
    active_high = (concentration >= threshold) & np.asarray(grid.mask)[None, ...]
    high_area = active_high.sum(axis=(1, 2)) * grid.cell_area
    return np.sqrt(high_area / np.pi)


def verify_refinement(
    trajectories: dict[int, tuple[SimulationResult, Grid]],
    tolerance_pixels: float = 1.0,
) -> RefinementGate:
    """Compare an equivalent-area front radius against the finest grid."""

    sizes = tuple(sorted(trajectories))
    if len(sizes) < 2:
        raise ValueError("at least two grid refinements are required")
    finest_result, finest_grid = trajectories[sizes[-1]]
    threshold = 0.75
    finest_radius = _front_radius(np.asarray(finest_result.concentration), finest_grid, threshold)
    displacements: list[float] = []
    for size in sizes[:-1]:
        result, grid = trajectories[size]
        if not np.array_equal(np.asarray(result.times), np.asarray(finest_result.times)):
            raise ValueError("refinement trajectories must share matched times")
        radius = _front_radius(np.asarray(result.concentration), grid, threshold)
        displacements.append(float(np.max(np.abs(radius - finest_radius)) / grid.dx))
    maximum = max(displacements)
    return RefinementGate(
        passed=bool(maximum <= tolerance_pixels),
        max_displacement_pixels=float(maximum),
        tolerance_pixels=float(tolerance_pixels),
        grid_sizes=sizes,
    )


def verify_isotropy(
    concentration,
    grid: Grid,
    stage2: float,
    stage1: float,
    radial_bins: int = 12,
    angular_sectors: int = 16,
    tolerance: float = 0.05,
) -> IsotropyGate:
    """Measure angular variation after conditioning on radius.

    The score is the largest, over frames, of the RMS angular deviation and
    the largest absolute angular deviation, both normalized by the stage
    concentration span. Empty radius/angle bins are ignored.
    """
    values = np.asarray(concentration, dtype=np.float64)
    mask = np.asarray(grid.mask, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != mask.shape or not np.any(mask):
        raise ValueError("concentration must have shape (time, height, width) matching grid.mask")
    if radial_bins < 1 or angular_sectors < 4 or not stage2 < stage1:
        raise ValueError("invalid isotropy binning or stage bounds")
    y = np.asarray(grid.y, dtype=np.float64)
    x = np.asarray(grid.x, dtype=np.float64)
    radius = np.sqrt(x * x + y * y) / max(float(grid.radius), 1e-15)
    angle = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    radial_index = np.minimum((radius * radial_bins).astype(int), radial_bins - 1)
    angular_index = np.minimum((angle * angular_sectors).astype(int), angular_sectors - 1)
    valid = mask & (radius <= 1.0 + 1e-12)
    span = float(stage1 - stage2)
    rms_scores, max_scores = isotropy_scores(
        values, grid, stage2, stage1, radial_bins, angular_sectors
    )
    maximum_rms = max(rms_scores, default=0.0)
    maximum_deviation = max(max_scores, default=0.0)
    return IsotropyGate(
        passed=bool(maximum_rms <= tolerance and maximum_deviation <= 2.0 * tolerance),
        maximum_angular_rms=float(maximum_rms),
        maximum_angular_deviation=float(maximum_deviation),
        tolerance=float(tolerance),
        radial_bins=int(radial_bins),
        angular_sectors=int(angular_sectors),
    )


def verify_rotation_equivariance(
    reference,
    rotated,
    quarter_turns: int = 1,
    tolerance: float = 1e-10,
) -> RotationEquivarianceGate:
    """Compare a trajectory with the rotated output of a rotated input run."""
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(rotated, dtype=np.float64)
    if first.ndim != 3 or first.shape != second.shape:
        raise ValueError("rotation trajectories must have matching (time, height, width) shape")
    if quarter_turns not in (1, 2, 3) or tolerance < 0.0:
        raise ValueError("quarter_turns must be 1, 2, or 3 and tolerance nonnegative")
    expected = np.rot90(first, k=quarter_turns, axes=(1, 2))
    maximum = float(np.max(np.abs(expected - second)))
    return RotationEquivarianceGate(
        passed=bool(np.isfinite(maximum) and maximum <= tolerance),
        maximum_absolute_difference=maximum,
        tolerance=float(tolerance),
        quarter_turns=int(quarter_turns),
    )


def isotropy_scores(
    concentration,
    grid: Grid,
    stage2: float,
    stage1: float,
    radial_bins: int = 12,
    angular_sectors: int = 16,
) -> tuple[list[float], list[float]]:
    """Return per-frame normalized angular RMS and maximum deviations."""
    values = np.asarray(concentration, dtype=np.float64)
    mask = np.asarray(grid.mask, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != mask.shape or not np.any(mask):
        raise ValueError("concentration must have shape (time, height, width) matching grid.mask")
    if radial_bins < 1 or angular_sectors < 4 or not stage2 < stage1:
        raise ValueError("invalid isotropy binning or stage bounds")
    y = np.asarray(grid.y, dtype=np.float64)
    x = np.asarray(grid.x, dtype=np.float64)
    radius = np.sqrt(x * x + y * y) / max(float(grid.radius), 1e-15)
    angle = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    radial_index = np.minimum((radius * radial_bins).astype(int), radial_bins - 1)
    angular_index = np.minimum((angle * angular_sectors).astype(int), angular_sectors - 1)
    valid = mask & (radius <= 1.0 + 1e-12)
    span = float(stage1 - stage2)
    rms_scores: list[float] = []
    max_scores: list[float] = []
    for frame in values:
        frame_rms: list[float] = []
        frame_max: list[float] = []
        for radial_bin in range(radial_bins):
            radial_cells = valid & (radial_index == radial_bin)
            if not np.any(radial_cells):
                continue
            radial_mean = float(np.mean(frame[radial_cells]))
            deviations = []
            for sector in range(angular_sectors):
                cells = radial_cells & (angular_index == sector)
                if np.any(cells):
                    deviations.append(float(np.mean(frame[cells]) - radial_mean))
            if deviations:
                scaled = np.asarray(deviations) / span
                frame_rms.append(float(np.sqrt(np.mean(scaled * scaled))))
                frame_max.append(float(np.max(np.abs(scaled))))
        rms_scores.append(max(frame_rms, default=0.0))
        max_scores.append(max(frame_max, default=0.0))
    return rms_scores, max_scores


def _parameters(config: ProjectConfig) -> CHRParameters:
    model = config.model
    return CHRParameters(
        model.mobility,
        model.barrier,
        model.kappa,
        model.reaction_rate,
        model.stage2,
        model.stage1,
    )


def _relaxation_run(config: ProjectConfig, grid: Grid, parameters: CHRParameters):
    solver = replace(config.solver, perturbation_amplitude=0.0)
    protocol = make_constant_protocol(0.0, steps=32, dt=solver.dt, save_every=1)
    midpoint = 0.5 * (parameters.stage2 + parameters.stage1)
    perturbation = 0.025 * jnp.cos(2.0 * jnp.pi * grid.x) * jnp.cos(2.0 * jnp.pi * grid.y)
    initial = jnp.where(grid.mask, midpoint + perturbation, 0.0)
    return simulate(grid, protocol, parameters, solver, initial_concentration=initial)


def _refinement_runs(
    config: ProjectConfig,
    parameters: CHRParameters,
    sizes: tuple[int, ...] = (48, 64, 96),
) -> dict[int, tuple[SimulationResult, Grid]]:
    solver: SolverConfig = replace(config.solver, perturbation_amplitude=0.0)
    protocol = make_constant_protocol(0.0, steps=12, dt=solver.dt, save_every=3)
    trajectories: dict[int, tuple[SimulationResult, Grid]] = {}
    interface_width = max(2.0 * np.sqrt(parameters.kappa / parameters.barrier), 0.03)
    front_radius = 0.5 * config.grid.radius
    for size in sizes:
        grid_config = GridConfig(size, size, config.grid.length, config.grid.radius)
        grid = make_circle_grid(grid_config)
        radial_position = jnp.sqrt(grid.x**2 + grid.y**2)
        phase_fraction = 0.5 * (1.0 - jnp.tanh((radial_position - front_radius) / interface_width))
        initial = jnp.where(
            grid.mask,
            parameters.stage2 + (parameters.stage1 - parameters.stage2) * phase_fraction,
            0.0,
        )
        result = simulate(
            grid,
            protocol,
            parameters,
            solver,
            initial_concentration=initial,
            seed=solver.seed,
        )
        trajectories[size] = (result, grid)
    return trajectories


def run_verification_suite(config: ProjectConfig) -> VerificationReport:
    """Run the mandatory conservation, relaxation, refinement, and repeatability gates."""

    grid = make_circle_grid(config.grid)
    parameters = _parameters(config)
    protocol = build_protocol(config.protocol, config.solver.dt)
    first = simulate(grid, protocol, parameters, config.solver)
    second = simulate(grid, protocol, parameters, config.solver)
    relaxation = _relaxation_run(config, grid, parameters)
    refinements = _refinement_runs(config, parameters)
    return VerificationReport(
        mass_balance=verify_mass_balance(first.times, first.currents, first.mass),
        relaxation=verify_relaxation(relaxation.free_energy),
        refinement=verify_refinement(refinements),
        determinism=verify_determinism(first.concentration, second.concentration),
    )
