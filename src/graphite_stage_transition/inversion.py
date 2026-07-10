"""Autodifferentiable direct-field inversion for CHR concentration movies."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .config import SolverConfig
from .geometry import Grid
from .observables import (
    ObservableGeometry,
    PhysicsObservables,
    make_observable_geometry,
    observable_residual_vector,
    physics_observables,
)
from .protocols import Protocol
from .solver import CHRParameters, simulate


class LossComponents(NamedTuple):
    total: jax.Array
    radial: jax.Array
    structure: jax.Array
    boundary: jax.Array
    movie: jax.Array
    mass: jax.Array
    bounds: jax.Array


@dataclass(frozen=True)
class ParameterTransform:
    """Log transform and declared admissible bounds for positive parameters."""

    lower: tuple[float, float, float, float]
    upper: tuple[float, float, float, float]
    stage2: float = 0.5
    stage1: float = 1.0

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)
        if lower.shape != (4,) or upper.shape != (4,):
            raise ValueError("parameter bounds must contain four values")
        if np.any(lower <= 0.0) or np.any(lower >= upper):
            raise ValueError("parameter bounds must be positive and increasing")

    def to_unconstrained(self, parameters: CHRParameters) -> jax.Array:
        return jnp.log(parameters.as_array())

    def from_unconstrained(self, values) -> CHRParameters:
        raw = jnp.exp(jnp.asarray(values, dtype=jnp.float64))
        return CHRParameters(raw[0], raw[1], raw[2], raw[3], self.stage2, self.stage1)

    def scipy_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(zip(np.log(self.lower), np.log(self.upper)))


def dimensionless_groups(parameters: CHRParameters, length: float) -> dict[str, jax.Array]:
    """Return the three primary nondimensional recovery targets."""

    return {
        "epsilon_squared": parameters.kappa / (parameters.barrier * length**2),
        "tau_diffusion": length**2 / (parameters.mobility * parameters.barrier),
        "damkohler": parameters.reaction_rate
        * length
        / (parameters.mobility * parameters.barrier),
    }


@dataclass(frozen=True)
class InverseProblem:
    grid: Grid
    protocol: Protocol
    solver: SolverConfig
    observations: jax.Array
    initial_concentration: jax.Array
    transform: ParameterTransform
    mass_penalty: float = 1.0
    bound_penalty: float = 1e-4
    observable_geometry: ObservableGeometry = field(init=False, repr=False, compare=False)
    observed_observables: PhysicsObservables = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        geometry = make_observable_geometry(self.grid)
        observed = physics_observables(
            self.observations,
            geometry,
            self.transform.stage2,
            self.transform.stage1,
        )
        object.__setattr__(self, "observable_geometry", geometry)
        object.__setattr__(self, "observed_observables", observed)

    def components(self, unconstrained) -> LossComponents:
        return loss_components(self, unconstrained)

    def loss(self, unconstrained) -> jax.Array:
        return self.components(unconstrained).total


def loss_components(problem: InverseProblem, unconstrained) -> LossComponents:
    """Evaluate primary morphology losses and zero-weight field diagnostics."""

    parameters = problem.transform.from_unconstrained(unconstrained)
    prediction = simulate(
        problem.grid,
        problem.protocol,
        parameters,
        problem.solver,
        initial_concentration=problem.initial_concentration,
        seed=problem.solver.seed,
    )
    predicted_observables = physics_observables(
        prediction.concentration,
        problem.observable_geometry,
        problem.transform.stage2,
        problem.transform.stage1,
    )
    residual = observable_residual_vector(
        predicted_observables,
        problem.observed_observables,
    )
    radial_loss = jnp.mean(
        (predicted_observables.radial_profile - problem.observed_observables.radial_profile)
        ** 2
    )
    structure_loss = jnp.mean(
        (predicted_observables.structure_power - problem.observed_observables.structure_power)
        ** 2
    )
    boundary_loss = jnp.mean(
        (predicted_observables.boundary_excess - problem.observed_observables.boundary_excess)
        ** 2
    )
    mask = problem.grid.mask[None, ...]
    width = problem.transform.stage1 - problem.transform.stage2
    difference = jnp.where(mask, prediction.concentration - problem.observations, 0.0)
    normalizer = prediction.concentration.shape[0] * problem.grid.active_count * width**2
    movie_loss = jnp.sum(difference**2) / normalizer

    observed_mass = jnp.sum(
        jnp.where(mask, problem.observations, 0.0), axis=(1, 2)
    ) * problem.grid.cell_area
    mass_scale = problem.grid.active_count * problem.grid.cell_area * width
    mass_loss = jnp.mean(((prediction.mass - observed_mass) / mass_scale) ** 2)

    below = jax.nn.relu(problem.transform.stage2 - prediction.concentration)
    above = jax.nn.relu(prediction.concentration - problem.transform.stage1)
    bounds_loss = jnp.sum(jnp.where(mask, below**2 + above**2, 0.0)) / normalizer
    total = jnp.mean(residual**2) + problem.bound_penalty * bounds_loss
    return LossComponents(
        total,
        radial_loss,
        structure_loss,
        boundary_loss,
        movie_loss,
        mass_loss,
        bounds_loss,
    )


def inverse_residual_vector(problem: InverseProblem, unconstrained) -> jax.Array:
    """Return residuals whose mean square is the complete primary objective."""

    parameters = problem.transform.from_unconstrained(unconstrained)
    prediction = simulate(
        problem.grid,
        problem.protocol,
        parameters,
        problem.solver,
        initial_concentration=problem.initial_concentration,
        seed=problem.solver.seed,
    )
    predicted = physics_observables(
        prediction.concentration,
        problem.observable_geometry,
        problem.transform.stage2,
        problem.transform.stage1,
    )
    morphology = observable_residual_vector(predicted, problem.observed_observables)

    width = problem.transform.stage1 - problem.transform.stage2
    below = jax.nn.relu(problem.transform.stage2 - prediction.concentration)
    above = jax.nn.relu(prediction.concentration - problem.transform.stage1)
    spatial_size = int(np.prod(problem.grid.mask.shape))
    active_indices = jnp.asarray(
        np.flatnonzero(np.asarray(problem.grid.mask).reshape(-1)),
        dtype=jnp.int32,
    )
    bound_violations = jnp.take(
        (below + above).reshape((-1, spatial_size)),
        active_indices,
        axis=1,
    ).reshape(-1) / width

    total_size = morphology.size + bound_violations.size
    morphology = morphology * jnp.sqrt(total_size / morphology.size)
    bound_violations = bound_violations * jnp.sqrt(
        problem.bound_penalty * total_size / bound_violations.size
    )
    return jnp.concatenate((morphology, bound_violations))


def centered_finite_difference(function, values, step: float = 1e-4) -> np.ndarray:
    """Compute a centered numerical gradient for an explicit gradient gate."""

    point = np.asarray(values, dtype=np.float64)
    gradient = np.empty_like(point)
    for index in range(point.size):
        displacement = np.zeros_like(point)
        displacement[index] = step
        forward = float(function(jnp.asarray(point + displacement)))
        backward = float(function(jnp.asarray(point - displacement)))
        gradient[index] = (forward - backward) / (2.0 * step)
    return gradient


def relative_group_error(
    estimate: dict[str, float | jax.Array],
    truth: dict[str, float | jax.Array],
) -> np.ndarray:
    names = ("epsilon_squared", "tau_diffusion", "damkohler")
    return np.asarray(
        [abs(float(estimate[name]) - float(truth[name])) / abs(float(truth[name])) for name in names]
    )


@dataclass(frozen=True)
class FitResult:
    parameters: dict[str, float]
    groups: dict[str, float]
    loss: float
    components: dict[str, float | None]
    gradient_norm: float
    steps: int
    forward_solves: int
    status: str
    success: bool
    runtime_seconds: float
    initial_parameters: dict[str, float]


@dataclass(frozen=True)
class MultistartResult:
    best: FitResult
    starts: tuple[FitResult, ...]


def _parameter_dict(parameters: CHRParameters) -> dict[str, float]:
    names = ("mobility", "barrier", "kappa", "reaction_rate")
    return {name: float(value) for name, value in zip(names, parameters.as_array())}


def fit_single_start(
    problem: InverseProblem,
    initial: CHRParameters,
    maxiter: int,
) -> FitResult:
    """Fit one deterministic L-BFGS-B start and retain failed-run accounting."""

    transformed_initial = np.asarray(problem.transform.to_unconstrained(initial), dtype=np.float64)

    def objective_with_components(values):
        components = problem.components(values)
        return components.total, components

    objective_and_gradient = jax.jit(
        jax.value_and_grad(objective_with_components, has_aux=True)
    )
    evaluations = 0
    last_evaluation = None
    encountered_nonfinite = False

    def scipy_objective(values):
        nonlocal evaluations, last_evaluation, encountered_nonfinite
        (value, components), gradient = objective_and_gradient(jnp.asarray(values))
        evaluations += 1
        value_float = float(value)
        gradient_array = np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(value_float) or not np.all(np.isfinite(gradient_array)):
            encountered_nonfinite = True
            value_float = 1e30
            gradient_array = np.zeros_like(values)
        last_evaluation = (
            np.asarray(values, dtype=np.float64).copy(),
            value_float,
            gradient_array,
            components,
        )
        return value_float, gradient_array

    started = time.perf_counter()
    optimized = minimize(
        scipy_objective,
        transformed_initial,
        method="L-BFGS-B",
        jac=True,
        bounds=problem.transform.scipy_bounds(),
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8, "maxls": 20},
    )
    if last_evaluation is None or not np.array_equal(last_evaluation[0], optimized.x):
        scipy_objective(optimized.x)
    _, final_value, final_gradient, components = last_evaluation
    runtime = time.perf_counter() - started
    parameters = problem.transform.from_unconstrained(optimized.x)
    groups = {
        name: float(value)
        for name, value in dimensionless_groups(
            parameters, problem.grid.dx * problem.grid.mask.shape[0]
        ).items()
    }

    def finite_component(value) -> float | None:
        converted = float(value)
        return converted if np.isfinite(converted) else None

    component_values = {
        "radial": finite_component(components.radial),
        "structure": finite_component(components.structure),
        "boundary": finite_component(components.boundary),
        "movie": finite_component(components.movie),
        "mass": finite_component(components.mass),
        "bounds": finite_component(components.bounds),
    }
    success = bool(
        optimized.success
        and np.isfinite(final_value)
        and not encountered_nonfinite
    )
    if encountered_nonfinite:
        status = "failed: nonfinite objective or gradient evaluation"
    else:
        status = "converged" if success else f"failed: {optimized.message}"
    return FitResult(
        parameters=_parameter_dict(parameters),
        groups=groups,
        loss=float(final_value),
        components=component_values,
        gradient_norm=float(np.linalg.norm(final_gradient)),
        steps=int(optimized.nit),
        forward_solves=evaluations,
        status=status,
        success=success,
        runtime_seconds=float(runtime),
        initial_parameters=_parameter_dict(initial),
    )


def fit_multistart(
    problem: InverseProblem,
    starts: Sequence[CHRParameters],
    maxiter: int,
) -> MultistartResult:
    """Fit every declared start and choose the lowest finite objective."""

    if not starts:
        raise ValueError("at least one inversion start is required")
    results = tuple(fit_single_start(problem, start, maxiter) for start in starts)
    successful = [
        result for result in results if result.success and np.isfinite(result.loss)
    ]
    if not successful:
        raise RuntimeError("all inversion starts failed or returned nonfinite losses")
    return MultistartResult(min(successful, key=lambda result: result.loss), results)


def generate_starts(
    transform: ParameterTransform,
    central: CHRParameters,
    count: int,
    seed: int,
) -> list[CHRParameters]:
    """Generate recorded log-uniform starts while always including the center."""

    if count < 1:
        raise ValueError("count must be positive")
    starts = [central]
    rng = np.random.default_rng(seed)
    lower = np.log(np.asarray(transform.lower))
    upper = np.log(np.asarray(transform.upper))
    for values in rng.uniform(lower, upper, size=(count - 1, 4)):
        starts.append(transform.from_unconstrained(values))
    return starts
