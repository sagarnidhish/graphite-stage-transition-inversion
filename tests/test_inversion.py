from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.config import GridConfig, SolverConfig
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.inversion import (
    InverseProblem,
    LossComponents,
    ParameterTransform,
    centered_finite_difference,
    dimensionless_groups,
    fit_single_start,
    fit_multistart,
    inverse_residual_vector,
    loss_components,
    relative_group_error,
)
from graphite_stage_transition.protocols import make_constant_protocol
from graphite_stage_transition.solver import CHRParameters, simulate


TRANSFORM = ParameterTransform(
    lower=(0.02, 0.15, 0.0002, 0.03),
    upper=(0.5, 2.0, 0.006, 0.8),
    stage2=0.5,
    stage1=1.0,
)


def make_tiny_inverse_problem():
    grid = make_circle_grid(GridConfig(nx=20, ny=20, length=1.0, radius=0.38))
    solver = SolverConfig(
        dt=0.001,
        cg_tolerance=1e-10,
        cg_max_iterations=120,
        perturbation_amplitude=0.0,
        seed=4,
    )
    protocol = make_constant_protocol(current=0.008, steps=8, dt=solver.dt)
    truth = CHRParameters(0.11, 0.65, 0.0012, 0.19, 0.5, 1.0)
    radius = jnp.sqrt(grid.x**2 + grid.y**2)
    initial = jnp.where(
        grid.mask,
        0.63 + 0.055 * jnp.cos(3.0 * jnp.pi * grid.x) * jnp.cos(2.0 * jnp.pi * grid.y)
        + 0.02 * radius,
        0.0,
    )
    observed = simulate(
        grid,
        protocol,
        truth,
        solver,
        initial_concentration=initial,
        seed=4,
    ).concentration
    problem = InverseProblem(
        grid=grid,
        protocol=protocol,
        solver=replace(solver, perturbation_amplitude=0.0),
        observations=observed,
        initial_concentration=initial,
        transform=TRANSFORM,
        mass_penalty=0.0,
        bound_penalty=1e-4,
    )
    near_truth = CHRParameters(0.105, 0.68, 0.00115, 0.20, 0.5, 1.0)
    displaced = CHRParameters(0.065, 1.05, 0.0020, 0.11, 0.5, 1.0)
    truth_groups = dimensionless_groups(truth, length=1.0)
    return problem, truth, truth_groups, near_truth, displaced


def test_parameter_transform_round_trip():
    theta = CHRParameters(0.07, 1.8, 0.002, 0.3, 0.5, 1.0)

    unconstrained = TRANSFORM.to_unconstrained(theta)
    recovered = TRANSFORM.from_unconstrained(unconstrained)

    np.testing.assert_allclose(recovered.as_array(), theta.as_array(), rtol=1e-12)
    assert np.all(np.asarray(recovered.as_array()) > 0.0)


def test_loss_gradient_matches_finite_difference():
    problem, _, _, near_truth, _ = make_tiny_inverse_problem()
    z0 = TRANSFORM.to_unconstrained(near_truth)

    _, gradient = jax.value_and_grad(problem.loss)(z0)
    finite_difference = centered_finite_difference(problem.loss, z0, step=1e-4)

    np.testing.assert_allclose(
        np.asarray(gradient),
        finite_difference,
        rtol=2e-2,
        atol=2e-4,
    )


def test_total_loss_matches_frozen_observable_objective():
    problem, _, _, _, displaced = make_tiny_inverse_problem()

    components = problem.components(TRANSFORM.to_unconstrained(displaced))

    expected = (
        0.50 * components.radial
        + 0.35 * components.structure
        + 0.15 * components.boundary
        + problem.bound_penalty * components.bounds
    )
    np.testing.assert_allclose(components.total, expected, rtol=1e-12, atol=1e-12)
    assert float(components.movie) > 0.0


def test_pixel_mismatch_is_diagnostic_only_for_symmetric_rotation(monkeypatch):
    problem, _, _, near_truth, _ = make_tiny_inverse_problem()
    rotated = jnp.rot90(problem.observations, axes=(-2, -1))
    rotated_mass = jnp.sum(
        jnp.where(problem.grid.mask[None], rotated, 0.0), axis=(1, 2)
    ) * problem.grid.cell_area
    monkeypatch.setattr(
        "graphite_stage_transition.inversion.simulate",
        lambda *_args, **_kwargs: SimpleNamespace(
            concentration=rotated,
            mass=rotated_mass,
        ),
    )

    components = loss_components(problem, TRANSFORM.to_unconstrained(near_truth))

    assert float(components.movie) > 0.0
    np.testing.assert_allclose(components.radial, 0.0, atol=1e-12)
    np.testing.assert_allclose(components.structure, 0.0, atol=1e-12)
    np.testing.assert_allclose(components.boundary, 0.0, atol=1e-12)
    np.testing.assert_allclose(components.total, 0.0, atol=1e-12)


def test_inverse_residual_vector_is_zero_at_generating_parameters():
    problem, truth, _, _, _ = make_tiny_inverse_problem()

    residual = inverse_residual_vector(problem, TRANSFORM.to_unconstrained(truth))

    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_tiny_clean_observable_fit_recovers_two_strong_groups():
    problem, _, truth_groups, near_truth, displaced = make_tiny_inverse_problem()

    result = fit_multistart(problem, starts=[near_truth, displaced], maxiter=40)

    assert result.best.loss < float(problem.loss(TRANSFORM.to_unconstrained(displaced)))
    group_error = relative_group_error(result.best.groups, truth_groups)
    assert group_error[0] < 0.02
    assert group_error[1] < 0.02
    # Boundary kinetics remain weakly identified in this eight-step probe.
    assert group_error[2] < 0.5
    assert len(result.starts) == 2
    assert all(start.forward_solves > 0 for start in result.starts)


def test_fit_reuses_components_from_final_objective_evaluation(monkeypatch):
    class CountingProblem:
        transform = TRANSFORM
        grid = SimpleNamespace(dx=1.0, mask=np.ones((1, 1), dtype=bool))

        def __init__(self):
            self.calls = 0

        def components(self, values):
            self.calls += 1
            target = jnp.log(jnp.asarray([0.1, 0.7, 0.0012, 0.2]))
            movie = jnp.sum((values - target) ** 2)
            zero = jnp.asarray(0.0)
            return LossComponents(movie, zero, zero, zero, movie, zero, zero)

        def loss(self, values):
            return self.components(values).total

    def single_evaluation_minimize(objective, values, **_kwargs):
        value, gradient = objective(values)
        return SimpleNamespace(
            x=np.asarray(values),
            fun=value,
            jac=gradient,
            success=True,
            message="converged",
            nit=0,
        )

    problem = CountingProblem()
    initial = CHRParameters(0.1, 0.7, 0.0012, 0.2, 0.5, 1.0)
    monkeypatch.setattr("graphite_stage_transition.inversion.jax.jit", lambda function: function)
    monkeypatch.setattr(
        "graphite_stage_transition.inversion.minimize",
        single_evaluation_minimize,
    )

    result = fit_single_start(problem, initial, maxiter=1)

    assert problem.calls == 1
    assert result.forward_solves == 1
    assert result.components["movie"] == 0.0


def test_reaction_scale_changes_spatial_field_while_preserving_total_current():
    problem, truth, _, _, _ = make_tiny_inverse_problem()
    slow = truth._replace(reaction_rate=0.05)
    fast = truth._replace(reaction_rate=0.5)

    slow_result = simulate(
        problem.grid,
        problem.protocol,
        slow,
        problem.solver,
        initial_concentration=problem.initial_concentration,
    )
    fast_result = simulate(
        problem.grid,
        problem.protocol,
        fast,
        problem.solver,
        initial_concentration=problem.initial_concentration,
    )

    assert np.max(np.abs(slow_result.concentration - fast_result.concentration)) > 1e-3
    np.testing.assert_allclose(
        slow_result.summed_current[1:],
        problem.protocol.current,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fast_result.summed_current[1:],
        problem.protocol.current,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not np.allclose(slow_result.overpotential, fast_result.overpotential)
