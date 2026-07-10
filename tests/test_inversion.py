from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.config import GridConfig, SolverConfig
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.inversion import (
    InverseProblem,
    ParameterTransform,
    centered_finite_difference,
    dimensionless_groups,
    fit_multistart,
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
        mass_penalty=0.1,
        bound_penalty=1e-8,
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


def test_tiny_clean_recovery_reduces_group_error():
    problem, _, truth_groups, near_truth, displaced = make_tiny_inverse_problem()

    result = fit_multistart(problem, starts=[near_truth, displaced], maxiter=40)

    assert result.best.loss < float(problem.loss(TRANSFORM.to_unconstrained(displaced)))
    assert relative_group_error(result.best.groups, truth_groups).max() < 0.05
    assert len(result.starts) == 2
    assert all(start.forward_solves > 0 for start in result.starts)


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
