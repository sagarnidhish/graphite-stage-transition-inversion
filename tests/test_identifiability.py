from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.identifiability import (
    _residual_vector,
    fisher_spectrum,
    parameter_correlation,
    profile_likelihood,
    residual_jacobian,
)
from test_inversion import make_tiny_inverse_problem


def test_rank_deficiency_is_reported():
    jacobian = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    report = fisher_spectrum(jacobian, relative_cutoff=1e-8)

    assert report.rank == 1
    assert report.condition_number == np.inf


class QuadraticProfileProblem:
    @staticmethod
    def profile_loss(fixed_index, fixed_value, free_values):
        values = np.empty((2,), dtype=float)
        values[fixed_index] = fixed_value
        values[1 - fixed_index] = free_values[0]
        return values[0] ** 2 + 4.0 * values[1] ** 2


def test_profile_likelihood_holds_parameter_fixed():
    profile = profile_likelihood(
        QuadraticProfileProblem(),
        optimum=np.array([0.0, 0.0]),
        index=1,
        grid=np.array([-0.2, 0.0, 0.2]),
    )

    np.testing.assert_allclose(profile.fixed_values, [-0.2, 0.0, 0.2])
    assert profile.losses[1] <= profile.losses[[0, 2]].min()
    np.testing.assert_allclose(profile.optima[:, 1], profile.fixed_values)


def test_tiny_residual_jacobian_is_finite_and_has_four_columns():
    problem, _, _, near_truth, _ = make_tiny_inverse_problem()
    optimum = problem.transform.to_unconstrained(near_truth)

    jacobian = residual_jacobian(problem, optimum, max_residuals=80)

    assert jacobian.shape == (80, 4)
    assert np.all(np.isfinite(jacobian))
    correlation = parameter_correlation(jacobian)
    assert correlation.shape == (4, 4)
    np.testing.assert_allclose(np.diag(correlation), 1.0, atol=1e-10)


def test_identifiability_source_residual_matches_complete_objective(monkeypatch):
    problem, _, _, near_truth, _ = make_tiny_inverse_problem()
    predicted = jnp.where(
        problem.grid.mask[None],
        problem.observations + 0.55,
        0.0,
    )
    predicted_mass = jnp.sum(predicted, axis=(1, 2)) * problem.grid.cell_area
    monkeypatch.setattr(
        "graphite_stage_transition.inversion.simulate",
        lambda *_args, **_kwargs: SimpleNamespace(
            concentration=predicted,
            mass=predicted_mass,
        ),
    )
    values = problem.transform.to_unconstrained(near_truth)

    residual = _residual_vector(problem, values, max_residuals=100_000)

    np.testing.assert_allclose(
        jnp.mean(residual**2),
        problem.loss(values),
        rtol=1e-12,
        atol=1e-12,
    )
