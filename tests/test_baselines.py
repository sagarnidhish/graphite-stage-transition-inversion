import jax.numpy as jnp
import numpy as np
import pytest

from graphite_stage_transition.baselines import (
    fit_random_search,
    mean_only_loss,
    simulate_fickian,
    simulate_sharp_interface,
    spatial_loss,
)
from graphite_stage_transition.config import GridConfig, SolverConfig
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.inversion import ParameterTransform
from graphite_stage_transition.protocols import make_constant_protocol
from graphite_stage_transition.solver import CHRParameters


class QuadraticProblem:
    transform = ParameterTransform(
        lower=(0.02, 0.2, 0.0002, 0.03),
        upper=(0.5, 2.0, 0.006, 0.8),
    )

    @staticmethod
    def loss(values):
        target = jnp.log(jnp.array([0.1, 0.7, 0.0012, 0.2]))
        return jnp.sum((values - target) ** 2)


def _baseline_system():
    grid = make_circle_grid(GridConfig(nx=20, ny=20, length=1.0, radius=0.38))
    solver = SolverConfig(0.001, 1e-10, 100, 0.0, 2)
    protocol = make_constant_protocol(0.01, steps=10, dt=solver.dt, save_every=2)
    initial = jnp.where(grid.mask, 0.6, 0.0)
    return grid, solver, protocol, initial


def test_random_search_respects_forward_budget():
    result = fit_random_search(QuadraticProblem(), budget=17, seed=5)

    assert result.forward_solves == 17
    assert np.isfinite(result.loss)


def test_mean_only_discards_spatial_information():
    movie_a = np.zeros((3, 8, 8))
    movie_b = np.zeros_like(movie_a)
    movie_a[:, :, :4] = 1.0
    movie_b[:, :, 4:] = 1.0
    mask = np.ones((8, 8), dtype=bool)

    assert mean_only_loss(movie_a, movie_b, mask) == pytest.approx(0.0)
    assert spatial_loss(movie_a, movie_b, mask) > 0.0


def test_fickian_control_preserves_imposed_mass_change():
    grid, solver, protocol, initial = _baseline_system()

    result = simulate_fickian(grid, protocol, diffusivity=0.02, solver=solver, initial=initial)

    expected = float(protocol.current.sum() * solver.dt)
    assert float(result.mass[-1] - result.mass[0]) == pytest.approx(expected, rel=2e-8)


def test_fickian_control_is_stable_at_high_diffusivity():
    grid, solver, _, _ = _baseline_system()
    protocol = make_constant_protocol(0.0, steps=40, dt=solver.dt, save_every=4)
    initial = jnp.where(grid.mask, 0.55, 0.0)
    initial = initial.at[grid.mask.shape[0] // 2, grid.mask.shape[1] // 2].set(0.95)

    result = simulate_fickian(grid, protocol, diffusivity=2.0, solver=solver, initial=initial)
    active = np.asarray(result.concentration)[:, np.asarray(grid.mask)]

    assert np.all(np.isfinite(active))
    assert active.min() > 0.4
    assert active.max() < 1.1


def test_sharp_interface_is_bounded_and_conservative():
    grid, solver, protocol, initial = _baseline_system()

    result = simulate_sharp_interface(grid, protocol, initial_mean=0.6)
    active = np.asarray(result.concentration)[:, np.asarray(grid.mask)]

    assert active.min() >= 0.5
    assert active.max() <= 1.0
    expected = float(protocol.current.sum() * solver.dt)
    assert float(result.mass[-1] - result.mass[0]) == pytest.approx(expected, abs=1e-12)
