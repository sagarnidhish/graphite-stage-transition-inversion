import jax.numpy as jnp
import numpy as np
import pytest

from graphite_stage_transition.config import GridConfig, SolverConfig
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.protocols import make_constant_protocol
from graphite_stage_transition.solver import CHRParameters, simulate


def _small_system(initial):
    grid = make_circle_grid(GridConfig(nx=20, ny=20, length=1.0, radius=0.38))
    parameters = CHRParameters(
        mobility=0.05,
        barrier=0.4,
        kappa=0.001,
        reaction_rate=0.2,
        stage2=0.5,
        stage1=1.0,
    )
    solver = SolverConfig(
        dt=0.002,
        cg_tolerance=1e-10,
        cg_max_iterations=100,
        perturbation_amplitude=0.0,
        seed=0,
    )
    concentration = jnp.where(grid.mask, initial, 0.0)
    return grid, parameters, solver, concentration


def test_uniform_equilibrium_is_stationary():
    grid, parameters, solver, initial = _small_system(initial=0.5)
    protocol = make_constant_protocol(current=0.0, steps=8, dt=solver.dt)

    result = simulate(grid, protocol, parameters, solver, initial_concentration=initial, seed=0)

    np.testing.assert_allclose(result.concentration[-1], result.concentration[0], atol=2e-8)
    assert np.max(np.asarray(result.cg_residual)) < 1e-8


def test_current_changes_mass_with_correct_sign():
    grid, parameters, solver, initial = _small_system(initial=0.75)
    positive = make_constant_protocol(current=0.01, steps=10, dt=solver.dt)
    negative = make_constant_protocol(current=-0.01, steps=10, dt=solver.dt)

    lithiation = simulate(grid, positive, parameters, solver, initial_concentration=initial, seed=1)
    delithiation = simulate(grid, negative, parameters, solver, initial_concentration=initial, seed=1)

    assert lithiation.mass[-1] > lithiation.mass[0]
    assert delithiation.mass[-1] < delithiation.mass[0]


def test_mass_change_matches_integrated_current():
    grid, parameters, solver, initial = _small_system(initial=0.75)
    protocol = make_constant_protocol(current=0.012, steps=12, dt=solver.dt)

    result = simulate(grid, protocol, parameters, solver, initial_concentration=initial, seed=2)

    expected = float(protocol.current.sum() * solver.dt)
    measured = float(result.mass[-1] - result.mass[0])
    np.testing.assert_allclose(measured, expected, rtol=2e-7, atol=2e-10)


def test_simulation_is_deterministic_for_seed():
    grid, parameters, solver, initial = _small_system(initial=0.7)
    solver = SolverConfig(**{**solver.__dict__, "perturbation_amplitude": 1e-5})
    protocol = make_constant_protocol(current=0.005, steps=6, dt=solver.dt)

    first = simulate(grid, protocol, parameters, solver, initial_concentration=initial, seed=8)
    second = simulate(grid, protocol, parameters, solver, initial_concentration=initial, seed=8)

    np.testing.assert_array_equal(first.concentration, second.concentration)


def test_stiff_multistep_run_remains_finite_and_near_physical_range():
    grid = make_circle_grid(GridConfig(nx=32, ny=32, length=1.0, radius=0.4))
    parameters = CHRParameters(0.2, 1.0, 0.0015, 0.25, 0.5, 1.0)
    solver = SolverConfig(0.002, 1e-8, 200, 1e-4, 7)
    protocol = make_constant_protocol(current=0.02, steps=100, dt=solver.dt, save_every=10)

    result = simulate(grid, protocol, parameters, solver, seed=7)
    concentration = np.asarray(result.concentration)[:, np.asarray(grid.mask)]

    assert np.all(np.isfinite(concentration))
    assert np.all(np.isfinite(np.asarray(result.summed_current)))
    assert np.all(np.isfinite(np.asarray(result.cg_residual)))
    assert concentration.min() > 0.45
    assert concentration.max() < 1.05


def test_iterative_solve_projects_the_exact_mass_mode():
    grid, parameters, solver, initial = _small_system(initial=0.7)
    solver = SolverConfig(
        dt=solver.dt,
        cg_tolerance=1e-2,
        cg_max_iterations=2,
        perturbation_amplitude=0.0,
        seed=0,
    )
    protocol = make_constant_protocol(current=0.015, steps=120, dt=solver.dt)

    result = simulate(grid, protocol, parameters, solver, initial_concentration=initial)

    expected = float(protocol.current.sum() * solver.dt)
    measured = float(result.mass[-1] - result.mass[0])
    assert measured == pytest.approx(expected, abs=2e-13)
