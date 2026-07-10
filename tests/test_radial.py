import importlib.util
import importlib

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from graphite_stage_transition.config import SolverConfig
from graphite_stage_transition.config import GridConfig
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.protocols import make_constant_protocol
from graphite_stage_transition.solver import CHRParameters, SimulationResult


def test_radial_backend_module_is_available():
    assert importlib.util.find_spec("graphite_stage_transition.radial") is not None


def test_radial_geometry_api_is_available():
    radial = importlib.import_module("graphite_stage_transition.radial")
    grid = radial.make_radial_grid(cells=80, radius=0.4)

    assert float(grid.volumes.sum()) == pytest.approx(np.pi * 0.4**2)
    assert float(grid.boundary_weight.sum()) == pytest.approx(2.0 * np.pi * 0.4)
    assert np.all(np.asarray(grid.volumes) > 0.0)
    assert np.count_nonzero(np.asarray(grid.boundary_weight)) == 1


def test_radial_laplacian_api_is_available():
    radial = importlib.import_module("graphite_stage_transition.radial")
    grid = radial.make_radial_grid(cells=40, radius=0.4)
    constant = jnp.full((grid.cells,), 0.73)
    np.testing.assert_allclose(radial.radial_laplacian(constant, grid), 0.0, atol=1e-12)


def test_radial_laplacian_is_volume_conservative_and_self_adjoint():
    radial = importlib.import_module("graphite_stage_transition.radial")
    grid = radial.make_radial_grid(cells=48, radius=0.4)
    key_a, key_b = jax.random.split(jax.random.key(4))
    a = jax.random.normal(key_a, (grid.cells,))
    b = jax.random.normal(key_b, (grid.cells,))
    lap_a = radial.radial_laplacian(a, grid)
    lap_b = radial.radial_laplacian(b, grid)

    assert float(jnp.sum(grid.volumes * lap_a)) == pytest.approx(0.0, abs=1e-11)
    lhs = jnp.sum(grid.volumes * a * lap_b)
    rhs = jnp.sum(grid.volumes * lap_a * b)
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-11, abs=1e-11)


def test_radial_energy_api_is_available():
    radial = importlib.import_module("graphite_stage_transition.radial")
    grid = radial.make_radial_grid(cells=50, radius=0.4)
    concentration = 0.7 + 0.04 * jax.random.normal(jax.random.key(8), (grid.cells,))
    gradient = jax.grad(radial.radial_total_free_energy)(
        concentration, grid, 1.2, 0.003, 0.5, 1.0
    )
    chemical = radial.radial_chemical_potential(
        concentration, grid, 1.2, 0.003, 0.5, 1.0
    )

    np.testing.assert_allclose(
        np.asarray(gradient / grid.volumes),
        np.asarray(chemical),
        rtol=2e-10,
        atol=2e-10,
    )


def test_radial_simulation_api_is_available():
    radial = importlib.import_module("graphite_stage_transition.radial")
    assert hasattr(radial, "simulate_radial")


def test_radial_uniform_equilibrium_is_stationary():
    radial = importlib.import_module("graphite_stage_transition.radial")
    grid = radial.make_radial_grid(32, 0.4)
    parameters = CHRParameters(0.05, 0.4, 0.001, 0.2, 0.5, 1.0)
    solver = SolverConfig(0.001, 1e-10, 100, 0.0, 0)
    protocol = make_constant_protocol(0.0, steps=8, dt=solver.dt)

    result = radial.simulate_radial(grid, protocol, parameters, solver)

    np.testing.assert_allclose(result.concentration[-1], result.concentration[0], atol=2e-9)
    assert float(np.max(np.asarray(result.cg_residual))) < 1e-8


def test_radial_mass_change_matches_integrated_current():
    radial = importlib.import_module("graphite_stage_transition.radial")
    grid = radial.make_radial_grid(40, 0.4)
    parameters = CHRParameters(0.05, 0.4, 0.001, 0.2, 0.5, 1.0)
    solver = SolverConfig(0.001, 1e-10, 100, 0.0, 0)
    protocol = make_constant_protocol(0.012, steps=12, dt=solver.dt)
    initial = jnp.full((grid.cells,), 0.75)

    result = radial.simulate_radial(
        grid, protocol, parameters, solver, initial_concentration=initial
    )

    measured = float(result.mass[-1] - result.mass[0])
    expected = float(protocol.current.sum() * solver.dt)
    assert measured == pytest.approx(expected, abs=2e-12)


def test_radial_rasterization_api_is_available():
    radial = importlib.import_module("graphite_stage_transition.radial")
    radial_grid = radial.make_radial_grid(80, 0.4)
    cartesian_grid = make_circle_grid(GridConfig(64, 64, 1.0, 0.4))
    image = radial.rasterize_radial(
        jnp.full((radial_grid.cells,), 0.72), radial_grid, cartesian_grid
    )

    np.testing.assert_allclose(np.asarray(image)[cartesian_grid.mask], 0.72, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(image)[~np.asarray(cartesian_grid.mask)], 0.0)


def test_interface_resolution_api_is_available():
    radial = importlib.import_module("graphite_stage_transition.radial")
    parameters = CHRParameters(0.2, 1.0, 0.0015, 0.25, 0.5, 1.0)
    width = radial.diffuse_interface_width_10_90(parameters)

    assert width == pytest.approx(0.01504337, rel=2e-6)
    grid = radial.make_radial_grid(128, 0.4)
    assert width / grid.dr >= 4.0


def test_radial_full_cycle_gate_api_is_available():
    radial = importlib.import_module("graphite_stage_transition.radial")
    grid = radial.make_radial_grid(12, 0.4)
    valid = np.stack(
        (
            np.full(grid.cells, 0.5),
            np.full(grid.cells, 1.0),
            np.full(grid.cells, 0.5),
        )
    )

    assert radial.verify_radial_full_cycle(valid, grid, 0.5, 1.0).passed
    assert not radial.verify_radial_full_cycle(valid[:2], grid, 0.5, 1.0).passed


def test_radial_refinement_gate_api_is_available():
    radial = importlib.import_module("graphite_stage_transition.radial")
    trajectories = {}
    for cells in (40, 80):
        grid = radial.make_radial_grid(cells, 0.4)
        profile = 0.5 + 0.5 / (1.0 + np.exp(-(np.asarray(grid.centers) - 0.2) / 0.01))
        concentration = np.stack((profile, profile))
        zeros = np.zeros(2)
        result = SimulationResult(
            concentration, np.array([0.0, 1.0]), zeros, zeros, zeros,
            zeros, zeros, zeros, {},
        )
        trajectories[cells] = (result, grid)

    gate = radial.verify_radial_refinement(trajectories)

    assert gate.passed
    assert gate.max_displacement_pixels < 0.1
