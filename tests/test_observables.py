import jax
import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.config import GridConfig
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.observables import (
    make_observable_geometry,
    observable_residual_vector,
    physics_observables,
)


def _grid_and_geometry(size: int = 32):
    grid = make_circle_grid(
        GridConfig(nx=size, ny=size, length=1.0, radius=0.4)
    )
    return grid, make_observable_geometry(grid)


def test_uniform_movie_has_constant_radial_profile_and_zero_other_blocks():
    grid, geometry = _grid_and_geometry()
    filling = 0.37
    movie = jnp.where(grid.mask, 0.5 + 0.5 * filling, 0.0)[None, ...]

    observables = physics_observables(movie, geometry, stage2=0.5, stage1=1.0)

    np.testing.assert_allclose(observables.radial_profile, filling, atol=1e-12)
    np.testing.assert_allclose(observables.structure_power, 0.0, atol=1e-12)
    np.testing.assert_allclose(observables.boundary_excess, 0.0, atol=1e-12)


def test_primary_observables_are_invariant_to_square_grid_symmetries():
    grid, geometry = _grid_and_geometry()
    filling = jnp.where(
        grid.mask,
        0.45
        + 0.11 * grid.x / grid.radius
        - 0.08 * grid.y / grid.radius
        + 0.06 * grid.x * grid.y / grid.radius**2,
        0.0,
    )
    movie = jnp.stack((0.5 + 0.5 * filling, 0.5 + 0.45 * filling))
    reference = physics_observables(movie, geometry, 0.5, 1.0)

    transforms = (
        lambda values: jnp.rot90(values, 1, axes=(-2, -1)),
        lambda values: jnp.flip(values, axis=-1),
        lambda values: jnp.swapaxes(values, -2, -1),
    )
    for transform in transforms:
        transformed = physics_observables(transform(movie), geometry, 0.5, 1.0)
        np.testing.assert_allclose(
            transformed.radial_profile, reference.radial_profile, atol=1e-12
        )
        np.testing.assert_allclose(
            transformed.structure_power, reference.structure_power, atol=1e-12
        )
        np.testing.assert_allclose(
            transformed.boundary_excess, reference.boundary_excess, atol=1e-12
        )


def test_structure_power_sum_is_four_times_active_region_variance():
    grid, geometry = _grid_and_geometry()
    filling = jnp.where(
        grid.mask,
        0.35 + 0.2 * jnp.sin(7.0 * grid.x) - 0.15 * jnp.cos(9.0 * grid.y),
        0.0,
    )
    movie = (0.5 + 0.5 * filling)[None, ...]

    observables = physics_observables(movie, geometry, 0.5, 1.0)
    active = np.asarray(filling)[np.asarray(grid.mask)]

    np.testing.assert_allclose(
        np.asarray(observables.structure_power).sum(),
        4.0 * np.var(active),
        rtol=1e-12,
        atol=1e-12,
    )


def test_equal_mean_core_shell_reversal_is_distinguished():
    grid, geometry = _grid_and_geometry()
    radial_bin = np.argmax(np.asarray(geometry.radial_weights), axis=0)
    core = jnp.asarray((radial_bin < 4) & np.asarray(grid.mask), dtype=jnp.float64)
    shell = jnp.asarray((radial_bin >= 4) & np.asarray(grid.mask), dtype=jnp.float64)
    core = core * (0.5 * grid.active_count / jnp.sum(core))
    shell = shell * (0.5 * grid.active_count / jnp.sum(shell))
    first = jnp.where(grid.mask, core, 0.0)
    second = jnp.where(grid.mask, shell, 0.0)
    np.testing.assert_allclose(
        jnp.mean(first[grid.mask]), jnp.mean(second[grid.mask]), atol=1e-12
    )

    first_obs = physics_observables((0.5 + 0.5 * first)[None], geometry, 0.5, 1.0)
    second_obs = physics_observables((0.5 + 0.5 * second)[None], geometry, 0.5, 1.0)
    residual = observable_residual_vector(first_obs, second_obs)

    assert np.linalg.norm(np.asarray(residual)) > 0.1


def test_identical_movies_have_zero_weighted_residual():
    grid, geometry = _grid_and_geometry(24)
    movie = jnp.stack(
        (
            jnp.where(grid.mask, 0.62 + 0.03 * grid.x, 0.0),
            jnp.where(grid.mask, 0.83 - 0.02 * grid.y, 0.0),
        )
    )
    observables = physics_observables(movie, geometry, 0.5, 1.0)

    residual = observable_residual_vector(observables, observables)

    np.testing.assert_array_equal(residual, jnp.zeros_like(residual))


def test_weighted_residual_mean_square_matches_frozen_block_objective():
    grid, geometry = _grid_and_geometry(24)
    first_movie = jnp.stack(
        (
            jnp.where(grid.mask, 0.65 + 0.03 * grid.x, 0.0),
            jnp.where(grid.mask, 0.80 - 0.04 * grid.y, 0.0),
        )
    )
    second_movie = jnp.stack(
        (
            jnp.where(grid.mask, 0.70 - 0.02 * grid.x, 0.0),
            jnp.where(grid.mask, 0.77 + 0.05 * grid.y, 0.0),
        )
    )
    first = physics_observables(first_movie, geometry, 0.5, 1.0)
    second = physics_observables(second_movie, geometry, 0.5, 1.0)

    residual = observable_residual_vector(first, second)
    expected = (
        0.50 * jnp.mean((first.radial_profile - second.radial_profile) ** 2)
        + 0.35 * jnp.mean((first.structure_power - second.structure_power) ** 2)
        + 0.15 * jnp.mean((first.boundary_excess - second.boundary_excess) ** 2)
    )

    np.testing.assert_allclose(jnp.mean(residual**2), expected, rtol=1e-12, atol=1e-12)


def test_observable_autodiff_agrees_with_centered_finite_difference():
    grid, geometry = _grid_and_geometry(20)
    base = jnp.where(grid.mask, 0.68 + 0.04 * grid.x - 0.03 * grid.y, 0.0)
    target = physics_observables(base[None], geometry, 0.5, 1.0)

    def objective(scale):
        movie = jnp.where(grid.mask, base + scale * (grid.x**2 - grid.y**2), 0.0)[None]
        candidate = physics_observables(movie, geometry, 0.5, 1.0)
        residual = observable_residual_vector(candidate, target)
        return jnp.mean(residual**2)

    point = 0.07
    step = 1e-5
    automatic = jax.grad(objective)(point)
    numerical = (objective(point + step) - objective(point - step)) / (2.0 * step)

    np.testing.assert_allclose(automatic, numerical, rtol=1e-7, atol=1e-10)
