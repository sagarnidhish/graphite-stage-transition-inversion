import numpy as np
import jax.numpy as jnp

from graphite_stage_transition.config import GridConfig
from graphite_stage_transition.geometry import (
    _boundary_face_count,
    make_circle_grid,
    radial_bin_indices,
)


def test_boundary_face_count_is_integer_exposed_face_count():
    mask = jnp.ones((3, 3), dtype=bool)
    expected = np.array([[2, 1, 2], [1, 0, 1], [2, 1, 2]])

    np.testing.assert_array_equal(np.asarray(_boundary_face_count(mask)), expected)


def test_circle_boundary_is_strict_subset_of_active_cells():
    grid = make_circle_grid(GridConfig(nx=48, ny=48, length=1.0, radius=0.4))
    face_count = np.asarray(grid.boundary_weight / grid.dx)

    assert np.count_nonzero(face_count) == 108
    assert face_count.sum() == 152
    assert set(np.unique(face_count)) == {0.0, 1.0, 2.0}
    assert np.count_nonzero(face_count) < grid.active_count


def test_circle_grid_is_centered_and_has_boundary():
    grid = make_circle_grid(GridConfig(nx=64, ny=64, length=1.0, radius=0.4))

    assert grid.mask.shape == (64, 64)
    assert grid.boundary_weight.shape == grid.mask.shape
    assert int(grid.mask.sum()) > 0
    assert float(grid.boundary_weight.sum()) > 0.0
    assert abs(float((grid.x * grid.mask).sum())) < 1e-12
    assert abs(float((grid.y * grid.mask).sum())) < 1e-12


def test_radial_bins_cover_every_active_cell_once():
    grid = make_circle_grid(GridConfig(nx=48, ny=48, length=1.0, radius=0.4))

    indices = radial_bin_indices(grid, bins=12)
    active = np.asarray(grid.mask)

    assert indices.shape == active.shape
    assert np.all(indices[active] >= 0)
    assert np.all(indices[active] < 12)
    assert np.all(indices[~active] == -1)
