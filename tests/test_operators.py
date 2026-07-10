import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphite_stage_transition.config import GridConfig
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.operators import masked_laplacian


def test_masked_laplacian_annihilates_constant():
    grid = make_circle_grid(GridConfig(nx=48, ny=48, length=1.0, radius=0.4))
    field = jnp.where(grid.mask, 0.73, 0.0)

    laplacian = masked_laplacian(field, grid)

    np.testing.assert_allclose(np.asarray(laplacian[grid.mask]), 0.0, atol=1e-12)


def test_masked_laplacian_sums_to_zero():
    grid = make_circle_grid(GridConfig(nx=48, ny=48, length=1.0, radius=0.4))
    field = jax.random.normal(jax.random.key(2), (48, 48))

    laplacian = masked_laplacian(field, grid)

    assert abs(float(laplacian.sum())) < 1e-10


def test_masked_laplacian_is_symmetric():
    grid = make_circle_grid(GridConfig(nx=32, ny=32, length=1.0, radius=0.4))
    key_a, key_b = jax.random.split(jax.random.key(9))
    a = jax.random.normal(key_a, grid.mask.shape) * grid.mask
    b = jax.random.normal(key_b, grid.mask.shape) * grid.mask

    lhs = jnp.vdot(a, masked_laplacian(b, grid))
    rhs = jnp.vdot(masked_laplacian(a, grid), b)

    np.testing.assert_allclose(float(lhs), float(rhs), rtol=1e-6, atol=1e-6)


def test_masked_laplacian_uses_isotropic_nine_point_stencil_interior():
    grid = make_circle_grid(GridConfig(nx=32, ny=32, length=1.0, radius=0.4))
    center = (16, 16)
    impulse = jnp.zeros(grid.mask.shape).at[center].set(1.0)

    laplacian = np.asarray(masked_laplacian(impulse, grid)) * grid.dx**2

    assert laplacian[center] == pytest.approx(-20.0 / 6.0)
    for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        index = (center[0] + offset[0], center[1] + offset[1])
        assert laplacian[index] == pytest.approx(4.0 / 6.0)
    for offset in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        index = (center[0] + offset[0], center[1] + offset[1])
        assert laplacian[index] == pytest.approx(1.0 / 6.0)
