import jax
import jax.numpy as jnp
import numpy as np

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
