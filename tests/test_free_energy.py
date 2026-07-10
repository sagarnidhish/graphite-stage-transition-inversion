import jax
import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.config import GridConfig
from graphite_stage_transition.free_energy import (
    homogeneous_free_energy,
    homogeneous_mu,
    total_free_energy,
)
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.operators import masked_laplacian


def test_double_well_has_stage_minima_and_requested_barrier():
    concentration = jnp.array([0.5, 0.75, 1.0])

    energy = homogeneous_free_energy(concentration, barrier=2.0, stage2=0.5, stage1=1.0)
    chemical = homogeneous_mu(concentration, barrier=2.0, stage2=0.5, stage1=1.0)

    np.testing.assert_allclose(np.asarray(energy), [0.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(chemical)[[0, 2]], 0.0, atol=1e-12)


def test_analytic_chemical_term_matches_autodiff():
    concentrations = jnp.linspace(0.52, 0.98, 11)
    autodiff = jax.vmap(jax.grad(lambda value: homogeneous_free_energy(value, 1.7, 0.5, 1.0)))(
        concentrations
    )

    analytic = homogeneous_mu(concentrations, barrier=1.7, stage2=0.5, stage1=1.0)

    np.testing.assert_allclose(np.asarray(analytic), np.asarray(autodiff), rtol=1e-12, atol=1e-12)


def test_uniform_stage_has_zero_total_free_energy():
    grid = make_circle_grid(GridConfig(nx=32, ny=32, length=1.0, radius=0.4))
    concentration = jnp.where(grid.mask, 0.5, 0.0)

    energy = total_free_energy(concentration, grid, barrier=2.0, kappa=0.01, stage2=0.5, stage1=1.0)

    assert float(energy) == 0.0


def test_total_energy_gradient_matches_full_discrete_chemical_potential():
    grid = make_circle_grid(GridConfig(nx=20, ny=20, length=1.0, radius=0.4))
    key = jax.random.key(12)
    concentration = jnp.where(
        grid.mask,
        0.7 + 0.03 * jax.random.normal(key, grid.mask.shape),
        0.0,
    )
    barrier = 1.3
    kappa = 0.004

    gradient = jax.grad(total_free_energy)(
        concentration, grid, barrier, kappa, 0.5, 1.0
    )
    expected = jnp.where(
        grid.mask,
        homogeneous_mu(concentration, barrier, 0.5, 1.0)
        - kappa * masked_laplacian(concentration, grid),
        0.0,
    )

    np.testing.assert_allclose(
        np.asarray(gradient[grid.mask] / grid.cell_area),
        np.asarray(expected[grid.mask]),
        rtol=2e-10,
        atol=2e-10,
    )
