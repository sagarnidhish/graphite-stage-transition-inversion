import jax.numpy as jnp
import numpy as np
import pytest

from graphite_stage_transition.config import GridConfig
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.reaction import galvanostatic_reaction


@pytest.mark.parametrize("target", [0.03, 0.0, -0.03])
def test_galvanostatic_reaction_matches_target(target):
    concentration = jnp.array([0.55, 0.70, 0.90])
    chemical_potential = jnp.array([-0.2, 0.0, 0.15])
    boundary_weight = jnp.array([1.0, 2.0, 1.0])

    state = galvanostatic_reaction(
        concentration,
        chemical_potential,
        boundary_weight,
        target_current=target,
        reaction_rate=0.4,
        stage2=0.5,
        stage1=1.0,
    )

    np.testing.assert_allclose(float(state.summed_current), target, rtol=1e-10, atol=1e-12)
    assert np.all(np.isfinite(np.asarray(state.rate)))
    if target != 0.0:
        assert np.sign(float(state.summed_current)) == np.sign(target)


def test_reaction_and_exchange_are_zero_away_from_boundary():
    state = galvanostatic_reaction(
        jnp.array([0.6, 0.7, 0.8]),
        jnp.zeros(3),
        jnp.array([1.0, 0.0, 2.0]),
        target_current=0.01,
        reaction_rate=0.3,
        stage2=0.5,
        stage1=1.0,
    )

    assert float(state.rate[1]) == 0.0
    assert float(state.exchange_weight[1]) == 0.0


def test_circle_reaction_and_exchange_are_localized_to_boundary_cells():
    grid = make_circle_grid(GridConfig(nx=48, ny=48, length=1.0, radius=0.4))
    concentration = jnp.where(grid.mask, 0.75, 0.0)
    chemical_potential = jnp.zeros_like(concentration)

    state = galvanostatic_reaction(
        concentration,
        chemical_potential,
        grid.boundary_weight,
        target_current=0.01,
        reaction_rate=0.3,
        stage2=0.5,
        stage1=1.0,
    )

    interior = np.asarray(grid.mask & (grid.boundary_weight == 0.0))
    assert interior.any()
    assert np.all(np.asarray(state.rate)[interior] == 0.0)
    assert np.all(np.asarray(state.exchange_weight)[interior] == 0.0)
