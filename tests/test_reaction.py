import jax.numpy as jnp
import numpy as np
import pytest

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


def test_reaction_is_zero_away_from_boundary():
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
