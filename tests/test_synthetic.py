from collections import defaultdict

import numpy as np

from graphite_stage_transition.synthetic import (
    assign_case_splits,
    corrupt_concentration,
    sample_cases,
    validate_transition_trajectory,
)


BOUNDS = {
    "mobility": (0.1, 0.4),
    "barrier": (0.7, 1.3),
    "kappa": (0.001, 0.0025),
    "reaction_rate": (0.15, 0.4),
}


def test_lhs_cases_are_deterministic():
    first = sample_cases(64, BOUNDS, seed=20260710)
    second = sample_cases(64, BOUNDS, seed=20260710)

    assert first == second
    assert len({case.case_id for case in first}) == 64
    for case in first:
        for name, limits in BOUNDS.items():
            assert limits[0] <= getattr(case, name) <= limits[1]


def test_parameter_case_stays_in_one_split():
    cases = sample_cases(16, BOUNDS, seed=12)
    rows = assign_case_splits(cases, seed=19, replicates=3)
    grouped = defaultdict(set)
    replicate_counts = defaultdict(int)
    for row in rows:
        grouped[row.case.case_id].add(row.split)
        replicate_counts[row.case.case_id] += 1

    assert all(len(value) == 1 for value in grouped.values())
    assert set(replicate_counts.values()) == {3}
    assert {row.split for row in rows} == {"development", "validation", "test"}


def test_noise_is_scaled_to_clean_field_std():
    clean = np.linspace(0.5, 1.0, 400, dtype=float).reshape(4, 10, 10)

    noisy = corrupt_concentration(clean, noise_fraction=0.1, seed=4)
    delta = noisy - clean

    assert np.isclose(delta.std(), 0.1 * clean.std(), rtol=0.15)
    assert noisy.min() >= 0.5
    assert noisy.max() <= 1.0


def test_masked_corruption_preserves_inactive_background():
    clean = np.full((3, 8, 8), 0.75)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    clean[:, ~mask] = 0.0

    noisy = corrupt_concentration(clean, 0.2, seed=3, mask=mask)

    assert np.all(noisy[:, ~mask] == 0.0)
    assert noisy[:, mask].min() >= 0.5
    assert noisy[:, mask].max() <= 1.0


def test_transition_validation_rejects_finite_but_unphysical_field():
    mask = np.ones((6, 6), dtype=bool)
    concentration = np.linspace(0.5, 1.0, 5)[:, None, None] * np.ones((5, 6, 6))
    concentration[2, 0, 0] = -8.0

    try:
        validate_transition_trajectory(concentration, mask, stage2=0.5, stage1=1.0)
    except ValueError as error:
        assert "physical range" in str(error)
    else:
        raise AssertionError("finite but unphysical trajectory was accepted")
