from pathlib import Path

import numpy as np

from graphite_stage_transition.config import load_config
from graphite_stage_transition.verification import (
    run_verification_suite,
    verify_full_cycle_transition,
    verify_mass_balance,
)


def test_full_cycle_gate_requires_both_stage_endpoints_and_return():
    mask = np.ones((2, 2), dtype=bool)
    valid = np.asarray(
        [
            np.full((2, 2), 0.5),
            np.full((2, 2), 1.0),
            np.full((2, 2), 0.5),
        ]
    )

    passed = verify_full_cycle_transition(valid, mask, 0.5, 1.0)
    incomplete = verify_full_cycle_transition(valid[:2], mask, 0.5, 1.0)

    assert passed.passed
    assert not incomplete.passed


def test_full_cycle_gate_rejects_nonfinite_or_large_bound_excursion():
    mask = np.ones((2, 2), dtype=bool)
    movie = np.asarray(
        [
            np.full((2, 2), 0.5),
            np.asarray([[1.06, 0.98], [0.98, 0.98]]),
            np.full((2, 2), 0.5),
        ]
    )

    report = verify_full_cycle_transition(movie, mask, 0.5, 1.0)

    assert not report.passed
    assert report.maximum_concentration == 1.06


def test_mass_report_fails_bad_trajectory():
    times = np.linspace(0.0, 1.0, 11)
    currents = np.full_like(times, 0.02)
    mass_with_drift = 0.25 + 0.03 * times

    report = verify_mass_balance(times, currents, mass_with_drift, tolerance=1e-5)

    assert not report.passed
    assert report.relative_error > 1e-5


def test_mass_report_passes_exact_right_endpoint_integral():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    currents = np.array([0.02, 0.02, 0.0, -0.01])
    expected_change = np.concatenate(
        ([0.0], np.cumsum(currents[1:] * np.diff(times)))
    )

    report = verify_mass_balance(
        times,
        currents,
        0.25 + expected_change,
        tolerance=1e-12,
    )

    assert report.passed
    assert report.max_absolute_error < 1e-14


def test_canonical_verification_passes():
    report = run_verification_suite(load_config(Path("configs/canonical.toml")))

    assert report.mass_balance.passed
    assert report.relaxation.passed
    assert report.refinement.passed
    assert report.determinism.passed
    assert report.passed
