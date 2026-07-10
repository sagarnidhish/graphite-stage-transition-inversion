from pathlib import Path

import numpy as np
import graphite_stage_transition.verification as verification

from graphite_stage_transition.config import load_config
from graphite_stage_transition.verification import (
    run_verification_suite,
    verify_full_cycle_transition,
    verify_isotropy,
    verify_mass_balance,
)
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.config import GridConfig, SolverConfig
from graphite_stage_transition.protocols import make_constant_protocol
from graphite_stage_transition.solver import CHRParameters, simulate


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


def test_isotropy_gate_passes_radial_field_and_rejects_cross_field():
    grid = make_circle_grid(GridConfig(64, 64, 2.0, 0.9))
    radius = np.sqrt(np.asarray(grid.x) ** 2 + np.asarray(grid.y) ** 2)
    radial = 0.5 + 0.5 * np.exp(-((radius - 0.45) / 0.08) ** 2)
    radial = np.where(grid.mask, radial, 0.0)
    radial_gate = verify_isotropy(radial[None, ...], grid, 0.0, 1.0)
    assert radial_gate.passed
    cross = radial + 0.20 * (np.abs(np.asarray(grid.x)) < 0.08)
    cross = np.where(grid.mask, cross, 0.0)
    cross_gate = verify_isotropy(cross[None, ...], grid, 0.0, 1.0)
    assert not cross_gate.passed
    assert cross_gate.maximum_angular_rms > radial_gate.maximum_angular_rms


def test_rotation_equivariance_gate_is_available():
    reference = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    rotated = np.rot90(reference, axes=(1, 2))

    passed = verification.verify_rotation_equivariance(reference, rotated)
    failed = verification.verify_rotation_equivariance(reference, reference)

    assert passed.passed
    assert passed.maximum_absolute_difference == 0.0
    assert not failed.passed


def test_solver_is_quarter_turn_equivariant():
    grid = make_circle_grid(GridConfig(24, 24, 1.0, 0.4))
    parameters = CHRParameters(0.05, 0.4, 0.001, 0.2, 0.5, 1.0)
    solver = SolverConfig(0.001, 1e-10, 100, 0.0, 0)
    protocol = make_constant_protocol(0.005, steps=6, dt=solver.dt)
    initial = np.where(
        grid.mask,
        0.7 + 0.02 * np.asarray(grid.x) + 0.01 * np.asarray(grid.y) ** 2,
        0.0,
    )
    reference = simulate(grid, protocol, parameters, solver, initial_concentration=initial)
    rotated = simulate(
        grid,
        protocol,
        parameters,
        solver,
        initial_concentration=np.rot90(initial),
    )

    gate = verification.verify_rotation_equivariance(
        reference.concentration, rotated.concentration, tolerance=1e-9
    )

    assert gate.passed


def test_canonical_verification_passes():
    report = run_verification_suite(load_config(Path("configs/canonical.toml")))

    assert report.mass_balance.passed
    assert report.relaxation.passed
    assert report.refinement.passed
    assert report.determinism.passed
    assert report.passed
