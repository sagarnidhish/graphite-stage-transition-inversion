import json

import pytest

from graphite_stage_transition.backend_gate import (
    BackendGateThresholds,
    BackendProbe,
    ProbeCase,
    compare_backend_probes,
    load_backend_gate_result,
    load_backend_gate_thresholds,
    load_backend_probe,
    require_matching_passed_gate,
    save_backend_gate_result,
    save_backend_probe,
)


THRESHOLDS = BackendGateThresholds(
    observable_block_rms_max=0.02,
    objective_range_max=0.005,
    objective_cv_max=0.05,
    gradient_cosine_min=0.99,
    gradient_norm_disagreement_max=0.10,
    gradient_small_norm=1e-6,
)


def _probe(
    backend,
    *,
    fingerprint="fingerprint-a",
    radial=(0.2, 0.4, 0.6),
    structure=(0.1, 0.3),
    boundary=(0.05,),
    objective=0.10,
    gradient=(1.0, 2.0, 3.0, 4.0),
):
    return BackendProbe(
        backend=backend,
        fingerprint_sha256=fingerprint,
        cases=(
            ProbeCase(
                case_id="case-a",
                observable_blocks={
                    "radial_profile": radial,
                    "structure_power": structure,
                    "boundary_excess": boundary,
                },
                primary_objective=objective,
                gradient=gradient,
            ),
        ),
        metadata={"device": backend},
    )


def test_probe_serialization_round_trip(tmp_path):
    path = tmp_path / "probe.json"
    original = _probe("cpu")

    save_backend_probe(original, path)

    assert load_backend_probe(path) == original
    assert json.loads(path.read_text(encoding="ascii"))["schema_version"] == 1


def test_probe_rejects_negative_primary_objective():
    with pytest.raises(ValueError, match="nonnegative"):
        _probe("cpu", objective=-1e-9)


def test_checked_in_config_freezes_approved_thresholds():
    loaded = load_backend_gate_thresholds("configs/backend_gate.toml")

    assert loaded == THRESHOLDS


def test_matching_probes_inside_frozen_thresholds_pass():
    cpu = _probe("cpu")
    gpu = _probe(
        "p100",
        radial=(0.205, 0.395, 0.605),
        structure=(0.102, 0.298),
        boundary=(0.052,),
        objective=0.103,
        gradient=(1.01, 2.02, 3.03, 4.04),
    )

    result = compare_backend_probes((cpu, gpu), THRESHOLDS)

    assert result.passed
    assert result.failures == ()
    assert result.metrics["observable_block_rms"]["case-a"]["radial_profile"] < 0.02
    assert result.metrics["primary_objective"]["case-a"]["range"] == pytest.approx(0.003)
    assert result.metrics["gradient"]["case-a"]["minimum_cosine_similarity"] > 0.99


def test_observable_block_rms_over_threshold_fails():
    cpu = _probe("cpu")
    gpu = _probe("p100", boundary=(0.071,))

    result = compare_backend_probes((cpu, gpu), THRESHOLDS)

    assert not result.passed
    assert result.metrics["observable_block_rms"]["case-a"]["boundary_excess"] == pytest.approx(0.021)
    assert any("boundary_excess RMS" in failure for failure in result.failures)


def test_primary_objective_range_and_cv_are_both_gated():
    cpu = _probe("cpu", objective=0.10)
    gpu = _probe("p100", objective=0.106)

    result = compare_backend_probes((cpu, gpu), THRESHOLDS)

    assert not result.passed
    metrics = result.metrics["primary_objective"]["case-a"]
    assert metrics["range"] == pytest.approx(0.006)
    assert metrics["coefficient_of_variation"] < 0.05
    assert any("objective range" in failure for failure in result.failures)


def test_gradient_direction_and_norm_are_both_gated():
    cpu = _probe("cpu", gradient=(1.0, 0.0, 0.0, 0.0))
    gpu = _probe("p100", gradient=(0.0, 1.2, 0.0, 0.0))

    result = compare_backend_probes((cpu, gpu), THRESHOLDS)

    assert not result.passed
    metrics = result.metrics["gradient"]["case-a"]
    assert metrics["minimum_cosine_similarity"] == pytest.approx(0.0)
    assert metrics["maximum_norm_disagreement"] == pytest.approx(1.0 / 6.0)
    assert any("gradient cosine" in failure for failure in result.failures)
    assert any("gradient norm disagreement" in failure for failure in result.failures)


def test_two_tiny_gradients_bypass_direction_and_norm_gate():
    cpu = _probe("cpu", gradient=(1e-8, 0.0, 0.0, 0.0))
    gpu = _probe("p100", gradient=(0.0, 9e-7, 0.0, 0.0))

    result = compare_backend_probes((cpu, gpu), THRESHOLDS)

    assert result.passed
    assert result.metrics["gradient"]["case-a"]["all_pairs_below_small_norm"]


def test_fingerprint_mismatch_fails_closed():
    result = compare_backend_probes(
        (_probe("cpu"), _probe("p100", fingerprint="fingerprint-b")),
        THRESHOLDS,
    )

    assert not result.passed
    assert result.fingerprint_sha256 is None
    assert any("fingerprint" in failure for failure in result.failures)


def test_gate_result_round_trip_and_matching_requirement(tmp_path):
    path = tmp_path / "gate.json"
    result = compare_backend_probes((_probe("cpu"), _probe("p100")), THRESHOLDS)
    save_backend_gate_result(result, path)

    loaded = load_backend_gate_result(path)

    assert loaded == result
    require_matching_passed_gate(loaded, "fingerprint-a")
    with pytest.raises(ValueError, match="fingerprint"):
        require_matching_passed_gate(loaded, "fingerprint-b")


def test_failed_gate_cannot_authorize_claim_eligible_execution():
    failed = compare_backend_probes(
        (_probe("cpu"), _probe("p100", boundary=(0.08,))), THRESHOLDS
    )

    with pytest.raises(ValueError, match="did not pass"):
        require_matching_passed_gate(failed, "fingerprint-a")


def test_gate_with_noncanonical_thresholds_cannot_authorize_claims():
    loose = BackendGateThresholds(observable_block_rms_max=1.0)
    result = compare_backend_probes(
        (_probe("cpu"), _probe("p100", boundary=(0.08,))), loose
    )
    assert result.passed

    with pytest.raises(ValueError, match="frozen thresholds"):
        require_matching_passed_gate(result, "fingerprint-a")
