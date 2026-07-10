import json
import tomllib

import numpy as np
import pytest

from graphite_stage_transition.backend_gate import (
    ANALYTIC_TARGET_MODE,
    PROBE_DEFINITION_SHA256,
    BackendGateThresholds,
    BackendProbe,
    BackendGateResult,
    ProbeCase,
    analytic_reference_movie,
    compare_backend_probes,
    load_backend_gate_result,
    load_backend_gate_thresholds,
    load_backend_probe,
    require_matching_passed_gate,
    save_backend_gate_result,
    save_backend_probe,
)


def test_analytic_reference_movie_is_charge_consistent_and_bounded():
    size = 12
    coordinates = (np.arange(size) + 0.5) / size - 0.5
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    mask = x**2 + y**2 <= 0.4**2
    currents = np.asarray([0.01, 0.01, -0.01, -0.01])
    save_indices = np.asarray([0, 2, 4], dtype=np.int32)

    movie = analytic_reference_movie(
        x=x,
        y=y,
        mask=mask,
        currents=currents,
        save_indices=save_indices,
        dt=0.1,
        cell_area=(1.0 / size) ** 2,
        stage2=0.5,
        stage1=1.0,
    )

    expected_mean = 0.5 + np.asarray([0.0, 0.002, 0.0]) / (mask.sum() / size**2)
    np.testing.assert_allclose(movie[:, mask].mean(axis=1), expected_mean, atol=1e-14)
    assert movie[:, mask].min() >= 0.5
    assert movie[:, mask].max() <= 1.0


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
    backend_kind=None,
    fingerprint="fingerprint-a",
    target_mode=ANALYTIC_TARGET_MODE,
    probe_definition_sha256=PROBE_DEFINITION_SHA256,
    jax_default_backend=None,
    metadata_target_mode=None,
    radial=(0.2, 0.4, 0.6),
    structure=(0.1, 0.3),
    boundary=(0.05,),
    objective=0.10,
    gradient=(1.0, 2.0, 3.0, 4.0),
):
    if backend_kind is None:
        backend_kind = "canonical_cpu" if backend == "cpu" else "gpu"
    if jax_default_backend is None:
        jax_default_backend = "cpu" if backend_kind == "canonical_cpu" else "gpu"
    if metadata_target_mode is None:
        metadata_target_mode = target_mode
    return BackendProbe(
        backend=backend,
        backend_kind=backend_kind,
        fingerprint_sha256=fingerprint,
        target_mode=target_mode,
        probe_definition_sha256=probe_definition_sha256,
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
        metadata={
            "device": backend,
            "jax_default_backend": jax_default_backend,
            "devices": [backend],
            "target": metadata_target_mode,
        },
    )


def test_probe_serialization_round_trip(tmp_path):
    path = tmp_path / "probe.json"
    original = _probe("cpu")

    save_backend_probe(original, path)

    assert load_backend_probe(path) == original
    assert json.loads(path.read_text(encoding="ascii"))["schema_version"] == 2


def test_probe_rejects_negative_primary_objective():
    with pytest.raises(ValueError, match="nonnegative"):
        _probe("cpu", objective=-1e-9)


def test_checked_in_config_freezes_approved_thresholds():
    loaded = load_backend_gate_thresholds("configs/backend_gate.toml")
    with open("configs/backend_gate.toml", "rb") as handle:
        policy = tomllib.load(handle)["probe"]

    assert loaded == THRESHOLDS
    assert policy == {
        "target_mode": ANALYTIC_TARGET_MODE,
        "definition_sha256": PROBE_DEFINITION_SHA256,
    }


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
    assert result.backend_kinds == {"cpu": "canonical_cpu", "p100": "gpu"}
    assert result.target_mode == ANALYTIC_TARGET_MODE
    assert result.probe_definition_sha256 == PROBE_DEFINITION_SHA256
    assert set(result.probe_sha256) == {"cpu", "p100"}
    assert all(len(digest) == 64 for digest in result.probe_sha256.values())
    assert result.probe_sha256["cpu"] != result.probe_sha256["p100"]
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


def test_two_cpu_probes_cannot_pass_even_when_backend_labels_differ():
    result = compare_backend_probes(
        (
            _probe(
                "cpu-a",
                backend_kind="canonical_cpu",
                jax_default_backend="cpu",
            ),
            _probe(
                "cpu-b",
                backend_kind="canonical_cpu",
                jax_default_backend="cpu",
            ),
        ),
        THRESHOLDS,
    )

    assert not result.passed
    assert any("exactly one canonical_cpu" in failure for failure in result.failures)
    assert any("at least one gpu" in failure for failure in result.failures)


@pytest.mark.parametrize(
    ("backend_kind", "jax_default_backend"),
    (("canonical_cpu", "gpu"), ("gpu", "cpu")),
)
def test_backend_kind_must_be_backed_by_runtime_metadata(
    backend_kind, jax_default_backend
):
    result = compare_backend_probes(
        (
            _probe("cpu"),
            _probe(
                "second",
                backend_kind=backend_kind,
                jax_default_backend=jax_default_backend,
            ),
        ),
        THRESHOLDS,
    )

    assert not result.passed
    assert any("runtime metadata" in failure for failure in result.failures)


def test_target_mode_and_probe_definition_must_match_across_probes():
    target_mismatch = compare_backend_probes(
        (_probe("cpu"), _probe("p100", target_mode="manifest_observation")),
        THRESHOLDS,
    )
    definition_mismatch = compare_backend_probes(
        (_probe("cpu"), _probe("p100", probe_definition_sha256="b" * 64)),
        THRESHOLDS,
    )

    assert not target_mismatch.passed
    assert target_mismatch.target_mode is None
    assert any("target mode mismatch" in failure for failure in target_mismatch.failures)
    assert not definition_mismatch.passed
    assert definition_mismatch.probe_definition_sha256 is None
    assert any(
        "probe definition mismatch" in failure
        for failure in definition_mismatch.failures
    )


def test_explicit_target_mode_must_be_backed_by_probe_metadata():
    result = compare_backend_probes(
        (
            _probe("cpu"),
            _probe("p100", metadata_target_mode="manifest_observation"),
        ),
        THRESHOLDS,
    )

    assert not result.passed
    assert any("target mode is not backed" in failure for failure in result.failures)


def test_noncanonical_matching_target_cannot_authorize_by_default():
    result = compare_backend_probes(
        (
            _probe("cpu", target_mode="manifest_observation"),
            _probe("p100", target_mode="manifest_observation"),
        ),
        THRESHOLDS,
    )

    assert result.passed
    with pytest.raises(ValueError, match="target mode"):
        require_matching_passed_gate(result, "fingerprint-a")


def test_gate_result_round_trip_and_matching_requirement(tmp_path):
    path = tmp_path / "gate.json"
    result = compare_backend_probes((_probe("cpu"), _probe("p100")), THRESHOLDS)
    save_backend_gate_result(result, path)

    loaded = load_backend_gate_result(path)

    assert loaded == result
    require_matching_passed_gate(loaded, "fingerprint-a")
    with pytest.raises(ValueError, match="fingerprint"):
        require_matching_passed_gate(loaded, "fingerprint-b")
    with pytest.raises(ValueError, match="target mode"):
        require_matching_passed_gate(
            loaded,
            "fingerprint-a",
            expected_target_mode="manifest_observation",
        )
    with pytest.raises(ValueError, match="probe definition"):
        require_matching_passed_gate(
            loaded,
            "fingerprint-a",
            expected_probe_definition_sha256="b" * 64,
        )


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
    assert not result.passed

    with pytest.raises(ValueError, match="frozen thresholds"):
        require_matching_passed_gate(result, "fingerprint-a")


def test_gate_result_rejects_inconsistent_passed_flag():
    valid = compare_backend_probes((_probe("cpu"), _probe("p100")), THRESHOLDS)
    payload = valid.__dict__ | {
        "passed": True,
        "failures": ("fabricated failure",),
    }

    with pytest.raises(ValueError, match="passed must equal"):
        BackendGateResult(**payload)


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"fingerprint_sha256": None}, "nonnull fingerprint"),
        ({"probe_backends": ("cpu",)}, "at least two distinct backends"),
        ({"backend_kinds": {"cpu": "canonical_cpu", "p100": "canonical_cpu"}}, "exactly one canonical_cpu"),
        ({"metrics": {"observable_block_rms": {}, "primary_objective": {}, "gradient": {}}}, "nonempty"),
        (
            {
                "metrics": {
                    "observable_block_rms": {"case-a": {"radial_profile": 0.0}},
                    "primary_objective": {"case-b": {"range": 0.0, "coefficient_of_variation": 0.0}},
                    "gradient": {"case-a": {"minimum_cosine_similarity": 1.0, "maximum_norm_disagreement": 0.0, "all_pairs_below_small_norm": False}},
                }
            },
            "same cases",
        ),
        ({"probe_sha256": {"cpu": "a" * 64}}, "probe_sha256"),
    ),
)
def test_gate_result_rejects_adversarial_passed_payloads(replacement, message):
    valid = compare_backend_probes((_probe("cpu"), _probe("p100")), THRESHOLDS)
    payload = valid.__dict__ | replacement

    with pytest.raises(ValueError, match=message):
        BackendGateResult(**payload)


def test_loader_rejects_adversarial_passed_json(tmp_path):
    valid = compare_backend_probes((_probe("cpu"), _probe("p100")), THRESHOLDS)
    payload = valid.__dict__ | {"fingerprint_sha256": None}
    path = tmp_path / "forged-gate.json"
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match="invalid backend gate result"):
        load_backend_gate_result(path)
