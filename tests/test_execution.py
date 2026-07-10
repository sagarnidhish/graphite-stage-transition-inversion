import base64
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import pytest

from graphite_stage_transition.execution import (
    allocate_worker_cores,
    build_execution_fingerprint,
    canonical_json_sha256,
    load_canonical_environment,
    stable_task_seed,
    validate_canonical_environment,
    verify_execution_fingerprint,
)


def test_task_seed_depends_on_task_identity_not_queue_position():
    first = stable_task_seed(20260710, "task_alpha")
    repeated = stable_task_seed(20260710, "task_alpha")
    other_task = stable_task_seed(20260710, "task_beta")
    other_base = stable_task_seed(20260711, "task_alpha")

    assert first == repeated
    assert first != other_task
    assert first != other_base
    assert 0 <= first < 2**31


def test_canonical_json_hash_ignores_mapping_order():
    left = {"optimizer": {"maxiter": 4, "starts": 2}, "seed": 7}
    right = {"seed": 7, "optimizer": {"starts": 2, "maxiter": 4}}

    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_execution_fingerprint_binds_every_declared_component(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "model.py").write_text("VALUE = 1\n", encoding="ascii")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"revision": "r1"}) + "\n", encoding="ascii")
    config = tmp_path / "config.toml"
    config.write_text("[solver]\ndt = 0.1\n", encoding="ascii")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("jax==0.10.2\n", encoding="ascii")
    python_version = tmp_path / ".python-version"
    python_version.write_text("3.12.13\n", encoding="ascii")
    backend_probe = tmp_path / "run_backend_probe.py"
    backend_probe.write_text("PROBE = 1\n", encoding="ascii")

    common = dict(
        source_root=source,
        manifest_path=manifest,
        config_path=config,
        requirements_path=requirements,
        python_version_path=python_version,
        backend_probe_path=backend_probe,
        probe_case_count=2,
        probe_definition_schema="probe-v2|case-count=2",
        probe_definition_sha256="d" * 64,
        observable_schema="physics-observables-v1",
        optimizer={"starts": 2, "maxiter": 4},
        seed_policy="sha256(base_seed,task_id)-v1",
    )
    baseline = build_execution_fingerprint(**common)

    assert baseline["fingerprint_sha256"] == build_execution_fingerprint(**common)[
        "fingerprint_sha256"
    ]
    for key, replacement in (
        ("observable_schema", "physics-observables-v2"),
        ("optimizer", {"starts": 3, "maxiter": 4}),
        ("seed_policy", "different"),
        ("probe_case_count", 3),
        ("probe_definition_schema", "probe-v3|case-count=2"),
        ("probe_definition_sha256", "e" * 64),
    ):
        changed = build_execution_fingerprint(**{**common, key: replacement})
        assert changed["fingerprint_sha256"] != baseline["fingerprint_sha256"]

    (source / "model.py").write_text("VALUE = 2\n", encoding="ascii")
    changed_source = build_execution_fingerprint(**common)
    assert changed_source["fingerprint_sha256"] != baseline["fingerprint_sha256"]

    python_version.write_text("3.13.0\n", encoding="ascii")
    changed_python = build_execution_fingerprint(**common)
    assert changed_python["fingerprint_sha256"] != baseline["fingerprint_sha256"]
    assert changed_python["canonical_environment_sha256"] != baseline[
        "canonical_environment_sha256"
    ]

    backend_probe.write_text("PROBE = 2\n", encoding="ascii")
    changed_probe = build_execution_fingerprint(**common)
    assert changed_probe["fingerprint_sha256"] != baseline["fingerprint_sha256"]
    assert changed_probe["backend_probe_sha256"] != baseline["backend_probe_sha256"]


def test_canonical_environment_requires_exact_pins(tmp_path):
    python_version = tmp_path / ".python-version"
    python_version.write_text("3.12.13\n", encoding="ascii")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("jax>=0.10.2\n", encoding="ascii")

    with pytest.raises(ValueError, match="exact name==version pins"):
        load_canonical_environment(python_version, requirements)


def test_canonical_environment_validation_checks_python_and_packages(tmp_path):
    python_version = tmp_path / ".python-version"
    python_version.write_text("3.12.13\n", encoding="ascii")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "jax==0.10.2\nimageio-ffmpeg==0.6.0\n",
        encoding="ascii",
    )
    installed = {"jax": "0.10.2", "imageio-ffmpeg": "0.6.0"}

    declaration = validate_canonical_environment(
        python_version,
        requirements,
        python_version="3.12.13",
        package_version=installed.__getitem__,
    )

    assert declaration == {
        "python_version": "3.12.13",
        "dependencies": {"imageio-ffmpeg": "0.6.0", "jax": "0.10.2"},
    }


def test_canonical_environment_validation_reports_all_mismatches(tmp_path):
    python_version = tmp_path / ".python-version"
    python_version.write_text("3.12.13\n", encoding="ascii")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("jax==0.10.2\nnumpy==2.5.1\n", encoding="ascii")
    installed = {"jax": "0.9.0"}

    with pytest.raises(RuntimeError) as caught:
        validate_canonical_environment(
            python_version,
            requirements,
            python_version="3.12.3",
            package_version=installed.__getitem__,
        )

    message = str(caught.value)
    assert "Python: expected 3.12.13, found 3.12.3" in message
    assert "jax: expected 0.10.2, found 0.9.0" in message
    assert "numpy: expected 2.5.1, not installed" in message


def test_worker_core_allocation_is_disjoint_and_leaves_requested_headroom():
    groups = allocate_worker_cores(tuple(range(14)), workers=2, cores_per_worker=6)

    assert groups == ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11))
    assert set(groups[0]).isdisjoint(groups[1])


def test_worker_core_allocation_rejects_oversubscription():
    try:
        allocate_worker_cores(tuple(range(8)), workers=2, cores_per_worker=6)
    except ValueError as error:
        assert "requires 12 cores" in str(error)
    else:
        raise AssertionError("oversubscribed worker allocation was accepted")


def test_execution_fingerprint_verification_recomputes_actual_inputs(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "model.py").write_text("VALUE = 1\n", encoding="ascii")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"revision":"r1"}\n', encoding="ascii")
    config = tmp_path / "config.toml"
    config.write_text("[solver]\ndt=0.1\n", encoding="ascii")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("jax==0.10.2\n", encoding="ascii")
    python_version = tmp_path / ".python-version"
    python_version.write_text("3.12.13\n", encoding="ascii")
    backend_probe = tmp_path / "run_backend_probe.py"
    backend_probe.write_text("PROBE = 1\n", encoding="ascii")
    fingerprint = build_execution_fingerprint(
        source_root=source,
        manifest_path=manifest,
        config_path=config,
        requirements_path=requirements,
        python_version_path=python_version,
        backend_probe_path=backend_probe,
        probe_case_count=2,
        probe_definition_schema="probe-v2|case-count=2",
        probe_definition_sha256="d" * 64,
        observable_schema="physics-observables-v1",
        optimizer={"starts": 2, "maxiter": 4},
        seed_policy="sha256(base_seed,task_id)-v1",
    )
    record = {
        "execution": {
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "canonical_environment_sha256": fingerprint[
                "canonical_environment_sha256"
            ],
        },
        "fingerprint": fingerprint,
    }

    verified = verify_execution_fingerprint(
        record,
        source_root=source,
        manifest_path=manifest,
        config_path=config,
        requirements_path=requirements,
        python_version_path=python_version,
        backend_probe_path=backend_probe,
        expected_probe_case_count=2,
        expected_probe_definition_schema="probe-v2|case-count=2",
        expected_probe_definition_sha256="d" * 64,
    )
    assert verified == fingerprint["fingerprint_sha256"]

    (source / "model.py").write_text("VALUE = 2\n", encoding="ascii")
    with pytest.raises(ValueError, match="does not match actual inputs"):
        verify_execution_fingerprint(
            record,
            source_root=source,
            manifest_path=manifest,
            config_path=config,
            requirements_path=requirements,
            python_version_path=python_version,
            backend_probe_path=backend_probe,
            expected_probe_case_count=2,
            expected_probe_definition_schema="probe-v2|case-count=2",
            expected_probe_definition_sha256="d" * 64,
        )


def test_execution_fingerprint_verification_rejects_nonfrozen_probe_contract(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "model.py").write_text("VALUE = 1\n", encoding="ascii")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"revision":"r1"}\n', encoding="ascii")
    config = tmp_path / "config.toml"
    config.write_text("[solver]\ndt=0.1\n", encoding="ascii")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("jax==0.10.2\n", encoding="ascii")
    python_version = tmp_path / ".python-version"
    python_version.write_text("3.12.13\n", encoding="ascii")
    backend_probe = tmp_path / "run_backend_probe.py"
    backend_probe.write_text("PROBE = 1\n", encoding="ascii")
    fingerprint = build_execution_fingerprint(
        source_root=source,
        manifest_path=manifest,
        config_path=config,
        requirements_path=requirements,
        python_version_path=python_version,
        backend_probe_path=backend_probe,
        probe_case_count=3,
        probe_definition_schema="probe-v2|case-count=3",
        probe_definition_sha256="e" * 64,
        observable_schema="physics-observables-v1",
        optimizer={"starts": 2, "maxiter": 4},
        seed_policy="sha256(base_seed,task_id)-v1",
    )
    record = {
        "execution": {
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "canonical_environment_sha256": fingerprint[
                "canonical_environment_sha256"
            ],
        },
        "fingerprint": fingerprint,
    }

    with pytest.raises(ValueError, match="probe contract"):
        verify_execution_fingerprint(
            record,
            source_root=source,
            manifest_path=manifest,
            config_path=config,
            requirements_path=requirements,
            python_version_path=python_version,
            backend_probe_path=backend_probe,
            expected_probe_case_count=2,
            expected_probe_definition_schema="probe-v2|case-count=2",
            expected_probe_definition_sha256="d" * 64,
        )


def test_backend_probe_payload_bundles_exact_inputs_without_observations(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.json"
    manifest_payload = {
        "metadata": {"config_path": "configs/transition.toml"},
        "records": [
            {
                "case_id": f"case_{index}",
                "split": "development",
                "noise_fraction": 0.0,
                "observation_path": f"missing/case_{index}.npz",
            }
            for index in range(3)
        ],
    }
    manifest.write_text(
        json.dumps(manifest_payload, indent=2) + "\n",
        encoding="ascii",
    )
    fingerprint = tmp_path / "execution.json"
    fingerprint.write_text('{"fingerprint":"exact"}\n', encoding="ascii")
    output = tmp_path / "payload.py"

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_backend_probe_payload.py"),
            "--manifest",
            str(manifest),
            "--fingerprint",
            str(fingerprint),
            "--max-cases",
            "2",
            "--out",
            str(output),
        ],
        check=True,
        cwd=project_root,
    )

    payload = output.read_text(encoding="ascii")
    encoded = re.search(r'ARCHIVE_B64 = "([A-Za-z0-9+/=]+)"', payload).group(1)
    with tarfile.open(
        fileobj=io.BytesIO(base64.b64decode(encoded)), mode="r:gz"
    ) as archive:
        names = set(archive.getnames())
        bundled_manifest = archive.extractfile("benchmark/manifest.json").read()
    assert bundled_manifest == manifest.read_bytes()
    assert ".python-version" in names
    assert "requirements/canonical-cpu.txt" in names
    assert "scripts/run_backend_probe.py" in names
    assert not any(name.endswith(".npz") for name in names)
    assert '"--max-cases", "2"' in payload
    assert '"--analytic-targets"' in payload
