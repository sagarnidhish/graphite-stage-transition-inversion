import json
from pathlib import Path
import base64
import io
import re
import tarfile

import pytest

from graphite_stage_transition.benchmark import (
    BenchmarkExecution,
    aggregate_task_status,
    build_task_table,
    require_all_tasks_succeeded,
    resume_benchmark,
    run_task,
    write_success_marker,
)
from graphite_stage_transition.gpu_payload import (
    build_gpu_payload,
    build_public_gpu_bootstrap,
)


def _smoke_manifest():
    return {
        "metadata": {"config_path": "configs/transition.toml"},
        "records": [
            {
                "case_id": "case_a",
                "replicate": 0,
                "noise_fraction": 0.0,
                "split": "development",
                "observation_path": "case_a/replicate_00/observation_noise_000.npz",
                "clean_path": "case_a/replicate_00/clean.npz",
            },
            {
                "case_id": "case_b",
                "replicate": 0,
                "noise_fraction": 0.1,
                "split": "validation",
                "observation_path": "case_b/replicate_00/observation_noise_100.npz",
                "clean_path": "case_b/replicate_00/clean.npz",
            },
        ],
    }


def test_task_ids_are_stable_and_method_specific():
    first = build_task_table(_smoke_manifest(), methods=("chr", "fickian"))
    second = build_task_table(_smoke_manifest(), methods=("chr", "fickian"))

    assert first == second
    assert len({task.task_id for task in first}) == len(first) == 4


def test_resume_skips_only_verified_success(tmp_path: Path):
    table = build_task_table(_smoke_manifest(), methods=("chr", "fickian"))
    write_success_marker(tmp_path, table[0], checksum="valid")
    write_success_marker(tmp_path, table[1], checksum="invalid")

    pending = resume_benchmark(table, tmp_path)

    assert table[0] not in pending
    assert table[1] in pending
    assert len(pending) == len(table) - 1
    status = aggregate_task_status(table, tmp_path)
    assert status["success"] == 1
    assert status["pending"] == len(table) - 1


def test_resume_rejects_marker_from_different_execution(tmp_path: Path):
    task = build_task_table(_smoke_manifest(), methods=("chr",))[0]
    original = BenchmarkExecution(
        fingerprint_sha256="a" * 64,
        canonical_environment_sha256="c" * 64,
        base_seed=20260710,
        starts=2,
        maxiter=4,
        claim_eligible=False,
    )
    write_success_marker(tmp_path, task, execution=original)

    assert resume_benchmark([task], tmp_path, execution=original) == []
    for changed in (
        BenchmarkExecution("b" * 64, "c" * 64, 20260710, 2, 4, False),
        BenchmarkExecution("a" * 64, "d" * 64, 20260710, 2, 4, False),
        BenchmarkExecution("a" * 64, "c" * 64, 20260711, 2, 4, False),
        BenchmarkExecution("a" * 64, "c" * 64, 20260710, 3, 4, False),
        BenchmarkExecution("a" * 64, "c" * 64, 20260710, 2, 5, False),
        BenchmarkExecution("a" * 64, "c" * 64, 20260710, 2, 4, True),
    ):
        assert resume_benchmark([task], tmp_path, execution=changed) == [task]


def test_task_result_stamps_claim_eligibility(tmp_path, monkeypatch):
    task = build_task_table(_smoke_manifest(), methods=("chr",))[0]
    execution = BenchmarkExecution("a" * 64, "c" * 64, 20260710, 1, 1, False)
    monkeypatch.setattr(
        "graphite_stage_transition.benchmark._execute_task",
        lambda *_args, **_kwargs: {"method": "chr", "best": {"loss": 0.0}},
    )

    succeeded = run_task(
        task,
        tmp_path / "manifest.json",
        tmp_path,
        starts=1,
        maxiter=1,
        execution=execution,
    )

    result = json.loads(
        (tmp_path / "tasks" / task.task_id / "result.json").read_text(encoding="ascii")
    )
    assert succeeded
    assert result["claim_eligible"] is False


def test_worker_outcome_check_raises_if_any_task_failed():
    with pytest.raises(RuntimeError, match="task_b"):
        require_all_tasks_succeeded((("task_a", True), ("task_b", False)))


def test_worker_outcome_check_accepts_all_successes():
    require_all_tasks_succeeded((("task_a", True), ("task_b", True)))


def test_gpu_payload_embeds_sources_manifest_and_selected_data(tmp_path: Path):
    project = tmp_path / "project"
    (project / "src" / "graphite_stage_transition").mkdir(parents=True)
    (project / "src" / "graphite_stage_transition" / "__init__.py").write_text("\n")
    (project / "configs").mkdir()
    (project / "configs" / "transition.toml").write_text("[grid]\n")
    manifest_path = project / "manifest.json"
    manifest = _smoke_manifest()
    manifest["records"][0]["split"] = "test"
    manifest["records"][1]["split"] = "development"
    manifest["records"][1]["noise_fraction"] = 0.0
    for record in manifest["records"]:
        for key in ("observation_path", "clean_path"):
            path = project / record[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"npz")
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "payload.py"

    build_gpu_payload(project, manifest_path, output, max_cases=1, starts=1, maxiter=1)

    text = output.read_text()
    assert "ARCHIVE_B64" in text
    assert "stable_task_seed(20260710, task.task_id)" in text
    assert "kgpu_graphite_results.tar.gz" in text
    assert "graphite_stage_transition" not in output.name
    encoded = re.search(r'ARCHIVE_B64 = "([A-Za-z0-9+/=]+)"', text).group(1)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)), mode="r:gz") as archive:
        embedded = json.load(archive.extractfile("benchmark/manifest.json"))
    assert {record["split"] for record in embedded["records"]} == {"development"}


def test_public_gpu_bootstrap_contains_only_public_urls_and_hashes(tmp_path: Path):
    output = tmp_path / "public_probe.py"

    build_public_gpu_bootstrap(
        release_base="https://github.com/example/project/releases/download/v1",
        source_asset="source.tar.gz",
        source_sha256="a" * 64,
        data_asset="synthetic.tar.gz",
        data_sha256="b" * 64,
        output_script=output,
        max_cases=2,
        starts=1,
        maxiter=1,
    )

    text = output.read_text()
    assert "ARCHIVE_B64" not in text
    assert "https://github.com/example/project/releases/download/v1/source.tar.gz" in text
    assert "a" * 64 in text
    assert "development" in text
    assert "stable_task_seed(20260710, task.task_id)" in text
    assert "kgpu_graphite_results.tar.gz" in text
