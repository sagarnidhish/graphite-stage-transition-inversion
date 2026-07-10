import json
from pathlib import Path
import base64
import io
import re
import tarfile

from graphite_stage_transition.benchmark import (
    aggregate_task_status,
    build_task_table,
    resume_benchmark,
    write_success_marker,
)
from graphite_stage_transition.gpu_payload import build_gpu_payload


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
    assert "kgpu_graphite_results.tar.gz" in text
    assert "graphite_stage_transition" not in output.name
    encoded = re.search(r'ARCHIVE_B64 = "([A-Za-z0-9+/=]+)"', text).group(1)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)), mode="r:gz") as archive:
        embedded = json.load(archive.extractfile("benchmark/manifest.json"))
    assert {record["split"] for record in embedded["records"]} == {"development"}
