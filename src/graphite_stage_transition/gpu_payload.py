"""Build and retrieve self-contained single-file kgpu benchmark payloads."""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile


def _selected_records(
    manifest: dict,
    max_cases: int,
    splits: tuple[str, ...] = ("development",),
) -> list[dict]:
    selected_ids = []
    for record in manifest["records"]:
        if record["split"] not in splits or float(record["noise_fraction"]) != 0.0:
            continue
        if record["case_id"] not in selected_ids:
            selected_ids.append(record["case_id"])
        if len(selected_ids) == max_cases:
            break
    return [
        record
        for record in manifest["records"]
        if record["case_id"] in selected_ids
        and record["split"] in splits
        and float(record["noise_fraction"]) == 0.0
    ]


def build_gpu_payload(
    project_root: Path,
    manifest_path: Path,
    output_script: Path,
    max_cases: int = 2,
    starts: int = 2,
    maxiter: int = 3,
) -> None:
    """Embed source, config, manifest, and selected arrays in one Python file."""

    project_root = Path(project_root)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    selected = _selected_records(manifest, max_cases)
    if not selected:
        raise ValueError("manifest contains no clean development records for the GPU payload")
    subset = deepcopy(manifest)
    subset["records"] = selected
    subset["metadata"]["config_path"] = "configs/transition.toml"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        source_root = project_root / "src" / "graphite_stage_transition"
        for path in sorted(source_root.glob("*.py")):
            archive.add(path, arcname=f"src/graphite_stage_transition/{path.name}")
        config_source = project_root / "configs" / "transition.toml"
        archive.add(config_source, arcname="configs/transition.toml")
        manifest_bytes = (json.dumps(subset, sort_keys=True) + "\n").encode("ascii")
        info = tarfile.TarInfo("benchmark/manifest.json")
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
        for record in selected:
            for key in ("observation_path", "clean_path"):
                source = manifest_path.parent / record[key]
                archive.add(source, arcname=f"benchmark/{record[key]}")
    archive_bytes = buffer.getvalue()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    encoded = base64.b64encode(archive_bytes).decode("ascii")
    script = f'''#!/usr/bin/env python3
import base64, io, json, os, pathlib, sys, tarfile
ARCHIVE_B64 = "{encoded}"
ARCHIVE_SHA256 = "{archive_sha256}"
ROOT = pathlib.Path("/kaggle/working/graphite_benchmark")
ROOT.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE_B64)), mode="r:gz") as archive:
    archive.extractall(ROOT)
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
import jax
from graphite_stage_transition.benchmark import build_task_table, run_task, aggregate_task_status
manifest_path = ROOT / "benchmark" / "manifest.json"
manifest = json.loads(manifest_path.read_text())
tasks = build_task_table(manifest, methods=("chr",))
output = ROOT / "results"
for index, task in enumerate(tasks, 1):
    print(f"TASK {{index}}/{{len(tasks)}} {{task.task_id}}", flush=True)
    run_task(task, manifest_path, output, starts={int(starts)}, maxiter={int(maxiter)}, seed=20260710 + index)
status = aggregate_task_status(tasks, output)
(output / "run_metadata.json").write_text(json.dumps({{"device": str(jax.devices()), "archive_sha256": ARCHIVE_SHA256, "starts": {int(starts)}, "maxiter": {int(maxiter)}}}, indent=2, sort_keys=True) + "\\n")
result_tar = pathlib.Path("/kaggle/working/kgpu_graphite_results.tar.gz")
with tarfile.open(result_tar, "w:gz") as archive:
    archive.add(output, arcname="results")
print(json.dumps({{"device": str(jax.devices()), "status": status, "artifact": str(result_tar), "archive_sha256": ARCHIVE_SHA256}}, sort_keys=True), flush=True)
'''
    output_script = Path(output_script)
    output_script.parent.mkdir(parents=True, exist_ok=True)
    output_script.write_text(script, encoding="ascii")


def retrieve_gpu_results(kernel_output_dir: Path, destination: Path) -> Path:
    """Extract and return the downloaded kgpu result archive."""

    archive_path = Path(kernel_output_dir) / "kgpu_graphite_results.tar.gz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination)
    return destination


def build_public_gpu_bootstrap(
    release_base: str,
    source_asset: str,
    source_sha256: str,
    data_asset: str,
    data_sha256: str,
    output_script: Path,
    max_cases: int = 2,
    starts: int = 1,
    maxiter: int = 1,
) -> None:
    """Build a kgpu job that downloads already-public, checksum-pinned assets."""

    if not release_base.startswith("https://"):
        raise ValueError("release_base must be an HTTPS URL")
    for name, digest in (("source", source_sha256), ("data", data_sha256)):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{name} SHA-256 must be 64 lowercase hexadecimal characters")
    if max_cases < 1 or starts < 1 or maxiter < 1:
        raise ValueError("max_cases, starts, and maxiter must be positive")
    source_url = release_base.rstrip("/") + "/" + source_asset
    data_url = release_base.rstrip("/") + "/" + data_asset
    script = f'''#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys, tarfile, time, urllib.request

ROOT = pathlib.Path("/kaggle/working/graphite_public_probe")
ROOT.mkdir(parents=True, exist_ok=True)
ASSETS = (
    ("{source_url}", "{source_asset}", "{source_sha256}"),
    ("{data_url}", "{data_asset}", "{data_sha256}"),
)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for url, name, expected in ASSETS:
    destination = ROOT / name
    print(f"DOWNLOAD {{url}}", flush=True)
    urllib.request.urlretrieve(url, destination)
    actual = sha256(destination)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {{name}}: {{actual}} != {{expected}}")
    with tarfile.open(destination, "r:gz") as archive:
        archive.extractall(ROOT)

PROJECT = ROOT / "graphite_stage_transition_inversion"
DATA = ROOT / "benchmark_stage16_dt000125"
sys.path.insert(0, str(PROJECT / "src"))
os.chdir(PROJECT)

import jax
from graphite_stage_transition.benchmark import (
    aggregate_task_status,
    build_task_table,
    run_task,
)

manifest_path = DATA / "manifest.json"
manifest = json.loads(manifest_path.read_text())
selected_ids = []
selected_records = []
for record_index, record in enumerate(manifest["records"]):
    if record["split"] != "development" or float(record["noise_fraction"]) != 0.0:
        continue
    if record["case_id"] not in selected_ids:
        selected_ids.append(record["case_id"])
    if record["case_id"] in selected_ids[:{int(max_cases)}]:
        selected_records.append({{**record, "_manifest_index": record_index}})
    if len(selected_ids) > {int(max_cases)}:
        break

selected_manifest = dict(manifest)
selected_manifest["records"] = selected_records
tasks = build_task_table(selected_manifest, methods=("chr",))
output = ROOT / "results"
started = time.perf_counter()
for index, task in enumerate(tasks, 1):
    print(f"TASK {{index}}/{{len(tasks)}} {{task.task_id}} case={{task.case_id}}", flush=True)
    run_task(
        task,
        manifest_path,
        output,
        starts={int(starts)},
        maxiter={int(maxiter)},
        seed=20260710 + task.record_index,
    )
status = aggregate_task_status(tasks, output)
metadata = {{
    "device": [str(device) for device in jax.devices()],
    "status": status,
    "runtime_seconds": time.perf_counter() - started,
    "source_url": ASSETS[0][0],
    "source_sha256": ASSETS[0][2],
    "data_url": ASSETS[1][0],
    "data_sha256": ASSETS[1][2],
    "max_cases": {int(max_cases)},
    "starts": {int(starts)},
    "maxiter": {int(maxiter)},
}}
(output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\\n")
result_tar = pathlib.Path("/kaggle/working/kgpu_graphite_results.tar.gz")
with tarfile.open(result_tar, "w:gz") as archive:
    archive.add(output, arcname="results")
print(json.dumps({{**metadata, "artifact": str(result_tar)}}, sort_keys=True), flush=True)
'''
    output_script = Path(output_script)
    output_script.parent.mkdir(parents=True, exist_ok=True)
    output_script.write_text(script, encoding="ascii")
