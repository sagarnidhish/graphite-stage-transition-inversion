"""Stable seeds and provenance fingerprints for canonical benchmark execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SEED_POLICY = "sha256(base_seed,task_id)-v1"
OBSERVABLE_SCHEMA = "physics-observables-v1"


def allocate_worker_cores(
    available_cpus: Sequence[int],
    *,
    workers: int,
    cores_per_worker: int,
) -> tuple[tuple[int, ...], ...]:
    """Assign deterministic disjoint CPU sets without oversubscription."""

    cpus = tuple(int(cpu) for cpu in available_cpus)
    required = int(workers) * int(cores_per_worker)
    if workers < 1 or cores_per_worker < 1:
        raise ValueError("workers and cores_per_worker must be positive")
    if len(cpus) < required:
        raise ValueError(f"worker allocation requires {required} cores, found {len(cpus)}")
    return tuple(
        cpus[index * cores_per_worker : (index + 1) * cores_per_worker]
        for index in range(workers)
    )


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON-compatible data with stable mapping and whitespace semantics."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def stable_task_seed(base_seed: int, task_id: str) -> int:
    """Derive a JAX-compatible nonnegative seed from immutable task identity."""

    identity = {"base_seed": int(base_seed), "task_id": str(task_id), "policy": SEED_POLICY}
    digest = bytes.fromhex(canonical_json_sha256(identity))
    return int.from_bytes(digest[:8], "big") % (2**31)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(source_root: Path) -> str:
    """Hash relative paths and bytes for all Python source files in a tree."""

    root = Path(source_root)
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_execution_fingerprint(
    *,
    source_root: Path,
    manifest_path: Path,
    config_path: Path,
    requirements_path: Path,
    observable_schema: str,
    optimizer: Mapping[str, Any],
    seed_policy: str,
) -> dict[str, Any]:
    """Build a self-verifying record of every claim-relevant execution input."""

    components: dict[str, Any] = {
        "source_sha256": source_tree_sha256(source_root),
        "manifest_sha256": file_sha256(manifest_path),
        "config_sha256": file_sha256(config_path),
        "requirements_sha256": file_sha256(requirements_path),
        "observable_schema": str(observable_schema),
        "optimizer": dict(optimizer),
        "seed_policy": str(seed_policy),
    }
    return {**components, "fingerprint_sha256": canonical_json_sha256(components)}
