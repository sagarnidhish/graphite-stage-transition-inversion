"""Stable seeds and provenance fingerprints for canonical benchmark execution."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
from typing import Any, Callable, Mapping, Sequence


SEED_POLICY = "sha256(base_seed,task_id)-v1"
OBSERVABLE_SCHEMA = "physics-observables-v1"
_EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")


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


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_canonical_environment(
    python_version_path: Path,
    requirements_path: Path,
) -> dict[str, Any]:
    """Load the exact Python and direct-dependency declaration."""

    python_version = Path(python_version_path).read_text(encoding="ascii").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", python_version):
        raise ValueError("canonical Python version must be an exact X.Y.Z version")

    dependencies: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        Path(requirements_path).read_text(encoding="ascii").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_REQUIREMENT.fullmatch(line)
        if match is None:
            raise ValueError(
                "canonical requirements must contain only exact name==version pins "
                f"(line {line_number}: {line!r})"
            )
        name = _canonical_package_name(match.group(1))
        if name in dependencies:
            raise ValueError(f"duplicate canonical requirement {name!r}")
        dependencies[name] = match.group(2)
    if not dependencies:
        raise ValueError("canonical requirements must contain at least one dependency")
    return {
        "python_version": python_version,
        "dependencies": dict(sorted(dependencies.items())),
    }


def validate_canonical_environment(
    python_version_path: Path,
    requirements_path: Path,
    *,
    python_version: str | None = None,
    package_version: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Fail unless the active interpreter exactly matches the declaration."""

    declaration = load_canonical_environment(python_version_path, requirements_path)
    actual_python = platform.python_version() if python_version is None else python_version
    get_version = metadata.version if package_version is None else package_version
    mismatches = []
    if actual_python != declaration["python_version"]:
        mismatches.append(
            f"Python: expected {declaration['python_version']}, found {actual_python}"
        )
    for name, expected in declaration["dependencies"].items():
        try:
            actual = get_version(name)
        except (metadata.PackageNotFoundError, KeyError):
            mismatches.append(f"{name}: expected {expected}, not installed")
            continue
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, found {actual}")
    if mismatches:
        raise RuntimeError("canonical environment mismatch:\n- " + "\n- ".join(mismatches))
    return declaration


def build_execution_fingerprint(
    *,
    source_root: Path,
    manifest_path: Path,
    config_path: Path,
    requirements_path: Path,
    python_version_path: Path,
    observable_schema: str,
    optimizer: Mapping[str, Any],
    seed_policy: str,
) -> dict[str, Any]:
    """Build a self-verifying record of every claim-relevant execution input."""

    environment = load_canonical_environment(python_version_path, requirements_path)
    components: dict[str, Any] = {
        "source_sha256": source_tree_sha256(source_root),
        "manifest_sha256": file_sha256(manifest_path),
        "config_sha256": file_sha256(config_path),
        "requirements_sha256": file_sha256(requirements_path),
        "canonical_environment": environment,
        "canonical_environment_sha256": canonical_json_sha256(environment),
        "observable_schema": str(observable_schema),
        "optimizer": dict(optimizer),
        "seed_policy": str(seed_policy),
    }
    return {**components, "fingerprint_sha256": canonical_json_sha256(components)}


def verify_execution_fingerprint(
    record: Mapping[str, Any],
    *,
    source_root: Path,
    manifest_path: Path,
    config_path: Path,
    requirements_path: Path,
    python_version_path: Path,
) -> str:
    """Recompute a persisted fingerprint from the probe's actual local inputs."""

    try:
        declared = dict(record["fingerprint"])
        execution = dict(record["execution"])
        declared_digest = declared.pop("fingerprint_sha256")
        observable_schema = declared["observable_schema"]
        optimizer = declared["optimizer"]
        seed_policy = declared["seed_policy"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid persisted execution fingerprint") from error
    if canonical_json_sha256(declared) != declared_digest:
        raise ValueError("persisted execution fingerprint is internally inconsistent")
    if execution.get("fingerprint_sha256") != declared_digest:
        raise ValueError("execution record does not match persisted fingerprint")
    if execution.get("canonical_environment_sha256") != declared.get(
        "canonical_environment_sha256"
    ):
        raise ValueError("execution record does not match canonical environment")

    actual = build_execution_fingerprint(
        source_root=source_root,
        manifest_path=manifest_path,
        config_path=config_path,
        requirements_path=requirements_path,
        python_version_path=python_version_path,
        observable_schema=observable_schema,
        optimizer=optimizer,
        seed_policy=seed_policy,
    )
    if actual != {**declared, "fingerprint_sha256": declared_digest}:
        raise ValueError("persisted execution fingerprint does not match actual inputs")
    return str(declared_digest)
