"""Resumable, checksum-verified benchmark task orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import jax.numpy as jnp
import numpy as np

from .baselines import mean_only_loss, simulate_fickian, simulate_sharp_interface, spatial_loss
from .config import load_config
from .geometry import make_circle_grid
from .inversion import InverseProblem, ParameterTransform, fit_multistart, generate_starts
from .protocols import build_protocol
from .solver import CHRParameters


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    record_index: int
    case_id: str
    replicate: int
    noise_fraction: float
    split: str
    method: str


def _task_id(record: dict, method: str) -> str:
    identity = {
        "case_id": record["case_id"],
        "replicate": int(record["replicate"]),
        "noise_fraction": float(record["noise_fraction"]),
        "method": method,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "task_" + hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]


def build_task_table(manifest: dict, methods=("chr",)) -> list[BenchmarkTask]:
    """Create stable method-specific tasks without splitting parameter cases."""

    allowed = {"chr", "fickian", "sharp_interface"}
    if not methods or not set(methods) <= allowed:
        raise ValueError(f"methods must be drawn from {sorted(allowed)}")
    tasks = []
    for record_index, record in enumerate(manifest["records"]):
        source_record_index = int(record.get("_manifest_index", record_index))
        for method in methods:
            tasks.append(
                BenchmarkTask(
                    _task_id(record, method),
                    source_record_index,
                    record["case_id"],
                    int(record["replicate"]),
                    float(record["noise_fraction"]),
                    record["split"],
                    method,
                )
            )
    return tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_success_marker(
    output_root: Path,
    task: BenchmarkTask,
    checksum: str | None = None,
) -> None:
    """Test/helper API that creates a checksum-bearing completed task."""

    directory = Path(output_root) / "tasks" / task.task_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "task.json").write_text(
        json.dumps(asdict(task), sort_keys=True) + "\n", encoding="ascii"
    )
    result_path = directory / "result.json"
    if not result_path.exists():
        result_path.write_text(
            json.dumps({"task_id": task.task_id, "status": "success"}, sort_keys=True) + "\n",
            encoding="ascii",
        )
    actual = _sha256(result_path)
    marker_checksum = actual if checksum in (None, "valid") else checksum
    (directory / "success.json").write_text(
        json.dumps({"task_id": task.task_id, "result_sha256": marker_checksum}, sort_keys=True)
        + "\n",
        encoding="ascii",
    )


def _verified_success(output_root: Path, task: BenchmarkTask) -> bool:
    directory = Path(output_root) / "tasks" / task.task_id
    marker_path = directory / "success.json"
    result_path = directory / "result.json"
    task_path = directory / "task.json"
    if not (marker_path.is_file() and result_path.is_file() and task_path.is_file()):
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="ascii"))
        stored_task = json.loads(task_path.read_text(encoding="ascii"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        marker.get("task_id") == task.task_id
        and stored_task == asdict(task)
        and marker.get("result_sha256") == _sha256(result_path)
    )


def resume_benchmark(tasks, output_root: Path) -> list[BenchmarkTask]:
    return [task for task in tasks if not _verified_success(output_root, task)]


def aggregate_task_status(tasks, output_root: Path) -> dict[str, int]:
    counts = {"success": 0, "failed": 0, "pending": 0}
    for task in tasks:
        if _verified_success(output_root, task):
            counts["success"] += 1
        elif (Path(output_root) / "failures" / f"{task.task_id}.json").is_file():
            counts["failed"] += 1
        else:
            counts["pending"] += 1
    return counts


def _resolve_config_path(manifest_path: Path, configured: str) -> Path:
    candidate = Path(configured)
    for path in (
        candidate,
        manifest_path.parent / candidate,
        manifest_path.parent.parent / candidate,
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"cannot resolve benchmark config {configured}")


def _load_problem(manifest_path: Path, task: BenchmarkTask):
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    record = manifest["records"][task.record_index]
    config = load_config(_resolve_config_path(manifest_path, manifest["metadata"]["config_path"]))
    archive = np.load(manifest_path.parent / record["observation_path"])
    observations = jnp.asarray(archive["concentration"])
    grid = make_circle_grid(config.grid)
    protocol = build_protocol(config.protocol, config.solver.dt)
    names = ("mobility", "barrier", "kappa", "reaction_rate")
    bounds = manifest["metadata"]["bounds"]
    transform = ParameterTransform(
        tuple(bounds[name][0] for name in names),
        tuple(bounds[name][1] for name in names),
        config.model.stage2,
        config.model.stage1,
    )
    solver = replace(config.solver, perturbation_amplitude=0.0, seed=int(record["seed"]))
    problem = InverseProblem(
        grid,
        protocol,
        solver,
        observations,
        observations[0],
        transform,
        config.inversion.mass_penalty,
        config.inversion.bound_penalty,
    )
    return manifest, record, config, problem


def _execute_task(
    task: BenchmarkTask,
    manifest_path: Path,
    starts: int,
    maxiter: int,
    seed: int,
) -> dict:
    manifest, record, config, problem = _load_problem(manifest_path, task)
    if task.method == "chr":
        central = CHRParameters(
            config.model.mobility,
            config.model.barrier,
            config.model.kappa,
            config.model.reaction_rate,
            config.model.stage2,
            config.model.stage1,
        )
        start_values = generate_starts(problem.transform, central, starts, seed)
        fit = fit_multistart(problem, start_values, maxiter)
        return {
            "method": task.method,
            "best": asdict(fit.best),
            "starts": [asdict(result) for result in fit.starts],
            "truth_parameters": record.get("parameters"),
            "truth_groups": record.get("groups"),
        }

    observations = np.asarray(problem.observations)
    if task.method == "fickian":
        candidates = np.geomspace(0.002, 2.0, max(8, starts * 4))
        rows = []
        width_squared = (config.model.stage1 - config.model.stage2) ** 2
        for diffusivity in candidates:
            result = simulate_fickian(
                problem.grid,
                problem.protocol,
                float(diffusivity),
                problem.solver,
                problem.initial_concentration,
            )
            raw = spatial_loss(result.concentration, observations, problem.grid.mask)
            rows.append(
                {
                    "diffusivity": float(diffusivity),
                    "loss": raw / width_squared,
                    "mean_only_loss": mean_only_loss(
                        result.concentration, observations, problem.grid.mask
                    ),
                }
            )
        return {"method": task.method, "best": min(rows, key=lambda row: row["loss"]), "all": rows}

    initial_mean = float(observations[0, np.asarray(problem.grid.mask)].mean())
    result = simulate_sharp_interface(
        problem.grid,
        problem.protocol,
        initial_mean,
        config.model.stage2,
        config.model.stage1,
    )
    raw = spatial_loss(result.concentration, observations, problem.grid.mask)
    return {
        "method": task.method,
        "best": {
            "loss": raw / (config.model.stage1 - config.model.stage2) ** 2,
            "mean_only_loss": mean_only_loss(result.concentration, observations, problem.grid.mask),
        },
    }


def run_task(
    task: BenchmarkTask,
    manifest_path: Path,
    output_root: Path,
    starts: int,
    maxiter: int,
    seed: int,
) -> bool:
    """Run one task atomically and retain explicit failure evidence."""

    output_root = Path(output_root)
    tasks_root = output_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    temporary = tasks_root / f".{task.task_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    started = time.perf_counter()
    try:
        result = _execute_task(task, Path(manifest_path), starts, maxiter, seed)
        result["task_id"] = task.task_id
        result["status"] = "success"
        result["runtime_seconds"] = time.perf_counter() - started
        (temporary / "task.json").write_text(
            json.dumps(asdict(task), sort_keys=True) + "\n", encoding="ascii"
        )
        result_path = temporary / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="ascii",
        )
        (temporary / "success.json").write_text(
            json.dumps(
                {"task_id": task.task_id, "result_sha256": _sha256(result_path)},
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        destination = tasks_root / task.task_id
        if destination.exists():
            destination.rename(tasks_root / f".{task.task_id}.stale-{uuid.uuid4().hex}")
        temporary.rename(destination)
        return True
    except Exception as error:
        failures = output_root / "failures"
        failures.mkdir(parents=True, exist_ok=True)
        failure = {
            "task": asdict(task),
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "seed": seed,
            "starts": starts,
            "maxiter": maxiter,
        }
        (failures / f"{task.task_id}.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        return False
