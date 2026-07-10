#!/usr/bin/env python3
"""Run or resume selected benchmark tasks with verified completion markers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path


def _worker_main(worker_index, core_ids, cache_root, payloads) -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["JAX_ENABLE_X64"] = "true"
    worker_cache = Path(cache_root) / f"worker_{worker_index:02d}"
    worker_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(worker_cache)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(core_ids))

    from graphite_stage_transition.benchmark import (
        BenchmarkExecution,
        BenchmarkTask,
        require_all_tasks_succeeded,
        run_task,
    )

    outcomes = []
    for payload in payloads:
        task = BenchmarkTask(**payload["task"])
        execution = BenchmarkExecution(**payload["execution"])
        print(f"[{worker_index}] {task.task_id} {task.method}", flush=True)
        outcomes.append(
            (
                task.task_id,
                run_task(
                    task,
                    Path(payload["manifest_path"]),
                    Path(payload["output_root"]),
                    starts=execution.starts,
                    maxiter=execution.maxiter,
                    execution=execution,
                ),
            )
        )
    require_all_tasks_succeeded(outcomes)


def _run_spawned(payloads, cache_root, workers, cores_per_worker) -> None:
    if not payloads:
        return
    from graphite_stage_transition.execution import allocate_worker_cores

    available = (
        tuple(sorted(os.sched_getaffinity(0)))
        if hasattr(os, "sched_getaffinity")
        else tuple(range(os.cpu_count() or 1))
    )
    active_workers = min(int(workers), len(payloads))
    core_groups = allocate_worker_cores(
        available,
        workers=active_workers,
        cores_per_worker=cores_per_worker,
    )
    batches = tuple(tuple(payloads[index::active_workers]) for index in range(active_workers))
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_worker_main,
            args=(index, core_groups[index], str(cache_root), batches[index]),
        )
        for index in range(active_workers)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"spawned benchmark workers failed with exit codes {failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=["chr"])
    parser.add_argument("--splits", nargs="+", default=["development"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cores-per-worker", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=20260710)
    parser.add_argument("--backend-gate", type=Path)
    parser.add_argument("--backend-probes", nargs="+", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    python_version_path = project_root / ".python-version"
    requirements_path = project_root / "requirements" / "canonical-cpu.txt"
    from graphite_stage_transition.execution import validate_canonical_environment

    validate_canonical_environment(python_version_path, requirements_path)

    # These imports initialize JAX in the parent only. Spawn workers import this
    # script without entering main, then set affinity before importing the model.
    from graphite_stage_transition.benchmark import (
        BenchmarkExecution,
        aggregate_task_status,
        build_task_table,
        resume_benchmark,
    )
    from graphite_stage_transition.execution import (
        OBSERVABLE_SCHEMA,
        SEED_POLICY,
        build_execution_fingerprint,
    )
    from graphite_stage_transition.backend_gate import (
        PROBE_CASE_COUNT,
        PROBE_DEFINITION_SCHEMA,
        PROBE_DEFINITION_SHA256,
    )

    manifest = json.loads(args.manifest.read_text(encoding="ascii"))
    case_ids = []
    selected_records = []
    for record_index, record in enumerate(manifest["records"]):
        if record["split"] not in args.splits or float(record["noise_fraction"]) != 0.0:
            continue
        if record["case_id"] not in case_ids:
            case_ids.append(record["case_id"])
        if record["case_id"] in case_ids[: args.max_cases]:
            selected_records.append({**record, "_manifest_index": record_index})
        if len(case_ids) > args.max_cases:
            break
    selected_manifest = dict(manifest)
    selected_manifest["records"] = selected_records
    tasks = build_task_table(selected_manifest, tuple(args.methods))

    configured = Path(manifest["metadata"]["config_path"])
    config_candidates = (
        configured,
        args.manifest.parent / configured,
        args.manifest.parent.parent / configured,
        project_root / configured,
    )
    try:
        config_path = next(path for path in config_candidates if path.is_file())
    except StopIteration as error:
        raise FileNotFoundError(f"cannot resolve benchmark config {configured}") from error
    fingerprint = build_execution_fingerprint(
        source_root=project_root / "src" / "graphite_stage_transition",
        manifest_path=args.manifest,
        config_path=config_path,
        requirements_path=requirements_path,
        python_version_path=python_version_path,
        backend_probe_path=project_root / "scripts" / "run_backend_probe.py",
        probe_case_count=PROBE_CASE_COUNT,
        probe_definition_schema=PROBE_DEFINITION_SCHEMA,
        probe_definition_sha256=PROBE_DEFINITION_SHA256,
        observable_schema=OBSERVABLE_SCHEMA,
        optimizer={"starts": args.starts, "maxiter": args.maxiter},
        seed_policy=SEED_POLICY,
    )
    claim_eligible = False
    if "chr" in args.methods:
        if args.backend_gate is None:
            if any(split != "development" for split in args.splits):
                parser.error(
                    "CHR validation/test execution requires a matching passed --backend-gate"
                )
        else:
            if not args.backend_probes:
                parser.error("--backend-gate requires --backend-probes evidence files")
            from graphite_stage_transition.backend_gate import (
                load_backend_gate_result,
                require_matching_passed_gate,
            )

            gate = load_backend_gate_result(args.backend_gate)
            require_matching_passed_gate(
                gate,
                fingerprint["fingerprint_sha256"],
                args.backend_probes,
            )
            claim_eligible = True
    execution = BenchmarkExecution(
        fingerprint_sha256=fingerprint["fingerprint_sha256"],
        canonical_environment_sha256=fingerprint["canonical_environment_sha256"],
        base_seed=args.base_seed,
        starts=args.starts,
        maxiter=args.maxiter,
        claim_eligible=claim_eligible,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "execution.json").write_text(
        json.dumps(
            {"execution": asdict(execution), "fingerprint": fingerprint},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    (args.out / "plan.json").write_text(
        json.dumps([asdict(task) for task in tasks], indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    if args.plan_only:
        print(json.dumps({"fingerprint_sha256": fingerprint["fingerprint_sha256"]}))
        return

    pending = resume_benchmark(tasks, args.out, execution=execution)
    payloads = [
        {
            "task": asdict(task),
            "execution": asdict(execution),
            "manifest_path": str(args.manifest.resolve()),
            "output_root": str(args.out.resolve()),
        }
        for task in pending
    ]
    _run_spawned(
        payloads,
        args.out / ".worker_cache",
        args.workers,
        args.cores_per_worker,
    )
    status = aggregate_task_status(tasks, args.out, execution=execution)
    (args.out / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
