#!/usr/bin/env python3
"""Run or resume selected benchmark tasks with verified completion markers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphite_stage_transition.benchmark import (
    aggregate_task_status,
    build_task_table,
    resume_benchmark,
    run_task,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=["chr"])
    parser.add_argument("--splits", nargs="+", default=["development"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

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
    pending = resume_benchmark(tasks, args.out)
    for index, task in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {task.task_id} {task.method}", flush=True)
        run_task(
            task,
            args.manifest,
            args.out,
            starts=args.starts,
            maxiter=args.maxiter,
            seed=20260710 + task.record_index,
        )
    status = aggregate_task_status(tasks, args.out)
    (args.out / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
