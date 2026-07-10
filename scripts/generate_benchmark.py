#!/usr/bin/env python3
"""Generate a versioned concentration-field inversion benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from graphite_stage_transition.config import load_config
from graphite_stage_transition.synthetic import (
    assign_case_splits,
    generate_case,
    sample_cases,
    write_manifest,
)


BOUNDS = {
    "mobility": (0.12, 0.32),
    "barrier": (0.75, 1.25),
    "kappa": (0.0010, 0.0022),
    "reaction_rate": (0.16, 0.38),
}


def _revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    cases = sample_cases(args.cases, BOUNDS, args.seed)
    runs = assign_case_splits(cases, args.seed + 1, args.replicates)
    records = []
    for index, run in enumerate(runs, start=1):
        print(
            f"[{index}/{len(runs)}] {run.case.case_id} replicate={run.replicate} split={run.split}",
            flush=True,
        )
        records.extend(generate_case(config, run, args.out))
    metadata = {
        "generator": "graphite_stage_transition.synthetic",
        "config_path": str(args.config),
        "config": asdict(config),
        "bounds": BOUNDS,
        "sampling_seed": args.seed,
        "case_count": args.cases,
        "replicates_per_case": args.replicates,
        "noise_levels": [0.0, 0.05, 0.1, 0.2],
        "subsampling_factors": [1, 2, 4],
        "git_revision": _revision(),
        "claim_boundary": (
            "Synthetic effective-scalar CHR concentration fields; no iSCAT observation "
            "model and no claim of measured graphite parameters."
        ),
    }
    write_manifest(records, args.out, metadata)
    split_counts = {
        split: len({record["case_id"] for record in records if record["split"] == split})
        for split in ("development", "validation", "test")
    }
    (args.out / "generation_summary.json").write_text(
        json.dumps(
            {"records": len(records), "case_split_counts": split_counts},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
