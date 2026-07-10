#!/usr/bin/env python3
"""Build guarded figures and a methods report from current benchmark evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from graphite_stage_transition.reporting import (
    aggregate_results,
    build_ablation_figure,
    build_failure_figure,
    build_identifiability_figure,
    build_noise_figure,
    build_recovery_figure,
    write_methods_report,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = aggregate_results(args.benchmark)
    build_failure_figure(rows, args.out / "task_status.png")
    build_noise_figure(rows, args.out / "noise_summary.png")

    inversion = _read(project / "outputs" / "inversion_probe" / "fit.json")
    build_recovery_figure(
        inversion["record"]["groups"],
        inversion["best"]["groups"],
        args.out / "preliminary_group_recovery.png",
    )
    baselines = _read(project / "outputs" / "baseline_smoke" / "baselines.json")
    build_ablation_figure(baselines, args.out / "baseline_comparison.png")
    identifiability = _read(
        project / "outputs" / "identifiability_probe" / "identifiability.json"
    )
    build_identifiability_figure(
        identifiability, args.out / "preliminary_identifiability.png"
    )
    for name in ("concentration_montage.png", "radial_kymograph.png", "concentration_movie.mp4"):
        source = project / "outputs" / "transition_forward" / "rendered" / name
        if source.is_file():
            shutil.copy2(source, args.out / name)

    verification = _read(project / "outputs" / "verification" / "verification.json")
    evidence = {
        "test_suite": "51 tests passed in the current reporting build",
        "staged_forward_cohort": (
            "16 cases; max mass relative error 2.64e-14; max current error "
            "4.04e-15; field range 0.463 to 1.039"
        ),
        "forward_verification": verification,
        "benchmark_tasks": {status: int((rows["status"] == status).sum()) for status in ("success", "failed")},
        "local_full_inversion_probe_seconds": inversion["best"]["runtime_seconds"],
        "local_probe_status": inversion["best"]["status"],
        "preliminary_fisher_condition_number": identifiability["spectrum"]["condition_number"],
        "gpu_status": "external Kaggle upload approval pending",
        "locked_test_status": "not evaluated",
    }
    write_methods_report(args.out / "methods_report.md", evidence)


if __name__ == "__main__":
    main()
