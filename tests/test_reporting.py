import json
from pathlib import Path

import numpy as np

from graphite_stage_transition.reporting import (
    aggregate_results,
    build_failure_figure,
    build_recovery_figure,
    write_methods_report,
)


def _write_task(root: Path, task_id: str, status: str, split: str, loss: float):
    directory = root / "tasks" / task_id
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "record_index": 0,
                "case_id": task_id,
                "replicate": 0,
                "noise_fraction": 0.0,
                "split": split,
                "method": "chr",
            }
        )
    )
    (directory / "result.json").write_text(
        json.dumps({"status": status, "best": {"loss": loss}})
    )


def test_aggregation_keeps_failed_runs(tmp_path: Path):
    _write_task(tmp_path, "task_success", "success", "development", 0.01)
    _write_task(tmp_path, "task_failed", "failed", "test", np.nan)

    rows = aggregate_results(tmp_path)

    assert len(rows) == 2
    assert "status" in rows.dtype.names
    assert set(rows["split"]) == {"development", "test"}
    assert set(rows["status"]) == {"success", "failed"}


def test_recovery_and_failure_figures_write_sidecars(tmp_path: Path):
    truth = {"epsilon_squared": 0.0012, "tau_diffusion": 6.0, "damkohler": 1.1}
    estimate = {"epsilon_squared": 0.0013, "tau_diffusion": 5.8, "damkohler": 1.0}
    rows = np.array(
        [("success", "development"), ("failed", "test")],
        dtype=[("status", "U16"), ("split", "U16")],
    )

    build_recovery_figure(truth, estimate, tmp_path / "recovery.png")
    build_failure_figure(rows, tmp_path / "failures.png")

    assert (tmp_path / "recovery.png").is_file()
    assert (tmp_path / "recovery.csv").is_file()
    assert (tmp_path / "failures.png").is_file()
    assert (tmp_path / "failures.csv").is_file()


def test_methods_report_preserves_claim_boundary(tmp_path: Path):
    output = tmp_path / "methods_report.md"

    write_methods_report(
        output,
        evidence={"tests": "47 passed", "gpu_status": "approval pending"},
    )

    text = output.read_text()
    assert "Simulation-only" in text
    assert "does not analyze the experimental iSCAT movie" in text
    assert "approval pending" in text
