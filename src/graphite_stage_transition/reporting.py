"""Aggregate benchmark evidence and build guarded simulation-only figures."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "graphite-matplotlib"))
import matplotlib.pyplot as plt
import numpy as np


RESULT_DTYPE = np.dtype(
    [
        ("task_id", "U40"),
        ("case_id", "U40"),
        ("status", "U16"),
        ("split", "U16"),
        ("method", "U24"),
        ("noise_fraction", "f8"),
        ("loss", "f8"),
    ]
)


def aggregate_results(output_root: Path) -> np.ndarray:
    """Aggregate every task record, retaining failures in all denominators."""

    rows = []
    output_root = Path(output_root)
    for task_path in sorted((output_root / "tasks").glob("*/task.json")):
        task = json.loads(task_path.read_text(encoding="ascii"))
        result_path = task_path.parent / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="ascii"))
            status = result.get("status", "failed")
            best = result.get("best", {})
            loss = best.get("loss", np.nan)
        else:
            status = "failed"
            loss = np.nan
        rows.append(
            (
                task["task_id"],
                task["case_id"],
                status,
                task["split"],
                task["method"],
                float(task["noise_fraction"]),
                float(loss),
            )
        )
    known = {row[0] for row in rows}
    for failure_path in sorted((output_root / "failures").glob("*.json")):
        failure = json.loads(failure_path.read_text(encoding="ascii"))
        task = failure["task"]
        if task["task_id"] in known:
            continue
        rows.append(
            (
                task["task_id"],
                task["case_id"],
                "failed",
                task["split"],
                task["method"],
                float(task["noise_fraction"]),
                np.nan,
            )
        )
    return np.asarray(rows, dtype=RESULT_DTYPE)


def _figure_paths(output: Path) -> tuple[Path, Path]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, output.with_suffix(".csv")


def build_recovery_figure(
    truth: Mapping[str, float],
    estimate: Mapping[str, float],
    output: Path,
) -> None:
    """Plot preliminary group recovery with a tabular data sidecar."""

    output, sidecar = _figure_paths(output)
    names = ("epsilon_squared", "tau_diffusion", "damkohler")
    truth_values = np.asarray([truth[name] for name in names], dtype=float)
    estimate_values = np.asarray([estimate[name] for name in names], dtype=float)
    figure, axis = plt.subplots(figsize=(4.3, 4.0))
    axis.scatter(truth_values, estimate_values, color="#0072B2", s=42)
    short_names = ("epsilon^2", "tau_D", "Da")
    for label, true, fitted in zip(short_names, truth_values, estimate_values):
        axis.annotate(label, (true, fitted), xytext=(5, 4), textcoords="offset points", fontsize=8)
    lower = min(truth_values.min(), estimate_values.min()) * 0.8
    upper = max(truth_values.max(), estimate_values.max()) * 1.2
    axis.plot([lower, upper], [lower, upper], color="#555555", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Ground-truth group")
    axis.set_ylabel("Estimated group")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    with sidecar.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(("group", "truth", "estimate", "relative_error"))
        for name, true, fitted in zip(names, truth_values, estimate_values):
            writer.writerow((name, true, fitted, abs(fitted - true) / abs(true)))


def build_noise_figure(rows: np.ndarray, output: Path) -> None:
    output, sidecar = _figure_paths(output)
    successful = rows[(rows["status"] == "success") & np.isfinite(rows["loss"])]
    levels = np.unique(successful["noise_fraction"])
    medians = np.asarray(
        [np.median(successful["loss"][successful["noise_fraction"] == level]) for level in levels]
    )
    counts = np.asarray([np.sum(successful["noise_fraction"] == level) for level in levels])
    figure, axis = plt.subplots(figsize=(5.0, 3.4))
    if levels.size:
        axis.plot(100 * levels, medians, marker="o", color="#0072B2")
    axis.set_xlabel("Observation noise (% clean-field std)")
    axis.set_ylabel("Median normalized loss")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    with sidecar.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(("noise_fraction", "median_loss", "successful_tasks"))
        writer.writerows(zip(levels, medians, counts))


def build_identifiability_figure(report: Mapping, output: Path) -> None:
    output, sidecar = _figure_paths(output)
    singular = np.asarray(report["spectrum"]["singular_values"], dtype=float)
    correlation = np.asarray(report["correlation"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].semilogy(np.arange(1, len(singular) + 1), singular, marker="o", color="#0072B2")
    axes[0].set_xlabel("Singular-value index")
    axes[0].set_ylabel("Residual Jacobian singular value")
    image = axes[1].imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    parameter_labels = ("M", "A", "kappa", "k0")
    axes[1].set_xticks(np.arange(4), parameter_labels)
    axes[1].set_yticks(np.arange(4), parameter_labels)
    axes[1].set_xlabel("Log parameter")
    axes[1].set_ylabel("Log parameter")
    figure.colorbar(image, ax=axes[1], fraction=0.05, pad=0.04, label="Correlation")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    with sidecar.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "singular_value"))
        writer.writerows((index + 1, value) for index, value in enumerate(singular))


def build_ablation_figure(baselines: Mapping, output: Path) -> None:
    output, sidecar = _figure_paths(output)
    methods = ("random CHR", "Fickian", "sharp interface")
    losses = (
        float(baselines["random_search"]["loss"]),
        float(baselines["fickian"]["best"]["normalized_spatial_loss"]),
        float(baselines["sharp_interface"]["normalized_spatial_loss"]),
    )
    figure, axis = plt.subplots(figsize=(5.5, 3.4))
    axis.bar(methods, losses, color=("#0072B2", "#E69F00", "#009E73"))
    axis.set_ylabel("Normalized spatial loss")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    with sidecar.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "normalized_spatial_loss"))
        writer.writerows(zip(methods, losses))


def build_failure_figure(rows: np.ndarray, output: Path) -> None:
    output, sidecar = _figure_paths(output)
    statuses = ("success", "failed", "pending")
    counts = [int(np.sum(rows["status"] == status)) for status in statuses]
    figure, axis = plt.subplots(figsize=(4.8, 3.2))
    axis.bar(statuses, counts, color=("#009E73", "#D55E00", "#999999"))
    axis.set_ylabel("Task count")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    with sidecar.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(("status", "count"))
        writer.writerows(zip(statuses, counts))


def write_methods_report(output: Path, evidence: Mapping[str, object]) -> None:
    """Write a guarded report whose evidence ledger is explicit and reproducible."""

    evidence_lines = "\n".join(f"- `{key}`: {value}" for key, value in evidence.items())
    text = f"""# Graphite Stage 2 to Stage 1 Simulation Benchmark

## Claim Boundary

This is a **Simulation-only** effective-scalar Cahn-Hilliard-reaction (CHR)
benchmark. It does not analyze the experimental iSCAT movie, calibrate an
optical observation model, resolve individual lithium ions, or estimate real
graphite material constants.

## Objective

Test which parameter combinations can be recovered from synthetic two-dimensional
concentration movies spanning stage 2 (`c=0.5`) to stage 1 (`c=1.0`) and back.

## Model

The free energy is `F = integral[f(c; A) + kappa |grad c|^2 / 2] dV`, with
`mu = df/dc - kappa laplacian(c)` and conserved bulk dynamics
`dc/dt = div(M grad(mu))`. A galvanostatic boundary reaction enforces the exact
total applied current. Positive parameters are fitted in log space.

Primary groups are `epsilon^2 = kappa/(A L^2)`,
`tau_D = L^2/(M A)`, and `Da = k0 L/(M A)`.

## Verification Gates

The implementation gates discrete mass balance, zero-current free-energy
relaxation, deterministic replay, grid refinement, autodiff versus finite
differences, complete stage traversal, physical concentration bounds, and
case-level split isolation.

## Dataset and Evaluation

The declared full design is 64 Latin-hypercube parameter cases with three seeded
replicates, noise fractions 0, 0.05, 0.10, and 0.20, and temporal subsampling
factors 1, 2, and 4. Development, validation, and locked test splits are assigned
by parameter case before replicate expansion. Failed tasks remain in denominators.

The current artifacts include a 16-case, one-replicate full-transition stability
cohort. They are pipeline evidence, not a statistically powered recovery result.
Locked-test criteria have not been evaluated.

## Inversion, Baselines, and Identifiability

CHR fitting uses a normalized masked movie loss, mass and weak bound penalties,
L-BFGS-B, and recorded multistarts. Controls are equal-budget random CHR search,
implicit Fickian diffusion, a conserved sharp interface, and a mean-only ablation.
Identifiability uses a reduced exact residual Jacobian, Fisher/SVD rank, a
pseudoinverse correlation matrix, and fixed-coordinate profile likelihoods.

## Evidence Ledger

{evidence_lines}

## Limitations and Deferred Work

- The scalar order parameter is an effective stage field, not a graphite gallery model.
- The idealized particle is circular and two-dimensional.
- Direct concentration is observed; iSCAT intensity, optics, drift, and nuisance scales are absent.
- Smoke-case inversion and local curvature are preliminary until multistart GPU jobs converge.
- No dimensional estimate should be transferred to the experimental particle.

The next scientific phase should add a validated differentiable iSCAT observation
operator and compare simulated observables with the measured movie only after the
simulation-only recovery and identifiability gates pass.
"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="ascii")
