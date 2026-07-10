#!/usr/bin/env python3
"""Compare deterministic radial CPU/GPU probe payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu", type=Path)
    parser.add_argument("gpu", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cpu = json.loads(args.cpu.read_text())
    gpu = json.loads(args.gpu.read_text())
    concentration_difference = np.asarray(cpu["concentration"]) - np.asarray(gpu["concentration"])
    cpu_gradient = np.asarray(cpu["gradient"], dtype=np.float64)
    gpu_gradient = np.asarray(gpu["gradient"], dtype=np.float64)
    cosine = float(
        np.dot(cpu_gradient, gpu_gradient)
        / (np.linalg.norm(cpu_gradient) * np.linalg.norm(gpu_gradient))
    )
    norm_disagreement = float(
        abs(np.linalg.norm(cpu_gradient) - np.linalg.norm(gpu_gradient))
        / max(np.linalg.norm(cpu_gradient), np.linalg.norm(gpu_gradient))
    )
    metrics = {
        "concentration_max_absolute": float(np.max(np.abs(concentration_difference))),
        "concentration_rms": float(np.sqrt(np.mean(concentration_difference**2))),
        "objective_absolute": float(abs(cpu["objective"] - gpu["objective"])),
        "gradient_cosine": cosine,
        "gradient_norm_disagreement": norm_disagreement,
    }
    thresholds = {
        "concentration_max_absolute": 1e-8,
        "concentration_rms": 1e-9,
        "objective_absolute": 1e-9,
        "gradient_cosine": 0.999999,
        "gradient_norm_disagreement": 1e-6,
    }
    passed = bool(
        metrics["concentration_max_absolute"] <= thresholds["concentration_max_absolute"]
        and metrics["concentration_rms"] <= thresholds["concentration_rms"]
        and metrics["objective_absolute"] <= thresholds["objective_absolute"]
        and metrics["gradient_cosine"] >= thresholds["gradient_cosine"]
        and metrics["gradient_norm_disagreement"] <= thresholds["gradient_norm_disagreement"]
    )
    payload = {"passed": passed, "metrics": metrics, "thresholds": thresholds}
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
