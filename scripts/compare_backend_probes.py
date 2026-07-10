#!/usr/bin/env python3
"""Compare backend probe artifacts and write a fail-closed gate result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphite_stage_transition.backend_gate import (
    compare_backend_probes,
    load_backend_gate_thresholds,
    load_backend_probe,
    save_backend_gate_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probes", nargs="+", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/backend_gate.toml"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.probes) < 2:
        parser.error("at least two probe files are required")

    thresholds = load_backend_gate_thresholds(args.config)
    probes = tuple(load_backend_probe(path) for path in args.probes)
    result = compare_backend_probes(probes, thresholds)
    save_backend_gate_result(result, args.out)
    print(json.dumps({"passed": result.passed, "failures": result.failures}))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
