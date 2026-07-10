#!/usr/bin/env python3
"""Run and persist the forward-model verification gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphite_stage_transition.config import load_config
from graphite_stage_transition.verification import run_verification_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = run_verification_suite(load_config(args.config))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "verification.json").open("w", encoding="ascii") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
