#!/usr/bin/env python3
"""Build the standalone kgpu benchmark script."""

from __future__ import annotations

import argparse
from pathlib import Path

from graphite_stage_transition.gpu_payload import build_gpu_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    build_gpu_payload(
        project_root,
        args.manifest,
        args.out,
        args.max_cases,
        args.starts,
        args.maxiter,
    )


if __name__ == "__main__":
    main()
