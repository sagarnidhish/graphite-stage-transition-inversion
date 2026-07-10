#!/usr/bin/env python3
"""Retrieve and extract kgpu benchmark artifacts downloaded by kgpu."""

from __future__ import annotations

import argparse
from pathlib import Path

from graphite_stage_transition.gpu_payload import retrieve_gpu_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-output", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    retrieve_gpu_results(args.kernel_output, args.out)


if __name__ == "__main__":
    main()
