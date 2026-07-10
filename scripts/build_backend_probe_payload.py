#!/usr/bin/env python3
"""Build a self-contained kgpu payload for the bounded backend probe."""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--backend-name", default="kaggle-p100")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.manifest.read_text(encoding="ascii"))
    selected = []
    seen = set()
    for record in manifest["records"]:
        if record["split"] != "development" or float(record["noise_fraction"]) != 0.0:
            continue
        if record["case_id"] in seen:
            continue
        selected.append(record)
        seen.add(record["case_id"])
        if len(selected) == args.max_cases:
            break
    if len(selected) != args.max_cases:
        raise ValueError("not enough clean development cases for backend probe")

    subset = deepcopy(manifest)
    subset["records"] = selected
    subset["metadata"]["config_path"] = "configs/transition.toml"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted((project / "src" / "graphite_stage_transition").glob("*.py")):
            archive.add(path, arcname=f"src/graphite_stage_transition/{path.name}")
        archive.add(
            project / "scripts" / "run_backend_probe.py",
            arcname="scripts/run_backend_probe.py",
        )
        archive.add(project / "configs" / "transition.toml", arcname="configs/transition.toml")
        archive.add(args.fingerprint, arcname="execution.json")
        manifest_bytes = (json.dumps(subset, sort_keys=True) + "\n").encode("ascii")
        manifest_info = tarfile.TarInfo("benchmark/manifest.json")
        manifest_info.size = len(manifest_bytes)
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        for record in selected:
            source = args.manifest.parent / record["observation_path"]
            archive.add(source, arcname=f"benchmark/{record['observation_path']}")

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = f'''#!/usr/bin/env python3
import base64, io, pathlib, subprocess, sys, tarfile
ROOT = pathlib.Path("/kaggle/working/graphite_backend_probe")
ROOT.mkdir(parents=True, exist_ok=True)
ARCHIVE_B64 = "{encoded}"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE_B64)), mode="r:gz") as archive:
    archive.extractall(ROOT)
sys.path.insert(0, str(ROOT / "src"))
output = pathlib.Path("/kaggle/working/backend_probe_gpu.json")
subprocess.run([
    sys.executable,
    str(ROOT / "scripts" / "run_backend_probe.py"),
    "--manifest", str(ROOT / "benchmark" / "manifest.json"),
    "--fingerprint", str(ROOT / "execution.json"),
    "--backend-name", "{args.backend_name}",
    "--backend-kind", "gpu",
    "--max-cases", "{args.max_cases}",
    "--analytic-targets",
    "--out", str(output),
], check=True, cwd=ROOT)
print(f"BACKEND_PROBE={{output}}", flush=True)
'''
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="ascii")


if __name__ == "__main__":
    main()
