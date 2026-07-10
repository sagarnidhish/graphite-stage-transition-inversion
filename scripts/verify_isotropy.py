"""Run the angular isotropy gate on a saved transition trajectory."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from graphite_stage_transition.config import GridConfig, load_config
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.verification import isotropy_scores, verify_isotropy

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trajectory", type=Path); p.add_argument("--config", type=Path, default=Path("configs/transition.toml")); p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--tolerance", type=float, default=0.05)
    a = p.parse_args(); c = load_config(a.config); d = np.load(a.trajectory)
    g = make_circle_grid(GridConfig(c.grid.nx, c.grid.ny, c.grid.length, c.grid.radius))
    gate = verify_isotropy(d["concentration"], g, c.model.stage2, c.model.stage1, tolerance=a.tolerance)
    rms, mx = isotropy_scores(d["concentration"], g, c.model.stage2, c.model.stage1)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"passed": gate.passed, "gate": gate.__dict__, "times": d["times"].tolist(), "angular_rms": rms, "angular_max": mx}
    (a.out_dir / "isotropy_gate.json").write_text(json.dumps(payload, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    ax.plot(d["times"], rms, label="angular RMS", color="#1565c0"); ax.plot(d["times"], mx, label="angular maximum", color="#c62828")
    ax.axhline(a.tolerance, ls="--", color="#1565c0"); ax.axhline(2*a.tolerance, ls="--", color="#c62828")
    ax.set(xlabel="time", ylabel="normalized deviation", title="Angular isotropy diagnostic"); ax.set_ylim(0, max(0.12, 1.05*max(mx))); ax.legend(frameon=False, ncol=2)
    fig.savefig(a.out_dir / "angular_anisotropy.png", dpi=180); plt.close(fig)
    print(json.dumps(payload["gate"], indent=2)); raise SystemExit(0 if gate.passed else 2)
if __name__ == "__main__": main()
