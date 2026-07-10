from pathlib import Path

import pytest

from graphite_stage_transition.config import load_config


def test_canonical_config_loads():
    cfg = load_config(Path("configs/canonical.toml"))

    assert cfg.grid.nx == 48
    assert cfg.grid.ny == 48
    assert cfg.model.stage2 == 0.5
    assert cfg.model.stage1 == 1.0
    assert cfg.protocol.currents == (0.02, 0.0, -0.02, 0.0)


def test_invalid_stage_order_fails(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[model]\nstage2=1.0\nstage1=0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stage2 < stage1"):
        load_config(path)
