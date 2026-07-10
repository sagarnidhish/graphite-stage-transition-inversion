from pathlib import Path

import numpy as np

from graphite_stage_transition.config import load_config
from graphite_stage_transition.protocols import build_protocol, make_constant_protocol


def test_protocol_has_requested_current_and_rest_segments():
    canonical = load_config(Path("configs/canonical.toml"))

    protocol = build_protocol(canonical.protocol, canonical.solver.dt)

    assert protocol.current[0] > 0.0
    assert np.any(np.asarray(protocol.current) == 0.0)
    assert protocol.current[-1] == 0.0
    assert protocol.save_indices[0] == 0
    assert protocol.save_indices[-1] == len(protocol.current)
    assert np.all(np.diff(np.asarray(protocol.save_indices)) > 0)


def test_constant_protocol_has_step_aligned_times():
    protocol = make_constant_protocol(current=-0.01, steps=7, dt=0.002)

    assert protocol.current.shape == (7,)
    assert protocol.times.shape == (8,)
    np.testing.assert_allclose(np.diff(np.asarray(protocol.times)), 0.002)

