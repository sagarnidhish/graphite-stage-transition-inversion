from pathlib import Path

import numpy as np

from graphite_stage_transition.config import load_config
from graphite_stage_transition.protocols import (
    _step_save_slots,
    build_protocol,
    make_constant_protocol,
)


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
    assert protocol.save_indices.dtype == np.dtype(np.int32)
    np.testing.assert_allclose(np.diff(np.asarray(protocol.times)), 0.002)


def test_step_save_slots_map_irregular_state_indices_to_completed_steps():
    slots = _step_save_slots(np.asarray([0, 1, 4, 7, 9], dtype=np.int32), steps=9)

    np.testing.assert_array_equal(
        np.asarray(slots),
        np.asarray([1, -1, -1, 2, -1, -1, 3, -1, 4], dtype=np.int32),
    )


def test_step_save_slots_reject_malformed_state_indices():
    malformed = (
        np.asarray([0, 4, 6], dtype=np.int64),
        np.asarray([1, 4], dtype=np.int32),
        np.asarray([0, 4, 3, 6], dtype=np.int32),
        np.asarray([0, 4], dtype=np.int32),
    )

    for save_indices in malformed:
        with np.testing.assert_raises(ValueError):
            _step_save_slots(save_indices, steps=6)
