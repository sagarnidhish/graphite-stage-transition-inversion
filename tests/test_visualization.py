from pathlib import Path

import jax.numpy as jnp
import matplotlib.axes
import numpy as np

from graphite_stage_transition.config import GridConfig
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.solver import SimulationResult
from graphite_stage_transition.visualization import radial_kymograph, render_movie


def make_visual_test_result():
    grid = make_circle_grid(GridConfig(nx=24, ny=24, length=1.0, radius=0.4))
    radial = jnp.sqrt(grid.x**2 + grid.y**2) / grid.radius
    frames = jnp.stack(
        [
            jnp.where(grid.mask, 0.5 + 0.05 * index + 0.2 * (1.0 - radial), 0.0)
            for index in range(3)
        ]
    )
    result = SimulationResult(
        concentration=frames,
        times=jnp.array([0.0, 0.5, 1.0]),
        currents=jnp.array([0.02, 0.0, -0.02]),
        mass=jnp.array([0.25, 0.28, 0.26]),
        free_energy=jnp.array([0.1, 0.08, 0.07]),
        overpotential=jnp.array([0.1, 0.0, -0.1]),
        summed_current=jnp.array([0.02, 0.0, -0.02]),
        cg_residual=jnp.array([0.0, 1e-9, 1e-9]),
        metadata={"dt": 0.5, "seed": 1, "steps": 2},
    )
    return result, grid


def test_radial_kymograph_shape():
    result, grid = make_visual_test_result()

    kymograph = radial_kymograph(result.concentration, grid, bins=24)

    assert kymograph.shape == (result.concentration.shape[0], 24)
    assert np.all(np.isfinite(kymograph))


def test_radial_kymograph_does_not_invent_values_for_empty_bins():
    result, grid = make_visual_test_result()
    uniform = jnp.where(grid.mask, 0.7, 0.0)[None, ...]

    kymograph = radial_kymograph(uniform, grid, bins=48)

    np.testing.assert_allclose(kymograph, 0.7, atol=1e-14)


def test_movie_uses_fixed_stage_limits(monkeypatch, tmp_path: Path):
    result, grid = make_visual_test_result()
    calls = []
    original = matplotlib.axes.Axes.imshow

    def record(self, values, **kwargs):
        calls.append(kwargs.copy())
        return original(self, values, **kwargs)

    class MemoryWriter:
        def __init__(self):
            self.frames = []

        def append_data(self, frame):
            self.frames.append(frame)

        def close(self):
            pass

    writer = MemoryWriter()
    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", record)
    monkeypatch.setattr(
        "graphite_stage_transition.visualization.imageio.get_writer",
        lambda *args, **kwargs: writer,
    )

    render_movie(result, grid, tmp_path / "movie.mp4")

    assert calls
    assert all(call["vmin"] == 0.5 and call["vmax"] == 1.0 for call in calls)
    assert len(writer.frames) == result.concentration.shape[0]
