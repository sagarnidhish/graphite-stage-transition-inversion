"""Validated project configuration loaded from TOML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


def _positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class GridConfig:
    nx: int
    ny: int
    length: float
    radius: float

    def __post_init__(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("grid dimensions must be at least 8")
        _positive("grid.length", self.length)
        _positive("grid.radius", self.radius)
        if 2.0 * self.radius >= self.length:
            raise ValueError("particle diameter must be smaller than domain length")


@dataclass(frozen=True)
class ModelConfig:
    mobility: float
    barrier: float
    kappa: float
    reaction_rate: float
    stage2: float = 0.5
    stage1: float = 1.0

    def __post_init__(self) -> None:
        if not self.stage2 < self.stage1:
            raise ValueError("stage2 < stage1 is required")
        for name in ("mobility", "barrier", "kappa", "reaction_rate"):
            _positive(f"model.{name}", getattr(self, name))


@dataclass(frozen=True)
class ProtocolConfig:
    currents: tuple[float, ...]
    durations: tuple[float, ...]
    frames_per_segment: int

    def __post_init__(self) -> None:
        if len(self.currents) != len(self.durations) or not self.currents:
            raise ValueError("protocol currents and durations must have equal nonzero length")
        if any(duration <= 0.0 for duration in self.durations):
            raise ValueError("protocol durations must be positive")
        if self.frames_per_segment < 2:
            raise ValueError("frames_per_segment must be at least 2")


@dataclass(frozen=True)
class SolverConfig:
    dt: float
    cg_tolerance: float
    cg_max_iterations: int
    perturbation_amplitude: float
    seed: int

    def __post_init__(self) -> None:
        _positive("solver.dt", self.dt)
        _positive("solver.cg_tolerance", self.cg_tolerance)
        if self.cg_max_iterations < 1:
            raise ValueError("solver.cg_max_iterations must be positive")
        if self.perturbation_amplitude < 0.0:
            raise ValueError("solver.perturbation_amplitude must be nonnegative")


@dataclass(frozen=True)
class InversionConfig:
    starts: int
    max_iterations: int
    mass_penalty: float
    bound_penalty: float

    def __post_init__(self) -> None:
        if self.starts < 1 or self.max_iterations < 1:
            raise ValueError("inversion starts and max_iterations must be positive")
        if self.mass_penalty < 0.0 or self.bound_penalty < 0.0:
            raise ValueError("inversion penalties must be nonnegative")


@dataclass(frozen=True)
class ProjectConfig:
    grid: GridConfig
    model: ModelConfig
    protocol: ProtocolConfig
    solver: SolverConfig
    inversion: InversionConfig


def _required(data: dict, section: str) -> dict:
    try:
        value = data[section]
    except KeyError as exc:
        raise ValueError(f"missing [{section}] section") from exc
    if not isinstance(value, dict):
        raise ValueError(f"[{section}] must be a TOML table")
    return value


def load_config(path: Path) -> ProjectConfig:
    """Load and validate a project TOML configuration."""

    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)

    model_data = _required(data, "model")
    stage2 = float(model_data.get("stage2", 0.5))
    stage1 = float(model_data.get("stage1", 1.0))
    if not stage2 < stage1:
        raise ValueError("stage2 < stage1 is required")

    grid = GridConfig(**_required(data, "grid"))
    model = ModelConfig(**model_data)
    protocol_data = _required(data, "protocol")
    protocol = ProtocolConfig(
        currents=tuple(float(value) for value in protocol_data["currents"]),
        durations=tuple(float(value) for value in protocol_data["durations"]),
        frames_per_segment=int(protocol_data["frames_per_segment"]),
    )
    solver = SolverConfig(**_required(data, "solver"))
    inversion = InversionConfig(**_required(data, "inversion"))
    return ProjectConfig(grid, model, protocol, solver, inversion)

