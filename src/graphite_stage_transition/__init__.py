"""Differentiable graphite stage-transition benchmark."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .config import ProjectConfig, load_config
from .observables import (
    ObservableGeometry,
    PhysicsObservables,
    make_observable_geometry,
    observable_residual_vector,
    physics_observables,
)

__all__ = [
    "ObservableGeometry",
    "PhysicsObservables",
    "ProjectConfig",
    "load_config",
    "make_observable_geometry",
    "observable_residual_vector",
    "physics_observables",
]
