"""Differentiable graphite stage-transition benchmark."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .config import ProjectConfig, load_config

__all__ = ["ProjectConfig", "load_config"]

