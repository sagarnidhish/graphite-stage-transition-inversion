"""Local curvature, Fisher-spectrum, correlation, and profile diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .inversion import InverseProblem, inverse_residual_blocks


@dataclass(frozen=True)
class FisherSpectrum:
    singular_values: tuple[float, ...]
    rank: int
    condition_number: float
    relative_cutoff: float


@dataclass(frozen=True)
class ProfileResult:
    parameter_index: int
    fixed_values: np.ndarray
    losses: np.ndarray
    optima: np.ndarray
    statuses: tuple[str, ...]


@dataclass(frozen=True)
class IdentifiabilityReport:
    spectrum: FisherSpectrum
    fisher_matrix: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    jacobian_shape: tuple[int, int]
    profiles: tuple[ProfileResult, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "spectrum": asdict(self.spectrum),
            "fisher_matrix": self.fisher_matrix.tolist(),
            "covariance": self.covariance.tolist(),
            "correlation": self.correlation.tolist(),
            "jacobian_shape": list(self.jacobian_shape),
            "profiles": [
                {
                    "parameter_index": profile.parameter_index,
                    "fixed_values": profile.fixed_values.tolist(),
                    "losses": profile.losses.tolist(),
                    "optima": profile.optima.tolist(),
                    "statuses": list(profile.statuses),
                }
                for profile in self.profiles
            ],
        }


def local_hessian(jacobian) -> np.ndarray:
    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("jacobian must be two-dimensional")
    return values.T @ values


def fisher_spectrum(jacobian, relative_cutoff: float = 1e-8) -> FisherSpectrum:
    """Report numerical rank and Fisher condition without hiding degeneracy."""

    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2 or relative_cutoff <= 0.0:
        raise ValueError("invalid Fisher-spectrum inputs")
    singular_values = np.linalg.svd(values, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = relative_cutoff * largest
    rank = int(np.sum(singular_values > threshold))
    parameter_count = values.shape[1]
    if rank < parameter_count or not singular_values.size or singular_values[-1] == 0.0:
        condition_number = float("inf")
    else:
        condition_number = float((singular_values[0] / singular_values[-1]) ** 2)
    return FisherSpectrum(
        tuple(float(value) for value in singular_values),
        rank,
        condition_number,
        float(relative_cutoff),
    )


def parameter_correlation(jacobian, relative_cutoff: float = 1e-10) -> np.ndarray:
    """Return pseudoinverse-derived parameter correlation for rank-deficient cases."""

    fisher = local_hessian(jacobian)
    covariance = np.linalg.pinv(fisher, rcond=relative_cutoff, hermitian=True)
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    correlation[np.diag_indices_from(correlation)] = 1.0
    return np.clip(correlation, -1.0, 1.0)


def _residual_vector(problem: InverseProblem, unconstrained, max_residuals: int):
    """Return the exact residual or a block-stratified reduced approximation."""

    blocks = inverse_residual_blocks(problem, unconstrained)
    full_size = blocks.morphology.size + blocks.bounds.size
    if full_size <= max_residuals:
        return jnp.concatenate(blocks)

    morphology_count = min(blocks.morphology.size, max_residuals // 2)
    bounds_count = min(blocks.bounds.size, max_residuals - morphology_count)
    remaining = max_residuals - morphology_count - bounds_count
    morphology_count += min(remaining, blocks.morphology.size - morphology_count)
    remaining = max_residuals - morphology_count - bounds_count
    bounds_count += min(remaining, blocks.bounds.size - bounds_count)

    def reduced(block, count):
        if count == block.size:
            return block
        indices = jnp.asarray(
            np.rint(np.linspace(0, block.size - 1, count)).astype(np.int32)
        )
        return block[indices] * jnp.sqrt(block.size / count)

    reduced_morphology = reduced(blocks.morphology, morphology_count)
    reduced_bounds = reduced(blocks.bounds, bounds_count)
    return jnp.concatenate((reduced_morphology, reduced_bounds))


def residual_jacobian(
    problem: InverseProblem,
    optimum,
    max_residuals: int = 2048,
) -> np.ndarray:
    """Differentiate the full residual or its deterministic capped approximation."""

    if max_residuals < 4:
        raise ValueError("max_residuals must be at least four")
    function = lambda values: _residual_vector(problem, values, max_residuals)
    # Four inputs and many residuals make forward mode substantially cheaper
    # than one reverse sweep per residual while preserving the exact Jacobian.
    jacobian = jax.jit(jax.jacfwd(function))(jnp.asarray(optimum))
    return np.asarray(jacobian, dtype=np.float64)


def profile_likelihood(
    problem,
    optimum,
    index: int,
    grid,
    maxiter: int = 50,
) -> ProfileResult:
    """Fix one transformed coordinate and reoptimize all remaining coordinates."""

    optimum_values = np.asarray(optimum, dtype=np.float64)
    fixed_values = np.asarray(grid, dtype=np.float64)
    if optimum_values.ndim != 1 or not 0 <= index < optimum_values.size:
        raise ValueError("invalid profile coordinate")
    free_indices = np.asarray([item for item in range(optimum_values.size) if item != index])
    losses = []
    optima = []
    statuses = []

    for fixed_value in fixed_values:
        if hasattr(problem, "profile_loss"):
            objective = lambda free: float(problem.profile_loss(index, fixed_value, free))
            optimized = minimize(
                objective,
                optimum_values[free_indices],
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-12},
            )
        else:
            def assemble(free):
                values = jnp.asarray(optimum_values).at[index].set(fixed_value)
                return values.at[jnp.asarray(free_indices)].set(free)

            value_and_gradient = jax.jit(
                jax.value_and_grad(lambda free: problem.loss(assemble(free)))
            )

            def objective(free):
                value, gradient = value_and_gradient(jnp.asarray(free))
                return float(value), np.asarray(gradient, dtype=np.float64)

            full_bounds = problem.transform.scipy_bounds()
            free_bounds = tuple(full_bounds[item] for item in free_indices)
            optimized = minimize(
                objective,
                optimum_values[free_indices],
                method="L-BFGS-B",
                jac=True,
                bounds=free_bounds,
                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
            )
        full_optimum = optimum_values.copy()
        full_optimum[index] = fixed_value
        full_optimum[free_indices] = optimized.x
        losses.append(float(optimized.fun))
        optima.append(full_optimum)
        statuses.append("converged" if optimized.success else f"failed: {optimized.message}")
    return ProfileResult(
        index,
        fixed_values,
        np.asarray(losses),
        np.asarray(optima),
        tuple(statuses),
    )


def build_identifiability_report(
    jacobian,
    profiles: tuple[ProfileResult, ...] = (),
    relative_cutoff: float = 1e-8,
) -> IdentifiabilityReport:
    fisher = local_hessian(jacobian)
    covariance = np.linalg.pinv(fisher, rcond=relative_cutoff, hermitian=True)
    return IdentifiabilityReport(
        spectrum=fisher_spectrum(jacobian, relative_cutoff),
        fisher_matrix=fisher,
        covariance=covariance,
        correlation=parameter_correlation(jacobian, relative_cutoff),
        jacobian_shape=tuple(np.asarray(jacobian).shape),
        profiles=profiles,
    )
