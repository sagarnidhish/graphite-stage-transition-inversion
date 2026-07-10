"""Leakage-safe synthetic benchmark generation and provenance manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

from .config import ProjectConfig
from .geometry import make_circle_grid
from .protocols import build_protocol
from .solver import CHRParameters, simulate
from .verification import verify_mass_balance


PARAMETER_NAMES = ("mobility", "barrier", "kappa", "reaction_rate")
DEFAULT_NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20)
DEFAULT_SUBSAMPLING = (1, 2, 4)


@dataclass(frozen=True)
class ParameterCase:
    case_id: str
    mobility: float
    barrier: float
    kappa: float
    reaction_rate: float

    def groups(self, length: float = 1.0) -> dict[str, float]:
        return {
            "epsilon_squared": self.kappa / (self.barrier * length**2),
            "tau_diffusion": length**2 / (self.mobility * self.barrier),
            "damkohler": self.reaction_rate * length / (self.mobility * self.barrier),
        }


@dataclass(frozen=True)
class CaseRun:
    case: ParameterCase
    replicate: int
    seed: int
    split: str


def _case_id(parameters: Mapping[str, float]) -> str:
    canonical = json.dumps(
        {name: format(float(parameters[name]), ".17g") for name in PARAMETER_NAMES},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "case_" + hashlib.sha256(canonical.encode("ascii")).hexdigest()[:12]


def sample_cases(
    count: int,
    bounds: Mapping[str, tuple[float, float]],
    seed: int,
) -> list[ParameterCase]:
    """Draw deterministic Latin-hypercube cases in log-parameter space."""

    if count < 1:
        raise ValueError("count must be positive")
    for name in PARAMETER_NAMES:
        if name not in bounds:
            raise ValueError(f"missing bounds for {name}")
        lower, upper = bounds[name]
        if not (0.0 < lower < upper):
            raise ValueError(f"bounds for {name} must be positive and increasing")
    unit_samples = qmc.LatinHypercube(d=len(PARAMETER_NAMES), seed=seed).random(count)
    lower_log = np.log([bounds[name][0] for name in PARAMETER_NAMES])
    upper_log = np.log([bounds[name][1] for name in PARAMETER_NAMES])
    values = np.exp(qmc.scale(unit_samples, lower_log, upper_log))
    cases = []
    for row in values:
        parameters = dict(zip(PARAMETER_NAMES, map(float, row)))
        cases.append(ParameterCase(_case_id(parameters), **parameters))
    return cases


def assign_case_splits(
    cases: Sequence[ParameterCase],
    seed: int,
    replicates: int = 3,
) -> list[CaseRun]:
    """Assign whole parameter cases to splits before expanding replicates."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    unique = {case.case_id: case for case in cases}
    if len(unique) != len(cases):
        raise ValueError("case IDs must be unique before replicate expansion")
    ordered = [unique[key] for key in sorted(unique)]
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    development_count = max(1, int(round(0.6 * len(ordered))))
    validation_count = int(round(0.2 * len(ordered)))
    if len(ordered) >= 3:
        validation_count = max(1, validation_count)
        development_count = min(development_count, len(ordered) - 2)
    test_start = development_count + validation_count
    if test_start >= len(ordered) and len(ordered) > 1:
        validation_count = max(0, len(ordered) - development_count - 1)
        test_start = development_count + validation_count

    split_by_index = {}
    for rank, case_index in enumerate(permutation):
        if rank < development_count:
            split = "development"
        elif rank < test_start:
            split = "validation"
        else:
            split = "test"
        split_by_index[int(case_index)] = split

    seed_generator = np.random.default_rng(seed + 1)
    rows: list[CaseRun] = []
    for case_index, case in enumerate(ordered):
        for replicate in range(replicates):
            rows.append(
                CaseRun(
                    case=case,
                    replicate=replicate,
                    seed=int(seed_generator.integers(0, np.iinfo(np.int32).max)),
                    split=split_by_index[case_index],
                )
            )
    return rows


def corrupt_concentration(
    clean,
    noise_fraction: float,
    seed: int,
    mask=None,
    lower: float = 0.5,
    upper: float = 1.0,
) -> np.ndarray:
    """Add seeded Gaussian noise scaled to clean active-field variation."""

    values = np.asarray(clean, dtype=np.float64)
    if values.ndim < 2 or noise_fraction < 0.0 or not lower < upper:
        raise ValueError("invalid concentration corruption inputs")
    if mask is None:
        active = np.ones(values.shape[-2:], dtype=bool)
    else:
        active = np.asarray(mask, dtype=bool)
        if active.shape != values.shape[-2:]:
            raise ValueError("mask shape must match concentration spatial dimensions")
    selected = values[..., active]
    clean_scale = float(selected.std())
    corrupted = values.copy()
    if noise_fraction > 0.0 and clean_scale > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(selected.shape)
        noise -= noise.mean()
        noise *= noise_fraction * clean_scale / noise.std()
        corrupted[..., active] = selected + noise
    corrupted[..., active] = np.clip(corrupted[..., active], lower, upper)
    if mask is not None:
        corrupted[..., ~active] = values[..., ~active]
    return corrupted


def validate_transition_trajectory(
    concentration,
    mask,
    stage2: float,
    stage1: float,
    overshoot_fraction: float = 0.15,
    stage_tolerance: float = 0.02,
) -> dict[str, float]:
    """Fail closed on nonfinite, runaway, or incomplete transition movies."""

    values = np.asarray(concentration, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != active.shape or not np.any(active):
        raise ValueError("trajectory and mask shapes are inconsistent")
    selected = values[:, active]
    if not np.all(np.isfinite(selected)):
        raise ValueError("transition trajectory contains nonfinite concentration")
    width = stage1 - stage2
    minimum = float(selected.min())
    maximum = float(selected.max())
    lower_limit = stage2 - overshoot_fraction * width
    upper_limit = stage1 + overshoot_fraction * width
    if minimum < lower_limit or maximum > upper_limit:
        raise ValueError(
            "transition trajectory left the allowed physical range: "
            f"[{minimum:.6g}, {maximum:.6g}] not within "
            f"[{lower_limit:.6g}, {upper_limit:.6g}]"
        )
    means = selected.mean(axis=1)
    if abs(float(means[0]) - stage2) > stage_tolerance:
        raise ValueError("transition trajectory does not start at stage 2")
    if float(means.max()) < stage1 - stage_tolerance:
        raise ValueError("transition trajectory does not reach stage 1")
    if abs(float(means[-1]) - stage2) > stage_tolerance:
        raise ValueError("transition trajectory does not return to stage 2")
    return {
        "concentration_min": minimum,
        "concentration_max": maximum,
        "mean_initial": float(means[0]),
        "mean_maximum": float(means.max()),
        "mean_final": float(means[-1]),
    }


def generate_case(
    config: ProjectConfig,
    run: CaseRun,
    output_root: Path,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    subsampling: Sequence[int] = DEFAULT_SUBSAMPLING,
) -> list[dict[str, Any]]:
    """Simulate one case replicate and write clean and observation artifacts."""

    output_root = Path(output_root)
    directory = output_root / run.case.case_id / f"replicate_{run.replicate:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    grid = make_circle_grid(config.grid)
    protocol = build_protocol(config.protocol, config.solver.dt)
    parameters = CHRParameters(
        run.case.mobility,
        run.case.barrier,
        run.case.kappa,
        run.case.reaction_rate,
        config.model.stage2,
        config.model.stage1,
    )
    result = simulate(grid, protocol, parameters, config.solver, seed=run.seed)
    concentration = np.asarray(result.concentration)
    validation = validate_transition_trajectory(
        concentration,
        grid.mask,
        config.model.stage2,
        config.model.stage1,
    )
    diagnostics = (
        np.asarray(result.mass),
        np.asarray(result.free_energy),
        np.asarray(result.overpotential),
        np.asarray(result.summed_current),
        np.asarray(result.cg_residual),
    )
    if not all(np.all(np.isfinite(values)) for values in diagnostics):
        raise ValueError(f"nonfinite solver diagnostics for {run.case.case_id}")
    mass_gate = verify_mass_balance(
        result.times,
        result.currents,
        result.mass,
        tolerance=1e-8,
    )
    if not mass_gate.passed:
        raise ValueError(
            f"mass-balance gate failed for {run.case.case_id}: "
            f"relative error {mass_gate.relative_error:.6g}"
        )
    current_error = float(
        np.max(
            np.abs(
                np.asarray(result.summed_current)[1:]
                - np.asarray(result.currents)[1:]
            )
        )
    )
    if current_error > 1e-10:
        raise ValueError(
            f"current-summation gate failed for {run.case.case_id}: {current_error:.6g}"
        )
    validation.update(
        {
            "mass_relative_error": mass_gate.relative_error,
            "mass_max_absolute_error": mass_gate.max_absolute_error,
            "current_max_absolute_error": current_error,
            "cg_residual_max": float(np.max(np.asarray(result.cg_residual))),
        }
    )
    common = {
        "times": np.asarray(result.times),
        "currents": np.asarray(result.currents),
        "mask": np.asarray(grid.mask),
        "x": np.asarray(grid.x),
        "y": np.asarray(grid.y),
    }
    clean_path = directory / "clean.npz"
    np.savez_compressed(
        clean_path,
        concentration=concentration,
        mass=np.asarray(result.mass),
        free_energy=np.asarray(result.free_energy),
        overpotential=np.asarray(result.overpotential),
        summed_current=np.asarray(result.summed_current),
        cg_residual=np.asarray(result.cg_residual),
        **common,
    )

    subsample_indices = {
        str(factor): np.arange(0, concentration.shape[0], factor, dtype=np.int32)
        for factor in subsampling
    }
    records = []
    for noise_fraction in noise_levels:
        noise_code = int(round(1000.0 * noise_fraction))
        observation = (
            concentration.copy()
            if noise_fraction == 0.0
            else corrupt_concentration(
                concentration,
                noise_fraction,
                seed=run.seed + noise_code,
                mask=np.asarray(grid.mask),
                lower=config.model.stage2,
                upper=config.model.stage1,
            )
        )
        observation_path = directory / f"observation_noise_{noise_code:03d}.npz"
        np.savez_compressed(
            observation_path,
            concentration=observation,
            **common,
            **{f"subsample_indices_{factor}": indices for factor, indices in subsample_indices.items()},
        )
        records.append(
            {
                "case_id": run.case.case_id,
                "replicate": run.replicate,
                "seed": run.seed,
                "split": run.split,
                "noise_fraction": float(noise_fraction),
                "clean_path": clean_path.relative_to(output_root).as_posix(),
                "observation_path": observation_path.relative_to(output_root).as_posix(),
                "parameters": {name: getattr(run.case, name) for name in PARAMETER_NAMES},
                "groups": run.case.groups(config.grid.length),
                "subsampling_factors": list(subsampling),
                "frames": int(concentration.shape[0]),
                "finite": True,
                "validation": validation,
            }
        )
    return records


def write_manifest(
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
    metadata: Mapping[str, Any],
) -> None:
    """Write versioned JSON and flat CSV manifests from identical records."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "metadata": dict(metadata),
        "records": list(records),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    fieldnames = (
        "case_id",
        "replicate",
        "seed",
        "split",
        "noise_fraction",
        "clean_path",
        "observation_path",
        "frames",
        "finite",
        *PARAMETER_NAMES,
        "epsilon_squared",
        "tau_diffusion",
        "damkohler",
        "subsampling_factors",
    )
    with (output_root / "manifest.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            flat = {key: record[key] for key in fieldnames if key in record}
            flat.update(record["parameters"])
            flat.update(record["groups"])
            flat["subsampling_factors"] = json.dumps(record["subsampling_factors"])
            writer.writerow(flat)
