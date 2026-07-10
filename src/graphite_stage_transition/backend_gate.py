"""Fail-closed cross-backend evidence gate for primary inversion observables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from itertools import combinations
from pathlib import Path
import tomllib
from typing import Any, Mapping, Sequence

import numpy as np


PROBE_SCHEMA_VERSION = 1
GATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BackendGateThresholds:
    """Frozen tolerances required before a backend can support scientific claims."""

    observable_block_rms_max: float = 0.02
    objective_range_max: float = 0.005
    objective_cv_max: float = 0.05
    gradient_cosine_min: float = 0.99
    gradient_norm_disagreement_max: float = 0.10
    gradient_small_norm: float = 1e-6

    def __post_init__(self) -> None:
        if self.observable_block_rms_max < 0.0:
            raise ValueError("observable_block_rms_max must be nonnegative")
        if self.objective_range_max < 0.0 or self.objective_cv_max < 0.0:
            raise ValueError("objective thresholds must be nonnegative")
        if not -1.0 <= self.gradient_cosine_min <= 1.0:
            raise ValueError("gradient_cosine_min must lie in [-1, 1]")
        if self.gradient_norm_disagreement_max < 0.0:
            raise ValueError("gradient_norm_disagreement_max must be nonnegative")
        if self.gradient_small_norm < 0.0:
            raise ValueError("gradient_small_norm must be nonnegative")


DEFAULT_THRESHOLDS = BackendGateThresholds()


def _freeze_array(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError("probe arrays must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError("probe arrays must contain only finite values")
    return _nested_tuple(array.tolist())


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return float(value)


@dataclass(frozen=True)
class ProbeCase:
    """Claim-relevant outputs from one fixed model case on one backend."""

    case_id: str
    observable_blocks: Mapping[str, tuple[Any, ...]]
    primary_objective: float
    gradient: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be nonempty")
        if not self.observable_blocks:
            raise ValueError("observable_blocks must be nonempty")
        blocks = {
            str(name): _freeze_array(values)
            for name, values in sorted(self.observable_blocks.items())
        }
        if any(not name for name in blocks):
            raise ValueError("observable block names must be nonempty")
        objective = float(self.primary_objective)
        if not np.isfinite(objective):
            raise ValueError("primary_objective must be finite")
        if objective < 0.0:
            raise ValueError("primary_objective must be nonnegative")
        gradient = np.asarray(self.gradient, dtype=np.float64)
        if gradient.ndim != 1 or gradient.size == 0:
            raise ValueError("gradient must be a nonempty vector")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must contain only finite values")
        object.__setattr__(self, "observable_blocks", blocks)
        object.__setattr__(self, "primary_objective", objective)
        object.__setattr__(self, "gradient", tuple(float(value) for value in gradient))


@dataclass(frozen=True)
class BackendProbe:
    """Serializable evidence produced by one named numerical backend."""

    backend: str
    fingerprint_sha256: str
    cases: tuple[ProbeCase, ...]
    metadata: Mapping[str, Any]
    schema_version: int = PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROBE_SCHEMA_VERSION:
            raise ValueError(f"unsupported probe schema version {self.schema_version}")
        if not self.backend or not self.fingerprint_sha256:
            raise ValueError("backend and fingerprint_sha256 must be nonempty")
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("backend probe must contain at least one case")
        case_ids = tuple(case.case_id for case in cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("backend probe case IDs must be unique")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class BackendGateResult:
    """Auditable comparison record used to authorize claim-eligible execution."""

    passed: bool
    fingerprint_sha256: str | None
    probe_backends: tuple[str, ...]
    thresholds: Mapping[str, float]
    metrics: Mapping[str, Any]
    failures: tuple[str, ...]
    schema_version: int = GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported gate schema version {self.schema_version}")
        object.__setattr__(self, "probe_backends", tuple(self.probe_backends))
        object.__setattr__(self, "thresholds", dict(self.thresholds))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "failures", tuple(self.failures))


def load_backend_gate_thresholds(path: Path) -> BackendGateThresholds:
    """Load the frozen ``[thresholds]`` table from TOML."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        values = payload["thresholds"]
    except KeyError as error:
        raise ValueError("backend gate config requires a [thresholds] table") from error
    return BackendGateThresholds(**values)


def _probe_to_dict(probe: BackendProbe) -> dict[str, Any]:
    return {
        "schema_version": probe.schema_version,
        "backend": probe.backend,
        "fingerprint_sha256": probe.fingerprint_sha256,
        "cases": [asdict(case) for case in probe.cases],
        "metadata": dict(probe.metadata),
    }


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    temporary.replace(output)


def save_backend_probe(probe: BackendProbe, path: Path) -> None:
    """Atomically serialize a backend probe as deterministic JSON."""

    _write_json(_probe_to_dict(probe), path)


def load_backend_probe(path: Path) -> BackendProbe:
    """Load and validate a serialized backend probe."""

    payload = json.loads(Path(path).read_text(encoding="ascii"))
    cases = tuple(ProbeCase(**case) for case in payload["cases"])
    return BackendProbe(
        backend=payload["backend"],
        fingerprint_sha256=payload["fingerprint_sha256"],
        cases=cases,
        metadata=payload.get("metadata", {}),
        schema_version=payload["schema_version"],
    )


def _case_map(probe: BackendProbe) -> dict[str, ProbeCase]:
    return {case.case_id: case for case in probe.cases}


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    if mean == 0.0:
        return 0.0 if standard_deviation == 0.0 else float("inf")
    return standard_deviation / abs(mean)


def compare_backend_probes(
    probes: Sequence[BackendProbe],
    thresholds: BackendGateThresholds = DEFAULT_THRESHOLDS,
) -> BackendGateResult:
    """Compare two or more backend probes against every frozen gate threshold."""

    probes = tuple(probes)
    if len(probes) < 2:
        raise ValueError("backend gate requires at least two probes")
    backends = tuple(probe.backend for probe in probes)
    if len(backends) != len(set(backends)):
        raise ValueError("backend names must be unique within one gate comparison")

    failures: list[str] = []
    fingerprints = {probe.fingerprint_sha256 for probe in probes}
    fingerprint = next(iter(fingerprints)) if len(fingerprints) == 1 else None
    if fingerprint is None:
        failures.append("probe execution fingerprint mismatch")

    reference_ids = tuple(case.case_id for case in probes[0].cases)
    if any(tuple(case.case_id for case in probe.cases) != reference_ids for probe in probes[1:]):
        failures.append("probe case IDs or ordering mismatch")

    observable_metrics: dict[str, dict[str, float]] = {}
    objective_metrics: dict[str, dict[str, float]] = {}
    gradient_metrics: dict[str, dict[str, float | bool]] = {}
    case_maps = tuple(_case_map(probe) for probe in probes)
    common_case_ids = tuple(
        case_id for case_id in reference_ids if all(case_id in mapping for mapping in case_maps)
    )

    for case_id in common_case_ids:
        cases = tuple(mapping[case_id] for mapping in case_maps)
        reference_blocks = tuple(cases[0].observable_blocks)
        if any(tuple(case.observable_blocks) != reference_blocks for case in cases[1:]):
            failures.append(f"{case_id}: observable block names mismatch")
            continue

        observable_metrics[case_id] = {}
        for block_name in reference_blocks:
            arrays = tuple(
                np.asarray(case.observable_blocks[block_name], dtype=np.float64)
                for case in cases
            )
            if any(array.shape != arrays[0].shape for array in arrays[1:]):
                failures.append(f"{case_id}: {block_name} observable shape mismatch")
                continue
            pair_rms = [
                float(np.sqrt(np.mean((first - second) ** 2)))
                for first, second in combinations(arrays, 2)
            ]
            maximum_rms = max(pair_rms)
            observable_metrics[case_id][block_name] = maximum_rms
            if maximum_rms > thresholds.observable_block_rms_max:
                failures.append(
                    f"{case_id}: {block_name} RMS {maximum_rms:.12g} exceeds "
                    f"{thresholds.observable_block_rms_max:.12g}"
                )

        objectives = np.asarray(
            [case.primary_objective for case in cases], dtype=np.float64
        )
        objective_range = float(np.ptp(objectives))
        objective_cv = _coefficient_of_variation(objectives)
        objective_metrics[case_id] = {
            "range": objective_range,
            "coefficient_of_variation": objective_cv,
        }
        if objective_range > thresholds.objective_range_max:
            failures.append(
                f"{case_id}: objective range {objective_range:.12g} exceeds "
                f"{thresholds.objective_range_max:.12g}"
            )
        if objective_cv > thresholds.objective_cv_max:
            failures.append(
                f"{case_id}: objective coefficient of variation {objective_cv:.12g} "
                f"exceeds {thresholds.objective_cv_max:.12g}"
            )

        gradients = tuple(np.asarray(case.gradient, dtype=np.float64) for case in cases)
        if any(gradient.shape != gradients[0].shape for gradient in gradients[1:]):
            failures.append(f"{case_id}: gradient shape mismatch")
            continue
        cosines: list[float] = []
        norm_disagreements: list[float] = []
        all_pairs_small = True
        for first, second in combinations(gradients, 2):
            first_norm = float(np.linalg.norm(first))
            second_norm = float(np.linalg.norm(second))
            if first_norm < thresholds.gradient_small_norm and second_norm < thresholds.gradient_small_norm:
                continue
            all_pairs_small = False
            denominator = first_norm * second_norm
            cosine = float(np.dot(first, second) / denominator) if denominator > 0.0 else 0.0
            norm_denominator = max(first_norm, second_norm)
            norm_disagreement = (
                abs(first_norm - second_norm) / norm_denominator
                if norm_denominator > 0.0
                else 0.0
            )
            cosines.append(cosine)
            norm_disagreements.append(norm_disagreement)

        minimum_cosine = min(cosines, default=1.0)
        maximum_norm_disagreement = max(norm_disagreements, default=0.0)
        gradient_metrics[case_id] = {
            "minimum_cosine_similarity": minimum_cosine,
            "maximum_norm_disagreement": maximum_norm_disagreement,
            "all_pairs_below_small_norm": all_pairs_small,
        }
        if minimum_cosine < thresholds.gradient_cosine_min:
            failures.append(
                f"{case_id}: gradient cosine {minimum_cosine:.12g} is below "
                f"{thresholds.gradient_cosine_min:.12g}"
            )
        if maximum_norm_disagreement > thresholds.gradient_norm_disagreement_max:
            failures.append(
                f"{case_id}: gradient norm disagreement "
                f"{maximum_norm_disagreement:.12g} exceeds "
                f"{thresholds.gradient_norm_disagreement_max:.12g}"
            )

    metrics = {
        "observable_block_rms": observable_metrics,
        "primary_objective": objective_metrics,
        "gradient": gradient_metrics,
    }
    return BackendGateResult(
        passed=not failures,
        fingerprint_sha256=fingerprint,
        probe_backends=backends,
        thresholds=asdict(thresholds),
        metrics=metrics,
        failures=tuple(failures),
    )


def save_backend_gate_result(result: BackendGateResult, path: Path) -> None:
    """Atomically serialize a backend comparison result."""

    _write_json(asdict(result), path)


def load_backend_gate_result(path: Path) -> BackendGateResult:
    """Load a gate result, preserving its immutable authorization fields."""

    payload = json.loads(Path(path).read_text(encoding="ascii"))
    return BackendGateResult(**payload)


def require_matching_passed_gate(
    result: BackendGateResult,
    fingerprint_sha256: str,
) -> None:
    """Reject a failed or stale gate before claim-eligible model execution."""

    if not result.passed:
        raise ValueError("backend gate did not pass")
    if dict(result.thresholds) != asdict(DEFAULT_THRESHOLDS):
        raise ValueError("backend gate did not use the frozen thresholds")
    if result.fingerprint_sha256 != fingerprint_sha256:
        raise ValueError("backend gate fingerprint does not match execution fingerprint")


__all__ = [
    "BackendGateResult",
    "BackendGateThresholds",
    "BackendProbe",
    "DEFAULT_THRESHOLDS",
    "ProbeCase",
    "compare_backend_probes",
    "load_backend_gate_result",
    "load_backend_gate_thresholds",
    "load_backend_probe",
    "require_matching_passed_gate",
    "save_backend_gate_result",
    "save_backend_probe",
]
