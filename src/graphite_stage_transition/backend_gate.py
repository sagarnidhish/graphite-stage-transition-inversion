"""Fail-closed cross-backend evidence gate for primary inversion observables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from itertools import combinations
from pathlib import Path
import tomllib
from typing import Any, Mapping, Sequence

import numpy as np


PROBE_SCHEMA_VERSION = 2
GATE_SCHEMA_VERSION = 2

ANALYTIC_TARGET_MODE = "analytic_charge_consistent"
PROBE_CASE_COUNT = 2
PROBE_DEFINITION_SCHEMA = (
    "graphite-stage-transition-backend-probe-v2|clean-development-cases|"
    "case-count=2|"
    "manifest-case-parameters|analytic-charge-consistent-target|"
    "physics-observables-v1|primary-objective-gradient"
)
PROBE_DEFINITION_SHA256 = hashlib.sha256(
    PROBE_DEFINITION_SCHEMA.encode("ascii")
).hexdigest()
_BACKEND_KINDS = frozenset(("canonical_cpu", "gpu"))
_METRIC_SECTIONS = frozenset(
    ("observable_block_rms", "primary_objective", "gradient")
)


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


def analytic_reference_movie(
    *,
    x,
    y,
    mask,
    currents,
    save_indices,
    dt: float,
    cell_area: float,
    stage2: float,
    stage1: float,
) -> np.ndarray:
    """Construct a backend-independent, charge-consistent morphology target."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    current_values = np.asarray(currents, dtype=np.float64)
    saves = np.asarray(save_indices, dtype=np.int64)
    if (
        x_values.shape != active.shape
        or y_values.shape != active.shape
        or current_values.ndim != 1
        or saves.ndim != 1
        or saves.size < 2
        or saves[0] != 0
        or saves[-1] != current_values.size
        or np.any(np.diff(saves) <= 0)
        or dt <= 0.0
        or cell_area <= 0.0
        or not stage2 < stage1
        or not np.any(active)
    ):
        raise ValueError("invalid analytic reference inputs")
    cumulative_charge = np.concatenate(
        ([0.0], np.cumsum(current_values * float(dt)))
    )[saves]
    particle_area = float(np.sum(active)) * float(cell_area)
    filling_mean = cumulative_charge / (particle_area * (stage1 - stage2))
    if np.any(filling_mean < -1e-12) or np.any(filling_mean > 1.0 + 1e-12):
        raise ValueError("analytic reference charge leaves the stage interval")

    length = max(float(np.ptp(x_values)), float(np.ptp(y_values)))
    mode = np.cos(2.0 * np.pi * x_values / length) * np.cos(
        2.0 * np.pi * y_values / length
    )
    mode = np.where(active, mode - np.mean(mode[active]), 0.0)
    mode_scale = float(np.max(np.abs(mode[active])))
    if mode_scale > 0.0:
        mode = mode / mode_scale
    amplitude = 0.05 * np.sin(np.pi * filling_mean)
    filling = filling_mean[:, None, None] + amplitude[:, None, None] * mode[None, ...]
    concentration = stage2 + (stage1 - stage2) * filling
    return np.where(active[None, ...], concentration, 0.0)


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
    backend_kind: str
    fingerprint_sha256: str
    target_mode: str
    probe_definition_sha256: str
    cases: tuple[ProbeCase, ...]
    metadata: Mapping[str, Any]
    schema_version: int = PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROBE_SCHEMA_VERSION:
            raise ValueError(f"unsupported probe schema version {self.schema_version}")
        if not self.backend or not self.fingerprint_sha256:
            raise ValueError("backend and fingerprint_sha256 must be nonempty")
        if self.backend_kind not in _BACKEND_KINDS:
            raise ValueError(
                "backend_kind must be either 'canonical_cpu' or 'gpu'"
            )
        if not self.target_mode:
            raise ValueError("target_mode must be nonempty")
        if not _is_sha256(self.probe_definition_sha256):
            raise ValueError("probe_definition_sha256 must be a lowercase SHA-256")
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
    backend_kinds: Mapping[str, str]
    target_mode: str | None
    probe_definition_sha256: str | None
    probe_sha256: Mapping[str, str]
    thresholds: Mapping[str, float]
    metrics: Mapping[str, Any]
    failures: tuple[str, ...]
    schema_version: int = GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported gate schema version {self.schema_version}")
        object.__setattr__(self, "probe_backends", tuple(self.probe_backends))
        object.__setattr__(self, "backend_kinds", dict(self.backend_kinds))
        object.__setattr__(self, "probe_sha256", dict(self.probe_sha256))
        object.__setattr__(self, "thresholds", dict(self.thresholds))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "failures", tuple(self.failures))
        _validate_gate_result(self)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_gate_result(result: BackendGateResult) -> None:
    """Recheck every authorization invariant, including after in-memory mutation."""

    if type(result.passed) is not bool:
        raise ValueError("passed must be a boolean")
    if any(not isinstance(failure, str) or not failure for failure in result.failures):
        raise ValueError("failures must contain only nonempty strings")
    if result.passed != (len(result.failures) == 0):
        raise ValueError("passed must equal whether failures is empty")

    backends = result.probe_backends
    if (
        len(backends) < 2
        or len(backends) != len(set(backends))
        or any(not isinstance(backend, str) or not backend for backend in backends)
    ):
        raise ValueError("gate requires at least two distinct backends")
    backend_set = set(backends)
    if set(result.backend_kinds) != backend_set:
        raise ValueError("backend_kinds must cover exactly the probe backends")
    if any(kind not in _BACKEND_KINDS for kind in result.backend_kinds.values()):
        raise ValueError("backend_kinds contains an unsupported backend kind")
    if set(result.probe_sha256) != backend_set or any(
        not _is_sha256(digest) for digest in result.probe_sha256.values()
    ):
        raise ValueError(
            "probe_sha256 must contain one valid evidence hash per backend"
        )

    if not isinstance(result.thresholds, Mapping):
        raise ValueError("thresholds must be a mapping")
    if set(result.metrics) != _METRIC_SECTIONS:
        raise ValueError("metrics must contain every required metric section")
    if any(not isinstance(result.metrics[name], Mapping) for name in _METRIC_SECTIONS):
        raise ValueError("metric sections must be mappings")

    if result.passed:
        if not isinstance(result.fingerprint_sha256, str) or not result.fingerprint_sha256:
            raise ValueError("passed gate requires a nonnull fingerprint")
        kinds = tuple(result.backend_kinds.values())
        if kinds.count("canonical_cpu") != 1:
            raise ValueError("passed gate requires exactly one canonical_cpu backend")
        if "gpu" not in kinds:
            raise ValueError("passed gate requires at least one gpu backend")
        if not result.target_mode:
            raise ValueError("passed gate requires a nonempty target mode")
        if not _is_sha256(result.probe_definition_sha256):
            raise ValueError("passed gate requires a valid probe definition SHA-256")
        if dict(result.thresholds) != asdict(DEFAULT_THRESHOLDS):
            raise ValueError("passed gate requires the frozen thresholds")
        if any(
            type(value) not in (int, float) or not np.isfinite(value)
            for value in result.thresholds.values()
        ):
            raise ValueError("passed gate thresholds must be finite numbers")
        if any(not result.metrics[name] for name in _METRIC_SECTIONS):
            raise ValueError("passed gate requires nonempty complete metric sections")
        metric_case_sets = [set(result.metrics[name]) for name in _METRIC_SECTIONS]
        if any(case_ids != metric_case_sets[0] for case_ids in metric_case_sets[1:]):
            raise ValueError("passed gate metric sections must cover the same cases")
        _validate_passed_metrics(result.metrics)


def _finite_metric(value: Any) -> bool:
    return type(value) in (int, float) and bool(np.isfinite(value))


def _validate_passed_metrics(metrics: Mapping[str, Any]) -> None:
    for case_id in metrics["observable_block_rms"]:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("metric case IDs must be nonempty strings")
        observable = metrics["observable_block_rms"][case_id]
        objective = metrics["primary_objective"][case_id]
        gradient = metrics["gradient"][case_id]
        if not isinstance(observable, Mapping) or not observable:
            raise ValueError("observable metrics must contain nonempty blocks")
        if any(
            not isinstance(name, str)
            or not name
            or not _finite_metric(value)
            or value < 0.0
            for name, value in observable.items()
        ):
            raise ValueError("observable metrics must be finite nonnegative values")
        if not isinstance(objective, Mapping) or set(objective) != {
            "range",
            "coefficient_of_variation",
        }:
            raise ValueError("objective metrics are incomplete")
        if any(not _finite_metric(value) or value < 0.0 for value in objective.values()):
            raise ValueError("objective metrics must be finite nonnegative values")
        if not isinstance(gradient, Mapping) or set(gradient) != {
            "minimum_cosine_similarity",
            "maximum_norm_disagreement",
            "all_pairs_below_small_norm",
        }:
            raise ValueError("gradient metrics are incomplete")
        cosine = gradient["minimum_cosine_similarity"]
        disagreement = gradient["maximum_norm_disagreement"]
        if not _finite_metric(cosine) or not -1.0 <= cosine <= 1.0:
            raise ValueError("gradient cosine metric must be finite and lie in [-1, 1]")
        if not _finite_metric(disagreement) or disagreement < 0.0:
            raise ValueError(
                "gradient norm disagreement metric must be finite and nonnegative"
            )
        if type(gradient["all_pairs_below_small_norm"]) is not bool:
            raise ValueError("gradient small-norm metric must be a boolean")


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
        "backend_kind": probe.backend_kind,
        "fingerprint_sha256": probe.fingerprint_sha256,
        "target_mode": probe.target_mode,
        "probe_definition_sha256": probe.probe_definition_sha256,
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
        backend_kind=payload["backend_kind"],
        fingerprint_sha256=payload["fingerprint_sha256"],
        target_mode=payload["target_mode"],
        probe_definition_sha256=payload["probe_definition_sha256"],
        cases=cases,
        metadata=payload.get("metadata", {}),
        schema_version=payload["schema_version"],
    )


def _probe_evidence_sha256(probe: BackendProbe) -> str:
    encoded = json.dumps(
        _probe_to_dict(probe),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


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
    if thresholds != DEFAULT_THRESHOLDS:
        failures.append("backend gate did not use the frozen thresholds")
    fingerprints = {probe.fingerprint_sha256 for probe in probes}
    fingerprint = next(iter(fingerprints)) if len(fingerprints) == 1 else None
    if fingerprint is None:
        failures.append("probe execution fingerprint mismatch")
    if any(len(probe.cases) != PROBE_CASE_COUNT for probe in probes):
        failures.append(
            f"backend probes must contain exactly {PROBE_CASE_COUNT} cases"
        )

    backend_kinds = {probe.backend: probe.backend_kind for probe in probes}
    kinds = tuple(backend_kinds.values())
    if kinds.count("canonical_cpu") != 1:
        failures.append("backend gate requires exactly one canonical_cpu backend")
    if "gpu" not in kinds:
        failures.append("backend gate requires at least one gpu backend")
    for probe in probes:
        runtime_backend = probe.metadata.get("jax_default_backend")
        expected_runtime = "cpu" if probe.backend_kind == "canonical_cpu" else "gpu"
        devices = probe.metadata.get("devices")
        if runtime_backend != expected_runtime or not isinstance(devices, (list, tuple)) or not devices:
            failures.append(
                f"{probe.backend}: backend kind is not backed by runtime metadata"
            )
        if probe.metadata.get("target") != probe.target_mode:
            failures.append(
                f"{probe.backend}: target mode is not backed by probe metadata"
            )

    target_modes = {probe.target_mode for probe in probes}
    target_mode = next(iter(target_modes)) if len(target_modes) == 1 else None
    if target_mode is None:
        failures.append("probe target mode mismatch")
    probe_definitions = {probe.probe_definition_sha256 for probe in probes}
    probe_definition = (
        next(iter(probe_definitions)) if len(probe_definitions) == 1 else None
    )
    if probe_definition is None:
        failures.append("probe definition mismatch")

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

        minimum_cosine = float(np.clip(min(cosines, default=1.0), -1.0, 1.0))
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
        backend_kinds=backend_kinds,
        target_mode=target_mode,
        probe_definition_sha256=probe_definition,
        probe_sha256={
            probe.backend: _probe_evidence_sha256(probe) for probe in probes
        },
        thresholds=asdict(thresholds),
        metrics=metrics,
        failures=tuple(failures),
    )


def save_backend_gate_result(result: BackendGateResult, path: Path) -> None:
    """Atomically serialize a backend comparison result."""

    _write_json(asdict(result), path)


def load_backend_gate_result(path: Path) -> BackendGateResult:
    """Load a gate result, preserving its immutable authorization fields."""

    try:
        payload = json.loads(Path(path).read_text(encoding="ascii"))
        if not isinstance(payload, dict):
            raise TypeError("top-level JSON must be an object")
        return BackendGateResult(**payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid backend gate result: {error}") from error


def require_matching_passed_gate(
    result: BackendGateResult,
    fingerprint_sha256: str,
    probes: Sequence[BackendProbe | Path | str],
    *,
    expected_target_mode: str = ANALYTIC_TARGET_MODE,
    expected_probe_definition_sha256: str = PROBE_DEFINITION_SHA256,
) -> None:
    """Reject a failed or stale gate before claim-eligible model execution."""

    _validate_gate_result(result)
    if dict(result.thresholds) != asdict(DEFAULT_THRESHOLDS):
        raise ValueError("backend gate did not use the frozen thresholds")
    if not result.passed:
        raise ValueError("backend gate did not pass")
    if result.fingerprint_sha256 != fingerprint_sha256:
        raise ValueError("backend gate fingerprint does not match execution fingerprint")
    if result.target_mode != expected_target_mode:
        raise ValueError("backend gate target mode does not match required target mode")
    if result.probe_definition_sha256 != expected_probe_definition_sha256:
        raise ValueError(
            "backend gate probe definition does not match required probe definition"
        )
    loaded_probes = tuple(
        probe if isinstance(probe, BackendProbe) else load_backend_probe(Path(probe))
        for probe in probes
    )
    recomputed = compare_backend_probes(loaded_probes, DEFAULT_THRESHOLDS)
    if result != recomputed:
        raise ValueError("backend gate does not match recomputed probe evidence")


__all__ = [
    "ANALYTIC_TARGET_MODE",
    "analytic_reference_movie",
    "BackendGateResult",
    "BackendGateThresholds",
    "BackendProbe",
    "DEFAULT_THRESHOLDS",
    "ProbeCase",
    "PROBE_CASE_COUNT",
    "PROBE_DEFINITION_SCHEMA",
    "PROBE_DEFINITION_SHA256",
    "compare_backend_probes",
    "load_backend_gate_result",
    "load_backend_gate_thresholds",
    "load_backend_probe",
    "require_matching_passed_gate",
    "save_backend_gate_result",
    "save_backend_probe",
]
