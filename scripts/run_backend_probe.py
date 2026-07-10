#!/usr/bin/env python3
"""Evaluate a bounded, fingerprint-bound observable probe on one backend."""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from graphite_stage_transition.backend_gate import (
    ANALYTIC_TARGET_MODE,
    BackendProbe,
    ProbeCase,
    PROBE_DEFINITION_SHA256,
    analytic_reference_movie,
    save_backend_probe,
)
from graphite_stage_transition.config import load_config
from graphite_stage_transition.geometry import make_circle_grid
from graphite_stage_transition.execution import (
    validate_canonical_environment,
    verify_execution_fingerprint,
)
from graphite_stage_transition.inversion import ParameterTransform
from graphite_stage_transition.observables import (
    make_observable_geometry,
    observable_residual_vector,
    physics_observables,
)
from graphite_stage_transition.protocols import build_protocol
from graphite_stage_transition.solver import CHRParameters, simulate


PARAMETER_NAMES = ("mobility", "barrier", "kappa", "reaction_rate")


def _resolve_config_path(manifest_path: Path, manifest: dict) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    configured = Path(manifest["metadata"]["config_path"])
    candidates = (
        configured,
        manifest_path.parent / configured,
        manifest_path.parent.parent / configured,
        project_root / configured,
    )
    try:
        return next(path for path in candidates if path.is_file())
    except StopIteration as error:
        raise FileNotFoundError(f"cannot resolve benchmark config {configured}") from error


def _development_cases(manifest: dict, maximum: int) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    for record in manifest["records"]:
        if record["split"] != "development" or float(record["noise_fraction"]) != 0.0:
            continue
        if record["case_id"] in seen:
            continue
        selected.append(record)
        seen.add(record["case_id"])
        if len(selected) == maximum:
            break
    if len(selected) != maximum:
        raise ValueError(
            f"requested {maximum} clean development cases, found {len(selected)}"
        )
    return selected


def _probe_case(
    manifest_path: Path,
    manifest: dict,
    config,
    record: dict,
    analytic_targets: bool,
):
    grid = make_circle_grid(config.grid)
    protocol = build_protocol(config.protocol, config.solver.dt)
    if analytic_targets:
        observations = jnp.asarray(
            analytic_reference_movie(
                x=grid.x,
                y=grid.y,
                mask=grid.mask,
                currents=protocol.current,
                save_indices=protocol.save_indices,
                dt=protocol.dt,
                cell_area=grid.cell_area,
                stage2=config.model.stage2,
                stage1=config.model.stage1,
            ),
            dtype=jnp.float64,
        )
    else:
        archive_path = manifest_path.parent / record["observation_path"]
        with np.load(archive_path) as archive:
            observations = jnp.asarray(archive["concentration"], dtype=jnp.float64)
            observation_times = np.asarray(archive["times"], dtype=np.float64)
        saved_times = np.asarray(protocol.times[protocol.save_indices], dtype=np.float64)
        if not np.array_equal(saved_times, observation_times):
            raise ValueError(f"{record['case_id']}: observation times do not match protocol")

    bounds = manifest["metadata"]["bounds"]
    transform = ParameterTransform(
        lower=tuple(bounds[name][0] for name in PARAMETER_NAMES),
        upper=tuple(bounds[name][1] for name in PARAMETER_NAMES),
        stage2=config.model.stage2,
        stage1=config.model.stage1,
    )
    parameter_values = record.get("parameters", {})
    evaluation_parameters = CHRParameters(
        *(parameter_values.get(name, getattr(config.model, name)) for name in PARAMETER_NAMES),
        config.model.stage2,
        config.model.stage1,
    )
    solver = replace(
        config.solver,
        perturbation_amplitude=0.0,
        seed=int(record["seed"]),
    )
    initial = observations[0]

    # Bind the fixed geometry outside the differentiated function while preserving
    # one forward solve for the objective, observable blocks, and gradient.
    geometry = make_observable_geometry(grid)
    observed = physics_observables(
        observations,
        geometry,
        config.model.stage2,
        config.model.stage1,
    )

    def bound_objective(values):
        parameters = transform.from_unconstrained(values)
        prediction = simulate(
            grid,
            protocol,
            parameters,
            solver,
            initial_concentration=initial,
            seed=solver.seed,
        )
        predicted = physics_observables(
            prediction.concentration,
            geometry,
            config.model.stage2,
            config.model.stage1,
        )
        residual = observable_residual_vector(predicted, observed)
        mask = grid.mask[None, ...]
        width = config.model.stage1 - config.model.stage2
        below = jax.nn.relu(config.model.stage2 - prediction.concentration)
        above = jax.nn.relu(prediction.concentration - config.model.stage1)
        normalizer = prediction.concentration.shape[0] * grid.active_count * width**2
        bounds_loss = jnp.sum(jnp.where(mask, below**2 + above**2, 0.0)) / normalizer
        objective = jnp.mean(residual**2) + config.inversion.bound_penalty * bounds_loss
        return objective, predicted

    evaluate = jax.jit(jax.value_and_grad(bound_objective, has_aux=True))
    started = time.perf_counter()
    (objective, observables), gradient = evaluate(
        transform.to_unconstrained(evaluation_parameters)
    )
    jax.block_until_ready(gradient)
    runtime = time.perf_counter() - started
    blocks = {
        "radial_profile": np.asarray(observables.radial_profile),
        "structure_power": np.asarray(observables.structure_power),
        "boundary_excess": np.asarray(observables.boundary_excess),
    }
    case = ProbeCase(
        case_id=record["case_id"],
        observable_blocks=blocks,
        primary_objective=float(objective),
        gradient=tuple(np.asarray(gradient, dtype=np.float64)),
    )
    return case, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--backend-name", required=True)
    parser.add_argument(
        "--backend-kind",
        choices=("canonical_cpu", "gpu"),
        required=True,
    )
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--analytic-targets", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if not args.analytic_targets:
        parser.error("--analytic-targets is required by the frozen probe definition")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    config_path = _resolve_config_path(manifest_path, manifest)
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parents[1]
    requirements_path = project_root / "requirements" / "canonical-cpu.txt"
    python_version_path = project_root / ".python-version"
    fingerprint_record = json.loads(args.fingerprint.read_text(encoding="ascii"))
    fingerprint_sha256 = verify_execution_fingerprint(
        fingerprint_record,
        source_root=project_root / "src" / "graphite_stage_transition",
        manifest_path=manifest_path,
        config_path=config_path,
        requirements_path=requirements_path,
        python_version_path=python_version_path,
    )
    if args.backend_kind == "canonical_cpu":
        validate_canonical_environment(python_version_path, requirements_path)
    records = _development_cases(manifest, args.max_cases)
    cases = []
    runtimes = {}
    for record in records:
        case, runtime = _probe_case(
            manifest_path,
            manifest,
            config,
            record,
            args.analytic_targets,
        )
        cases.append(case)
        runtimes[case.case_id] = runtime
        print(f"{case.case_id}: {runtime:.3f} s", flush=True)

    devices = [str(device) for device in jax.devices()]
    probe = BackendProbe(
        backend=args.backend_name,
        backend_kind=args.backend_kind,
        fingerprint_sha256=fingerprint_sha256,
        target_mode=ANALYTIC_TARGET_MODE,
        probe_definition_sha256=PROBE_DEFINITION_SHA256,
        cases=tuple(cases),
        metadata={
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
            "numpy": version("numpy"),
            "python": platform.python_version(),
            "jax_default_backend": jax.default_backend(),
            "devices": devices,
            "runtime_seconds": runtimes,
            "parameter_point": "manifest_case_parameters",
            "target": ANALYTIC_TARGET_MODE,
            "parameter_order": list(PARAMETER_NAMES),
        },
    )
    save_backend_probe(probe, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
