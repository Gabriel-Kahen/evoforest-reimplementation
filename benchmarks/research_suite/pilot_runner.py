from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.common import markdown_table, write_report
from benchmarks.research_suite.compositional_dags import generate_benchmark
from benchmarks.research_suite.controlled_benchmark import ControlledRunConfig, run_controlled_benchmark
from benchmarks.research_suite.evolution_experiment import EvolutionExperimentConfig, run_evolution_experiment
from benchmarks.research_suite.study_spec import StudySpec, pilot_spec, write_spec
from benchmarks.research_suite.transfer_experiment import run_transfer_experiment
from evoforest_arch.paper_agents import PaperAgentBundle


def run_medium_pilot(
    output_dir: Path,
    spec: StudySpec | None = None,
    *,
    agents: PaperAgentBundle,
) -> dict[str, object]:
    spec = spec or pilot_spec()
    if spec.status != "pilot":
        raise ValueError("The medium pilot requires a pilot StudySpec.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_spec(output_dir / "pilot_spec.json", spec)
    started = time.monotonic()

    controlled = run_controlled_benchmark(
        ControlledRunConfig(
            task_names=spec.task_families,
            seeds=spec.data_seeds,
            n_train=spec.n_train,
            n_validation=spec.n_validation,
            n_test=spec.n_test,
            random_features=128,
        )
    )

    evolution_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for task_name in spec.task_families:
        for data_seed in spec.data_seeds:
            dataset = generate_benchmark(
                task_name,
                seed=data_seed,
                n_train=spec.n_train,
                n_validation=spec.n_validation,
                n_test=spec.n_test,
            )
            for search_seed in spec.search_seeds:
                try:
                    rows, selected = run_evolution_experiment(
                        dataset,
                        output_dir / "evolution",
                        agents=agents,
                        config=EvolutionExperimentConfig(
                            steps=spec.evolution_steps,
                            max_configurations=spec.max_configurations,
                            screening_finalists=spec.screening_finalists,
                            scientist_name="full",
                            search_seed=search_seed,
                        ),
                    )
                    evolution_rows.extend(row.to_dict() for row in rows)
                    evolution_rows[-1]["selected_validation_nrmse"] = selected.validation_nrmse
                except Exception as error:  # Pilot must report reliability rather than abort all tasks.
                    failures.append(
                        {
                            "task": task_name,
                            "data_seed": data_seed,
                            "search_seed": search_seed,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

    target = generate_benchmark(
        spec.task_families[0],
        seed=spec.data_seeds[0],
        n_train=spec.n_train,
        n_validation=spec.n_validation,
        n_test=spec.n_test,
    )
    donor = generate_benchmark(
        spec.task_families[1],
        seed=spec.data_seeds[0] + 1,
        n_train=spec.n_train,
        n_validation=spec.n_validation,
        n_test=spec.n_test,
    )
    transfer = run_transfer_experiment(
        target,
        generate_benchmark(
            spec.task_families[2],
            seed=spec.data_seeds[0] + 2,
            n_train=spec.n_train,
            n_validation=spec.n_validation,
            n_test=spec.n_test,
        ),
        output_dir / "transfer",
        agents=agents,
        source_steps=spec.evolution_steps,
        target_steps=spec.evolution_steps,
        max_configurations=spec.max_configurations,
        unrelated_source=donor,
    )
    elapsed = time.monotonic() - started
    transfer_payload = [asdict(row) for row in transfer]
    return {
        "pilot_spec_fingerprint": spec.fingerprint(),
        "elapsed_seconds": elapsed,
        "controlled": [row.to_dict() for row in controlled],
        "evolution": evolution_rows,
        "transfer": transfer_payload,
        "failures": failures,
        "assessment": _assess(controlled, evolution_rows, transfer_payload, failures, elapsed),
    }


def _assess(
    controlled: list[object],
    evolution: list[dict[str, object]],
    transfer: list[dict[str, object]],
    failures: list[dict[str, object]],
    elapsed: float,
) -> dict[str, object]:
    controlled_rows = [row.to_dict() for row in controlled]  # type: ignore[attr-defined]
    random_interp = [
        float(row["metrics"]["nrmse"])
        for row in controlled_rows
        if row["method"] == "random_features_ridge" and row["split_id"] == "interpolation"
    ]
    evolved_interp = [
        float(row["metrics"]["nrmse"])
        for row in evolution
        if row["split_id"] == "interpolation"
    ]
    transfer_scores = [float(row["selected_validation_nrmse"]) for row in transfer]
    return {
        "random_feature_interpolation_median_nrmse": float(np.median(random_interp)),
        "evolved_interpolation_median_nrmse": None if not evolved_interp else float(np.median(evolved_interp)),
        "transfer_score_range": [min(transfer_scores), max(transfer_scores)],
        "failure_rate": len(failures) / max(len(evolution) // 2 + len(failures), 1),
        "elapsed_seconds": elapsed,
        "task_difficulty_flag": (
            "too_easy" if random_interp and max(random_interp) < 0.1 else
            "too_hard" if evolved_interp and min(evolved_interp) > 1.5 else
            "usable"
        ),
        "transfer_separation_observed": max(transfer_scores) - min(transfer_scores) > 1e-6,
    }


def markdown_report(payload: dict[str, object]) -> str:
    assessment = payload["assessment"]
    assert isinstance(assessment, dict)
    failure_rows = [
        [row["task"], row["data_seed"], row["search_seed"], row["error"]]
        for row in payload["failures"]  # type: ignore[union-attr]
    ]
    lines = [
        "# EvoForest Medium Pilot",
        "",
        f"Spec fingerprint: `{payload['pilot_spec_fingerprint']}`",
        f"Elapsed: `{float(payload['elapsed_seconds']):.2f}s`",
        "",
        "## Assessment",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in assessment.items())
    if failure_rows:
        lines.extend(["", "## Failures", "", markdown_table(["Task", "Data seed", "Search seed", "Error"], failure_rows)])
    return "\n".join(lines)


def run(output_dir: Path, agents: PaperAgentBundle) -> tuple[Path, Path]:
    payload = run_medium_pilot(output_dir, agents=agents)
    return write_report(output_dir, "medium_pilot", payload, markdown_report(payload))
