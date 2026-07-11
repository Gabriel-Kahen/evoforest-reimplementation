from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmarks.research_suite.compositional_dags import BenchmarkDataset
from benchmarks.research_suite.evolution_experiment import EvolutionExperimentConfig, run_evolution_experiment
from benchmarks.research_suite.module_transfer import TransferReport, transfer_alternatives
from evoforest_arch.graph_io import graph_from_path
from evoforest_arch.paper_agents import PaperAgentBundle
from evoforest_arch.seed import build_seed_graph


@dataclass(frozen=True)
class TransferConditionResult:
    condition: str
    selected_validation_nrmse: float
    test_rows: tuple[dict[str, object], ...]
    transfer_report: dict[str, object] | None


def run_transfer_experiment(
    source: BenchmarkDataset,
    target: BenchmarkDataset,
    output_dir: Path,
    *,
    agents: PaperAgentBundle,
    source_steps: int = 12,
    target_steps: int = 12,
    max_configurations: int = 32,
    unrelated_source: BenchmarkDataset | None = None,
) -> list[TransferConditionResult]:
    """Compare scratch, related-module, whole-graph, and optional unrelated transfer."""

    source_rows, source_selected = run_evolution_experiment(
        source,
        output_dir / "source_runs",
        agents=agents,
        config=EvolutionExperimentConfig(
            steps=source_steps,
            max_configurations=max_configurations,
            scientist_name="source",
        ),
    )
    del source_rows
    source_graph = graph_from_path(source_selected.path)

    conditions: list[tuple[str, object, TransferReport | None]] = [("scratch", build_seed_graph(), None)]
    module_graph = build_seed_graph()
    related_report = transfer_alternatives(
        source_graph,
        module_graph,
        node_kinds={"intermediate", "callable", "output"},
        copy_missing_globals=True,
    )
    conditions.append(("related_modules", module_graph, related_report))
    conditions.append(("whole_graph", source_graph.clone(), None))

    if unrelated_source is not None:
        _, unrelated_selected = run_evolution_experiment(
            unrelated_source,
            output_dir / "unrelated_source_runs",
            agents=agents,
            config=EvolutionExperimentConfig(
                steps=source_steps,
                max_configurations=max_configurations,
                scientist_name="unrelated_source",
            ),
        )
        unrelated_graph = build_seed_graph()
        unrelated_report = transfer_alternatives(
            graph_from_path(unrelated_selected.path),
            unrelated_graph,
            node_kinds={"intermediate", "callable", "output"},
            copy_missing_globals=True,
        )
        conditions.append(("unrelated_modules", unrelated_graph, unrelated_report))

    results: list[TransferConditionResult] = []
    for condition, initial_graph, report in conditions:
        rows, selected = run_evolution_experiment(
            target,
            output_dir / "target_runs",
            agents=agents,
            config=EvolutionExperimentConfig(
                steps=target_steps,
                max_configurations=max_configurations,
                scientist_name=condition,
            ),
            initial_graph=initial_graph,  # type: ignore[arg-type]
        )
        results.append(
            TransferConditionResult(
                condition=condition,
                selected_validation_nrmse=selected.validation_nrmse,
                test_rows=tuple(row.to_dict() for row in rows),
                transfer_report=None if report is None else report.to_dict(),
            )
        )
    return results
