from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from benchmarks.research_suite.compositional_dags import BenchmarkDataset, DatasetSplit
from benchmarks.research_suite.evoforest_model import fit_frozen_evoforest_regressor
from benchmarks.research_suite.metrics import nrmse
from benchmarks.research_suite.protocol import BudgetUsage, DatasetPartition, EvaluationProtocol, ExperimentResultRow
from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import graph_from_path
from evoforest_arch.paper_agents import PaperAgentBundle
from evoforest_arch.seed import build_seed_graph


@dataclass(frozen=True)
class EvolutionExperimentConfig:
    steps: int = 12
    max_configurations: int = 32
    screening_finalists: int = 8
    refine_globals: bool = False
    scientist_name: str = "full_diagnostics"
    search_seed: int | None = None


@dataclass(frozen=True)
class SelectedArchiveGraph:
    path: Path
    config: dict[str, str]
    validation_nrmse: float
    archive_version: int


def run_evolution_experiment(
    dataset: BenchmarkDataset,
    output_dir: Path,
    *,
    agents: PaperAgentBundle,
    config: EvolutionExperimentConfig = EvolutionExperimentConfig(),
    initial_graph: Graph | None = None,
) -> tuple[list[ExperimentResultRow], SelectedArchiveGraph]:
    """Evolve on search-train, select an archive graph on validation, test once."""

    search_seed = dataset.seed if config.search_seed is None else int(config.search_seed)
    run_dir = output_dir / dataset.spec.name / f"data_{dataset.seed}" / f"search_{search_seed}" / config.scientist_name
    evaluation_counts = {"exact": 0, "screening": 0}

    def count_evaluation(payload: dict[str, object]) -> None:
        if payload.get("phase") == "configuration_evaluated":
            evaluation_counts["exact"] += 1
        elif payload.get("phase") == "configuration_screened":
            evaluation_counts["screening"] += 1

    evaluator = RidgeEvaluator(
        n_splits=3,
        seed=search_seed,
        scorer="rmse",
        max_configurations=config.max_configurations,
        screening_finalists=config.screening_finalists,
        refine_globals=config.refine_globals,
        progress_callback=count_evaluation,
    )
    loop = EvolutionLoop(
        initial_graph or build_seed_graph(),
        evaluator=evaluator,
        scientist=agents.scientist,
        engineer=agents.engineer,
        memorandum_agent=agents.memorandum,
        seed=search_seed,
    )
    loop.run({"x": dataset.train.X}, dataset.train.y, config.steps, run_dir)
    selected = select_archive_graph(run_dir, dataset)
    graph = graph_from_path(selected.path)

    final_x = np.vstack((dataset.train.X, dataset.validation.X))
    final_y = np.concatenate((dataset.train.y, dataset.validation.y))
    model = fit_frozen_evoforest_regressor(graph, selected.config, {"x": final_x}, final_y)
    rows: list[ExperimentResultRow] = []
    for regime, split in (
        ("interpolation", dataset.test_interpolation),
        ("extrapolation", dataset.test_extrapolation),
    ):
        protocol = _protocol(dataset, split, regime)
        token = protocol.finalize(
            f"evoforest:{config.scientist_name}:{dataset.spec.name}:{dataset.seed}:v{selected.archive_version}"
        )
        result = protocol.evaluate_test(token, lambda X: model.predict({"x": X}))
        rows.append(
            ExperimentResultRow(
                task_id=dataset.spec.name,
                task_family=dataset.spec.name,
                method=f"evoforest_{config.scientist_name}",
                seed=search_seed,
                split_id=regime,
                status="completed",
                metrics=result.metrics,
                usage=BudgetUsage(
                    exact_evaluations=evaluation_counts["exact"],
                    screening_evaluations=evaluation_counts["screening"],
                ),
                graph_nodes=len(graph.nodes),
                metadata={
                    "candidate_steps": config.steps,
                    "data_seed": dataset.seed,
                    "search_seed": search_seed,
                    "exact_configuration_evaluations": evaluation_counts["exact"],
                    "screening_configuration_evaluations": evaluation_counts["screening"],
                    "archive_version": selected.archive_version,
                    "selection_validation_nrmse": selected.validation_nrmse,
                    "config": selected.config,
                },
            )
        )
    return rows, selected


def select_archive_graph(run_dir: Path, dataset: BenchmarkDataset) -> SelectedArchiveGraph:
    candidates = score_archive_graphs(run_dir, dataset)
    if not candidates:
        raise RuntimeError(f"Evolution run produced no archive graphs in {run_dir}.")
    return min(candidates, key=lambda item: (item.validation_nrmse, item.archive_version))


def score_archive_graphs(run_dir: Path, dataset: BenchmarkDataset) -> list[SelectedArchiveGraph]:
    candidates: list[SelectedArchiveGraph] = []
    for path in sorted((run_dir / "archive").glob("global_best_v*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph = graph_from_path(path)
        result = payload.get("result", {})
        config = dict(result.get("config", payload.get("config", {})))
        model = fit_frozen_evoforest_regressor(graph, config, {"x": dataset.train.X}, dataset.train.y)
        prediction = model.predict({"x": dataset.validation.X})
        candidates.append(
            SelectedArchiveGraph(
                path=path,
                config=config,
                validation_nrmse=nrmse(dataset.validation.y, prediction),
                archive_version=int(payload.get("version", len(candidates))),
            )
        )
    return candidates


def _protocol(dataset: BenchmarkDataset, test: DatasetSplit, regime: str) -> EvaluationProtocol:
    def part(name: str, split: DatasetSplit, label: str) -> DatasetPartition:
        ids = tuple(f"{dataset.spec.name}:{dataset.seed}:{label}:{index}" for index in range(split.y.shape[0]))
        return DatasetPartition(name, split.X, split.y, ids)

    return EvaluationProtocol(
        part("search_train", dataset.train, "train"),
        part("selection_validation", dataset.validation, "validation"),
        part(f"test_{regime}", test, f"test:{regime}"),
    )
