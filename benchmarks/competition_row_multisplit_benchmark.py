from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common import fmt_float, graph_summary, markdown_table, output_argument, print_report_paths, report_scope, seed_argument, write_report
from benchmarks.competition_event_multisplit_benchmark import objective_description, parse_seeds, score_objective
from benchmarks.competition_row_benchmark import HoldoutResult, evaluate_baseline_holdout, graph_holdout_score, split_group_audit
from evoforest_arch.competition import COMPETITION_ROW_DATASET_NAME, DEFAULT_COMPETITION_DATA_DIR
from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import write_graph
from evoforest_arch.metrics import stratified_group_folds
from evoforest_arch.mutations import MutationDocument, MutationEngine, MutationSpec, RemoveSpec, built_in_mutations
from evoforest_arch.production import ProductionConfig, load_dataset_with_metadata
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.source_mutations import structural_break_source_mutations, validate_source_mutations
from evoforest_arch.splits import make_grouped_split_manifest, split_dataset, subset_inputs


ROW_BASELINE_SPEC = MutationSpec(
    kind="add_alternative",
    target_node="output",
    primitive="adia_row_baseline_outputs",
    alternative_id="adia_row_baseline_output",
    parents=("series",),
    description="Graph-embedded deterministic ADIA row-level baseline features.",
)


@dataclass(frozen=True)
class RowSplitContext:
    seed: int
    method: str
    group_key: str
    train_groups: tuple[int, ...]
    validation_groups: tuple[int, ...]
    test_groups: tuple[int, ...]
    train_inputs: dict[str, object]
    train_y: np.ndarray
    validation_inputs: dict[str, object]
    validation_y: np.ndarray
    test_inputs: dict[str, object]
    test_y: np.ndarray
    baseline: HoldoutResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "split": {
                "manifest_method": self.method,
                "group_key": self.group_key,
                "sizes": {
                    "train": int(self.train_y.shape[0]),
                    "validation": int(self.validation_y.shape[0]),
                    "test": int(self.test_y.shape[0]),
                },
                "group_counts": {
                    "train": len(self.train_groups),
                    "validation": len(self.validation_groups),
                    "test": len(self.test_groups),
                },
            },
            "baseline_validation_auc": float(self.baseline.validation_auc),
        }


@dataclass(frozen=True)
class RowSplitGraphScore:
    seed: int
    train_auc: float
    validation_auc: float
    delta_vs_baseline: float
    alpha: float
    n_features: int
    config: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "train_auc": float(self.train_auc),
            "validation_auc": float(self.validation_auc),
            "delta_vs_baseline": float(self.delta_vs_baseline),
            "alpha": float(self.alpha),
            "n_features": int(self.n_features),
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class RowMultiSplitScore:
    splits: tuple[RowSplitGraphScore, ...]
    mean_validation_auc: float
    min_validation_auc: float
    mean_delta_vs_baseline: float
    min_delta_vs_baseline: float
    objective: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_validation_auc": float(self.mean_validation_auc),
            "min_validation_auc": float(self.min_validation_auc),
            "mean_delta_vs_baseline": float(self.mean_delta_vs_baseline),
            "min_delta_vs_baseline": float(self.min_delta_vs_baseline),
            "objective": float(self.objective),
            "splits": [split.to_dict() for split in self.splits],
        }


def build_report(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 113,
    split_seeds: tuple[int, ...] = (113, 127, 149),
    series_length: int = 96,
    max_samples: int | None = None,
    max_ids: int | None = 1000,
    max_rows_per_id: int | None = 32,
    row_stride: int = 1,
    folds: int = 3,
    max_configurations: int = 64,
    stability_weight: float = 0.5,
    objective_mode: str = "auc",
    prune_tolerance: float = 0.001,
    include_builtin_templates: bool = True,
    include_source_mutations: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ProductionConfig(
        output_dir=Path("unused"),
        dataset_name=COMPETITION_ROW_DATASET_NAME,
        data_dir=data_dir,
        seed=seed,
        competition_series_length=series_length,
        max_samples=max_samples,
        competition_max_ids=max_ids,
        competition_max_rows_per_id=max_rows_per_id,
        competition_row_stride=row_stride,
        folds=folds,
        max_configurations=max_configurations,
        irls_steps=0,
        allow_source_mutations=include_source_mutations,
    )
    inputs, y, metadata = load_dataset_with_metadata(config.dataset_config())
    contexts, outer_split = build_row_split_contexts(inputs, y, outer_seed=seed, split_seeds=split_seeds)
    evaluator_kwargs = {
        "n_splits": int(folds),
        "max_configurations": int(max_configurations),
        "irls_steps": 0,
        "group_key": "sample_id",
    }

    seed_graph = build_seed_graph()
    engine = MutationEngine(allow_source=include_source_mutations)
    row_baseline_graph = engine.apply(seed_graph, ROW_BASELINE_SPEC)
    row_baseline_only_graph = build_output_only_graph(seed_graph, (ROW_BASELINE_SPEC,), allow_source=include_source_mutations)
    builtin_specs = tuple(built_in_mutations()) if include_builtin_templates else ()
    source_specs = structural_break_source_mutations() if include_source_mutations else ()
    source_checks = validate_source_mutations(seed_graph, source_specs, contexts[0].train_inputs) if source_specs else []
    passed_sources = tuple(spec for spec, check in zip(source_specs, source_checks, strict=True) if check.passed)
    suite_specs = (*builtin_specs, *passed_sources)
    if not any(spec.primitive == ROW_BASELINE_SPEC.primitive for spec in suite_specs):
        suite_specs = (ROW_BASELINE_SPEC, *suite_specs)
    suite_graph = engine.apply_document(seed_graph, MutationDocument(add=suite_specs)).graph
    suite_output_only_graph = build_output_only_graph(seed_graph, suite_specs, allow_source=include_source_mutations)

    seed_score = score_graph_across_splits(
        seed_graph,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    row_baseline_graph_score = score_graph_across_splits(
        row_baseline_graph,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    row_baseline_only_score = score_graph_across_splits(
        row_baseline_only_graph,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    suite_score = score_graph_across_splits(
        suite_graph,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    suite_output_only_score = score_graph_across_splits(
        suite_output_only_graph,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    pruned_graph, prune_rows, pruned_score = prune_graph(
        suite_graph,
        [(spec.target_node, spec.alternative_id, 0) for spec in suite_specs],
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
        prune_tolerance=prune_tolerance,
        allow_source=include_source_mutations,
    )
    pruned_output_only_graph, output_only_prune_rows, pruned_output_only_score = prune_graph(
        suite_output_only_graph,
        [(spec.target_node, spec.alternative_id, 0) for spec in suite_specs],
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
        prune_tolerance=prune_tolerance,
        allow_source=include_source_mutations,
    )
    row_baseline_path = write_graph(
        output_dir / "row_baseline_graph.json",
        row_baseline_graph,
        metadata={"benchmark": "competition_row_multisplit_benchmark", "stage": "row_baseline_graph", "score": row_baseline_graph_score.to_dict()},
    )
    row_baseline_only_path = write_graph(
        output_dir / "row_baseline_only_graph.json",
        row_baseline_only_graph,
        metadata={"benchmark": "competition_row_multisplit_benchmark", "stage": "row_baseline_only_graph", "score": row_baseline_only_score.to_dict()},
    )
    suite_path = write_graph(
        output_dir / "row_template_suite_graph.json",
        suite_graph,
        metadata={"benchmark": "competition_row_multisplit_benchmark", "stage": "pre_prune", "score": suite_score.to_dict()},
    )
    suite_output_only_path = write_graph(
        output_dir / "row_template_suite_output_only_graph.json",
        suite_output_only_graph,
        metadata={"benchmark": "competition_row_multisplit_benchmark", "stage": "output_only_pre_prune", "score": suite_output_only_score.to_dict()},
    )
    pruned_path = write_graph(
        output_dir / "pruned_row_template_suite_graph.json",
        pruned_graph,
        metadata={"benchmark": "competition_row_multisplit_benchmark", "stage": "post_prune", "score": pruned_score.to_dict()},
    )
    pruned_output_only_path = write_graph(
        output_dir / "pruned_row_template_suite_output_only_graph.json",
        pruned_output_only_graph,
        metadata={"benchmark": "competition_row_multisplit_benchmark", "stage": "output_only_post_prune", "score": pruned_output_only_score.to_dict()},
    )
    internal_test = internal_test_report(pruned_output_only_graph, contexts, evaluator_kwargs=evaluator_kwargs)
    split_audits = [
        {
            "seed": context.seed,
            "audit": split_group_audit(
                {
                    "train": (context.train_inputs, context.train_y),
                    "validation": (context.validation_inputs, context.validation_y),
                    "test": (context.test_inputs, context.test_y),
                }
            ),
        }
        for context in contexts
    ]
    read_paths = [str(path) for path in metadata.get("read_paths", [])]
    reduced_read_paths = [path for path in read_paths if "reduced" in Path(path).name]
    best_graph = max(
        (
            ("seed_graph", seed_score),
            ("row_baseline_graph", row_baseline_graph_score),
            ("row_baseline_only_graph", row_baseline_only_score),
            ("row_template_suite_graph", suite_score),
            ("pruned_row_template_suite_graph", pruned_score),
            ("row_template_suite_output_only_graph", suite_output_only_score),
            ("pruned_row_template_suite_output_only_graph", pruned_output_only_score),
        ),
        key=lambda row: float(row[1].objective),
    )
    return {
        "benchmark": "competition_row_multisplit_benchmark",
        "scope": report_scope(),
        "seed": int(seed),
        "dataset": metadata,
        "dataset_config": config.dataset_config(),
        "benchmark_config": {
            "folds": int(folds),
            "max_configurations": int(max_configurations),
            "split_seeds": list(split_seeds),
            "stability_weight": float(stability_weight),
            "objective_mode": objective_mode,
            "prune_tolerance": float(prune_tolerance),
            "include_builtin_templates": bool(include_builtin_templates),
            "include_source_mutations": bool(include_source_mutations),
            "objective": objective_description(objective_mode),
        },
        "outer_split": outer_split,
        "split_contexts": [context.to_dict() for context in contexts],
        "split_audits": split_audits,
        "reduced_test_access": {
            "accessed": bool(reduced_read_paths),
            "read_paths": read_paths,
            "reduced_read_paths": reduced_read_paths,
            "checked_paths": ["X_test.reduced.parquet", "y_test.reduced.parquet", "y_test_index.reduced.parquet"],
        },
        "source_mutations": {
            "enabled": bool(include_source_mutations),
            "templates": len(source_specs),
            "passed_repair_checks": sum(1 for check in source_checks if check.passed),
            "checks": [check.to_dict() for check in source_checks],
        },
        "template_suite": {
            "builtin_templates": len(builtin_specs),
            "source_templates": len(passed_sources),
            "added_templates": len(suite_specs),
            "include_builtin_templates": bool(include_builtin_templates),
        },
        "seed_graph": seed_score.to_dict(),
        "row_baseline_graph": {
            **row_baseline_graph_score.to_dict(),
            "path": str(row_baseline_path),
            "graph": graph_summary(row_baseline_graph),
        },
        "row_baseline_only_graph": {
            **row_baseline_only_score.to_dict(),
            "path": str(row_baseline_only_path),
            "graph": graph_summary(row_baseline_only_graph),
        },
        "row_template_suite_graph": {
            **suite_score.to_dict(),
            "path": str(suite_path),
            "graph": graph_summary(suite_graph),
        },
        "row_template_suite_output_only_graph": {
            **suite_output_only_score.to_dict(),
            "path": str(suite_output_only_path),
            "graph": graph_summary(suite_output_only_graph),
        },
        "pruned_row_template_suite_graph": {
            **pruned_score.to_dict(),
            "path": str(pruned_path),
            "graph": graph_summary(pruned_graph),
        },
        "pruned_row_template_suite_output_only_graph": {
            **pruned_output_only_score.to_dict(),
            "path": str(pruned_output_only_path),
            "graph": graph_summary(pruned_output_only_graph),
        },
        "internal_test": internal_test,
        "summary": {
            "best_graph_by_objective": best_graph[0],
            "best_objective": float(best_graph[1].objective),
            "best_mean_validation_auc": float(best_graph[1].mean_validation_auc),
            "best_mean_delta_vs_baseline": float(best_graph[1].mean_delta_vs_baseline),
            "row_baseline_graph_delta_vs_seed": float(row_baseline_graph_score.objective) - float(seed_score.objective),
            "row_baseline_only_delta_vs_row_baseline_graph": float(row_baseline_only_score.objective) - float(row_baseline_graph_score.objective),
            "pruned_delta_vs_suite": float(pruned_score.objective) - float(suite_score.objective),
            "pruned_output_only_delta_vs_suite_output_only": float(pruned_output_only_score.objective) - float(suite_output_only_score.objective),
        },
        "pruning": {
            "attempted": len(prune_rows),
            "removed": sum(1 for row in prune_rows if row.get("removed")),
            "rows": prune_rows,
        },
        "output_only_pruning": {
            "attempted": len(output_only_prune_rows),
            "removed": sum(1 for row in output_only_prune_rows if row.get("removed")),
            "rows": output_only_prune_rows,
        },
    }


def build_output_only_graph(seed_graph: Graph, specs: tuple[MutationSpec, ...], *, allow_source: bool) -> Graph:
    output_removals = tuple(
        RemoveSpec(node.name, alternative.id, "Remove seed output alternative for output-only suite benchmark.")
        for node in seed_graph.nodes.values()
        if node.kind == "output"
        for alternative in node.alternatives
    )
    return MutationEngine(allow_source=allow_source).apply_document(seed_graph, MutationDocument(remove=output_removals, add=specs)).graph


def build_row_split_contexts(
    inputs: dict[str, object],
    y: np.ndarray,
    *,
    outer_seed: int,
    split_seeds: tuple[int, ...],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[tuple[RowSplitContext, ...], dict[str, Any]]:
    contexts: list[RowSplitContext] = []
    n_samples = int(np.asarray(y).shape[0])
    groups = np.asarray(inputs["sample_id"])
    outer_manifest = make_grouped_split_manifest(
        inputs,
        y,
        groups=groups,
        group_key="sample_id",
        seed=int(outer_seed),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    outer_splits = split_dataset(inputs, y, outer_manifest)
    dev_indices = np.asarray(sorted((*outer_manifest.train_indices, *outer_manifest.validation_indices)), dtype=np.int64)
    dev_inputs = subset_inputs(inputs, dev_indices, n_samples=n_samples)
    dev_y = np.asarray(y)[dev_indices]
    dev_groups = np.asarray(dev_inputs["sample_id"])
    test_inputs, test_y = outer_splits["test"]
    test_groups = tuple(int(group) for group in sorted(np.unique(np.asarray(test_inputs["sample_id"])).tolist()))
    n_validation_folds = max(2, int(round(1.0 / max(float(validation_fraction), 1e-8))))
    for split_seed in split_seeds:
        folds = stratified_group_folds(dev_y, dev_groups, n_validation_folds, int(split_seed))
        train_idx, validation_idx = folds[0]
        train_inputs = subset_inputs(dev_inputs, train_idx, n_samples=int(dev_y.shape[0]))
        validation_inputs = subset_inputs(dev_inputs, validation_idx, n_samples=int(dev_y.shape[0]))
        train_y = dev_y[train_idx]
        validation_y = dev_y[validation_idx]
        train_groups = tuple(int(group) for group in sorted(np.unique(np.asarray(train_inputs["sample_id"])).tolist()))
        validation_groups = tuple(int(group) for group in sorted(np.unique(np.asarray(validation_inputs["sample_id"])).tolist()))
        baseline = evaluate_baseline_holdout(train_inputs, train_y, validation_inputs, validation_y)
        contexts.append(
            RowSplitContext(
                seed=int(split_seed),
                method="fixed_outer_test_grouped_development_validation",
                group_key="sample_id",
                train_groups=train_groups,
                validation_groups=validation_groups,
                test_groups=test_groups,
                train_inputs=train_inputs,
                train_y=train_y,
                validation_inputs=validation_inputs,
                validation_y=validation_y,
                test_inputs=test_inputs,
                test_y=test_y,
                baseline=baseline,
            )
        )
    return tuple(contexts), {
        "manifest_method": outer_manifest.method,
        "group_key": outer_manifest.group_key,
        "seed": int(outer_seed),
        "development_groups": len(set(outer_manifest.train_groups) | set(outer_manifest.validation_groups)),
        "test_groups": len(outer_manifest.test_groups),
        "development_samples": int(dev_y.shape[0]),
        "test_samples": int(test_y.shape[0]),
    }


def score_graph_across_splits(
    graph: Graph,
    contexts: tuple[RowSplitContext, ...],
    *,
    evaluator_kwargs: dict[str, Any],
    stability_weight: float,
    objective_mode: str,
) -> RowMultiSplitScore:
    split_scores: list[RowSplitGraphScore] = []
    for context in contexts:
        evaluator = RidgeEvaluator(seed=context.seed, **evaluator_kwargs)
        train_cv = evaluator.evaluate(graph, context.train_inputs, context.train_y, update_graph=False)
        holdout = graph_holdout_score(
            graph,
            context.train_inputs,
            context.train_y,
            context.validation_inputs,
            context.validation_y,
            config=train_cv.config,
        )
        split_scores.append(
            RowSplitGraphScore(
                seed=context.seed,
                train_auc=holdout.train_auc,
                validation_auc=holdout.validation_auc,
                delta_vs_baseline=float(holdout.validation_auc) - float(context.baseline.validation_auc),
                alpha=holdout.alpha,
                n_features=holdout.n_features,
                config=dict(train_cv.config),
            )
        )
    deltas = np.asarray([score.delta_vs_baseline for score in split_scores], dtype=np.float64)
    validation = np.asarray([score.validation_auc for score in split_scores], dtype=np.float64)
    mean_delta = float(np.mean(deltas)) if deltas.size else 0.0
    min_delta = float(np.min(deltas)) if deltas.size else 0.0
    mean_auc = float(np.mean(validation)) if validation.size else 0.5
    min_auc = float(np.min(validation)) if validation.size else 0.5
    return RowMultiSplitScore(
        splits=tuple(split_scores),
        mean_validation_auc=mean_auc,
        min_validation_auc=min_auc,
        mean_delta_vs_baseline=mean_delta,
        min_delta_vs_baseline=min_delta,
        objective=score_objective(
            mean_validation_auc=mean_auc,
            min_validation_auc=min_auc,
            mean_delta_vs_baseline=mean_delta,
            min_delta_vs_baseline=min_delta,
            stability_weight=stability_weight,
            objective_mode=objective_mode,
        ),
    )


def prune_graph(
    graph: Graph,
    accepted: list[tuple[str, str, int]],
    contexts: tuple[RowSplitContext, ...],
    *,
    evaluator_kwargs: dict[str, Any],
    stability_weight: float,
    objective_mode: str,
    prune_tolerance: float,
    allow_source: bool,
) -> tuple[Graph, list[dict[str, Any]], RowMultiSplitScore]:
    engine = MutationEngine(allow_source=allow_source)
    current = graph.clone()
    current_score = score_graph_across_splits(
        current,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    rows: list[dict[str, Any]] = []
    for node_name, alternative_id, step in reversed(accepted):
        node = current.nodes.get(node_name)
        if node is None or len(node.alternatives) <= 1 or not any(alt.id == alternative_id for alt in node.alternatives):
            continue
        try:
            trial = engine.apply_document(
                current,
                MutationDocument(remove=(RemoveSpec(node_name, alternative_id, "Backward row multi-split prune candidate."),)),
            ).graph
            trial_score = score_graph_across_splits(
                trial,
                contexts,
                evaluator_kwargs=evaluator_kwargs,
                stability_weight=stability_weight,
                objective_mode=objective_mode,
            )
            objective_delta = float(trial_score.objective) - float(current_score.objective)
            removed = objective_delta >= -float(prune_tolerance)
            rows.append(
                {
                    "step": int(step),
                    "target_node": node_name,
                    "alternative_id": alternative_id,
                    "removed": bool(removed),
                    "objective_before": float(current_score.objective),
                    "objective_after": float(trial_score.objective),
                    "objective_delta": float(objective_delta),
                    "mean_validation_auc_after": float(trial_score.mean_validation_auc),
                    "mean_delta_after": float(trial_score.mean_delta_vs_baseline),
                    "min_delta_after": float(trial_score.min_delta_vs_baseline),
                }
            )
            if removed:
                current = trial
                current_score = trial_score
        except Exception as exc:
            rows.append(
                {
                    "step": int(step),
                    "target_node": node_name,
                    "alternative_id": alternative_id,
                    "removed": False,
                    "failed": True,
                    "error": str(exc),
                }
            )
    return current, rows, current_score


def internal_test_report(
    graph: Graph,
    contexts: tuple[RowSplitContext, ...],
    *,
    evaluator_kwargs: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for context in contexts:
        baseline = evaluate_baseline_holdout(context.train_inputs, context.train_y, context.test_inputs, context.test_y)
        evaluator = RidgeEvaluator(seed=context.seed, **evaluator_kwargs)
        train_cv = evaluator.evaluate(graph, context.train_inputs, context.train_y, update_graph=False)
        score = graph_holdout_score(graph, context.train_inputs, context.train_y, context.test_inputs, context.test_y, config=train_cv.config)
        rows.append(
            {
                "seed": context.seed,
                "baseline_auc": float(baseline.validation_auc),
                "graph_auc": float(score.validation_auc),
                "delta_vs_baseline": float(score.validation_auc) - float(baseline.validation_auc),
            }
        )
    deltas = np.asarray([row["delta_vs_baseline"] for row in rows], dtype=np.float64)
    aucs = np.asarray([row["graph_auc"] for row in rows], dtype=np.float64)
    return {
        "selection_used": False,
        "mean_graph_auc": float(np.mean(aucs)) if aucs.size else 0.5,
        "mean_delta_vs_baseline": float(np.mean(deltas)) if deltas.size else 0.0,
        "min_delta_vs_baseline": float(np.min(deltas)) if deltas.size else 0.0,
        "splits": rows,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    graph_rows = []
    for name in (
        "seed_graph",
        "row_baseline_graph",
        "row_baseline_only_graph",
        "row_template_suite_graph",
        "pruned_row_template_suite_graph",
        "row_template_suite_output_only_graph",
        "pruned_row_template_suite_output_only_graph",
    ):
        row = payload[name]
        graph_rows.append(
            [
                name,
                fmt_float(row["mean_validation_auc"]),
                fmt_float(row["mean_delta_vs_baseline"]),
                fmt_float(row["min_validation_auc"]),
                fmt_float(row["objective"]),
            ]
        )
    prune_rows = [
        [
            row["alternative_id"],
            "yes" if row.get("removed") else "no",
            fmt_float(row.get("objective_before")),
            fmt_float(row.get("objective_after")),
            fmt_float(row.get("objective_delta")),
        ]
        for row in payload["pruning"]["rows"]
    ]
    dataset = payload["dataset"]
    internal = payload["internal_test"]
    return "\n\n".join(
        [
            "# Competition Row Multi-Split Benchmark",
            str(payload["scope"]),
            (
                f"Dataset: `{dataset['name']}` from `{dataset['data_dir']}`, rows=`{dataset['n_samples']}`, "
                f"ids=`{dataset['n_ids']}`, max_ids=`{dataset['max_ids']}`, max_rows_per_id=`{dataset['max_rows_per_id']}`."
            ),
            f"Config: `{payload['benchmark_config']}`",
            f"Reduced test accessed: `{payload['reduced_test_access']['accessed']}`.",
            (
                f"Best graph by objective: `{payload['summary']['best_graph_by_objective']}` with mean validation AUC "
                f"`{fmt_float(payload['summary']['best_mean_validation_auc'])}` and mean delta "
                f"`{fmt_float(payload['summary']['best_mean_delta_vs_baseline'])}`."
            ),
            (
                f"Internal non-selection test audit for pruned output-only suite: mean graph AUC `{fmt_float(internal['mean_graph_auc'])}`, "
                f"mean delta `{fmt_float(internal['mean_delta_vs_baseline'])}`, minimum delta `{fmt_float(internal['min_delta_vs_baseline'])}`."
            ),
            markdown_table(["Graph", "Mean Val AUC", "Mean Delta", "Min Val AUC", "Objective"], graph_rows),
            markdown_table(["Pruned Alternative", "Removed", "Objective Before", "Objective After", "Delta"], prune_rows),
        ]
    )


def run(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 113,
    split_seeds: tuple[int, ...] = (113, 127, 149),
    series_length: int = 96,
    max_samples: int | None = None,
    max_ids: int | None = 1000,
    max_rows_per_id: int | None = 32,
    row_stride: int = 1,
    folds: int = 3,
    max_configurations: int = 64,
    objective_mode: str = "auc",
    include_builtin_templates: bool = True,
    include_source_mutations: bool = False,
) -> tuple[Path, Path]:
    payload = build_report(
        output_dir,
        data_dir=data_dir,
        seed=seed,
        split_seeds=split_seeds,
        series_length=series_length,
        max_samples=max_samples,
        max_ids=max_ids,
        max_rows_per_id=max_rows_per_id,
        row_stride=row_stride,
        folds=folds,
        max_configurations=max_configurations,
        objective_mode=objective_mode,
        include_builtin_templates=include_builtin_templates,
        include_source_mutations=include_source_mutations,
    )
    return write_report(output_dir, "competition_row_multisplit_benchmark", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a grouped multi-split row/time-level ADIA-style benchmark.")
    output_argument(parser)
    seed_argument(parser, default=113)
    parser.add_argument("--split-seeds", type=parse_seeds, default=(113, 127, 149), help="Comma-separated grouped split seeds.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_COMPETITION_DATA_DIR)
    parser.add_argument("--series-length", type=int, default=96)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-ids", type=int, default=1000)
    parser.add_argument("--max-rows-per-id", type=int, default=32)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-configurations", type=int, default=64)
    parser.add_argument("--objective-mode", choices=("delta", "auc"), default="auc")
    parser.add_argument("--disable-builtins", action="store_true")
    parser.add_argument("--include-source-mutations", action="store_true")
    args = parser.parse_args(argv)
    print_report_paths(
        run(
            args.output,
            data_dir=args.data_dir,
            seed=args.seed,
            split_seeds=args.split_seeds,
            series_length=args.series_length,
            max_samples=args.max_samples,
            max_ids=args.max_ids,
            max_rows_per_id=args.max_rows_per_id,
            row_stride=args.row_stride,
            folds=args.folds,
            max_configurations=args.max_configurations,
            objective_mode=args.objective_mode,
            include_builtin_templates=not args.disable_builtins,
            include_source_mutations=args.include_source_mutations,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
