from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common import fmt_float, markdown_table, output_argument, print_report_paths, report_scope, seed_argument, write_report
from benchmarks.competition_event_benchmark import (
    PredictionHoldoutResult,
    evaluate_structural_break_baseline,
    graph_holdout_score,
    split_group_audit,
)
from evoforest_arch.agents import EngineerAgent, ScientistAgent
from evoforest_arch.competition import COMPETITION_DATASET_NAME, DEFAULT_COMPETITION_DATA_DIR
from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import write_graph
from evoforest_arch.metrics import roc_auc_score
from evoforest_arch.metrics import stratified_group_folds
from evoforest_arch.mutations import MutationDocument, MutationEngine, RemoveSpec, built_in_mutations
from evoforest_arch.production import ProductionConfig, load_dataset_with_metadata
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.source_mutations import structural_break_source_mutations, validate_source_mutations
from evoforest_arch.splits import make_grouped_split_manifest, split_dataset, subset_inputs


@dataclass(frozen=True)
class SplitContext:
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
    baseline: PredictionHoldoutResult

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
class SplitGraphScore:
    seed: int
    train_auc: float
    validation_auc: float
    delta_vs_baseline: float
    alpha: float
    n_features: int
    config: dict[str, str]
    train_predictions: np.ndarray
    validation_predictions: np.ndarray

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
class MultiSplitScore:
    splits: tuple[SplitGraphScore, ...]
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


@dataclass(frozen=True)
class ArchiveEntry:
    name: str
    step: int
    accepted: bool
    graph: Graph
    score: MultiSplitScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "step": int(self.step),
            "accepted": bool(self.accepted),
            **self.score.to_dict(),
        }


def build_report(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 211,
    split_seeds: tuple[int, ...] = (211, 223, 307),
    series_length: int = 160,
    max_samples: int | None = None,
    steps: int = 96,
    folds: int = 3,
    max_configurations: int = 64,
    min_objective_improvement: float = 1e-4,
    stability_weight: float = 0.5,
    objective_mode: str = "delta",
    prune_tolerance: float = 0.001,
    include_source_mutations: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ProductionConfig(
        output_dir=Path("unused"),
        dataset_name=COMPETITION_DATASET_NAME,
        data_dir=data_dir,
        seed=seed,
        competition_series_length=series_length,
        max_samples=max_samples,
        folds=folds,
        max_configurations=max_configurations,
        irls_steps=0,
        min_train_improvement=-1.0,
        min_validation_improvement=min_objective_improvement,
        allow_source_mutations=include_source_mutations,
    )
    inputs, y, metadata = load_dataset_with_metadata(config.dataset_config())
    contexts, outer_split = build_split_contexts(inputs, y, outer_seed=seed, split_seeds=split_seeds)
    evaluator_kwargs = {
        "n_splits": int(folds),
        "max_configurations": int(max_configurations),
        "irls_steps": 0,
        "group_key": "sample_id",
    }

    scientist = ScientistAgent()
    source_specs = structural_break_source_mutations() if include_source_mutations else ()
    source_checks = validate_source_mutations(build_seed_graph(), source_specs, contexts[0].train_inputs) if source_specs else []
    passed_source_specs = tuple(spec for spec, check in zip(source_specs, source_checks) if check.passed)
    engineer = EngineerAgent(templates=passed_source_specs + tuple(built_in_mutations()))
    mutation_engine = MutationEngine(allow_source=include_source_mutations)

    current_graph = build_seed_graph()
    seed_score = score_graph_across_splits(
        current_graph,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    best_score = seed_score
    archive: list[ArchiveEntry] = [ArchiveEntry("seed_graph", 0, True, current_graph.clone(), seed_score)]
    rows: list[dict[str, Any]] = []
    accepted: list[tuple[str, str, int]] = []
    attempted_signatures: set[tuple[str, str, tuple[str, ...], str]] = set()
    rng = np.random.default_rng(seed)
    feedback_result = RidgeEvaluator(seed=seed, **evaluator_kwargs).evaluate(current_graph, contexts[0].train_inputs, contexts[0].train_y, update_graph=True)

    for step in range(1, int(steps) + 1):
        hypotheses = scientist.generate(current_graph, feedback_result)
        document = synthesize_unseen_document(
            engineer,
            current_graph,
            feedback_result,
            hypotheses,
            step=step,
            rng=rng,
            attempted_signatures=attempted_signatures,
        )
        add = document.add[0] if document.add else None
        if add:
            attempted_signatures.add(add_signature(add))
        try:
            application = mutation_engine.apply_document(current_graph, document)
            candidate_graph = application.graph
            candidate_score = score_graph_across_splits(
                candidate_graph,
                contexts,
                evaluator_kwargs=evaluator_kwargs,
                stability_weight=stability_weight,
                objective_mode=objective_mode,
            )
            objective_delta = float(candidate_score.objective) - float(best_score.objective)
            accepted_step = objective_delta > float(min_objective_improvement)
            archive.append(ArchiveEntry(add.alternative_id if add else f"step_{step}", step, accepted_step, candidate_graph.clone(), candidate_score))
            row = {
                "step": int(step),
                "accepted": bool(accepted_step),
                "failed": False,
                "target_node": add.target_node if add else "",
                "primitive": add.primitive if add else "",
                "alternative_id": add.alternative_id if add else "",
                "source_backed": bool(add and add.source),
                "mean_validation_auc": float(candidate_score.mean_validation_auc),
                "mean_delta_vs_baseline": float(candidate_score.mean_delta_vs_baseline),
                "min_delta_vs_baseline": float(candidate_score.min_delta_vs_baseline),
                "objective": float(candidate_score.objective),
                "objective_delta_vs_best": float(objective_delta),
                "split_deltas": [float(split.delta_vs_baseline) for split in candidate_score.splits],
                "maintenance": application.maintenance.to_dict(),
            }
            if accepted_step:
                current_graph = candidate_graph
                best_score = candidate_score
                if add:
                    accepted.append((add.target_node, add.alternative_id, step))
                feedback_result = RidgeEvaluator(seed=seed + step, **evaluator_kwargs).evaluate(
                    current_graph,
                    contexts[0].train_inputs,
                    contexts[0].train_y,
                    update_graph=True,
                )
        except Exception as exc:
            row = {
                "step": int(step),
                "accepted": False,
                "failed": True,
                "error": str(exc),
                "target_node": add.target_node if add else "",
                "primitive": add.primitive if add else "",
                "alternative_id": add.alternative_id if add else "",
                "source_backed": bool(add and add.source),
            }
        rows.append(row)

    consensus_graph = current_graph.clone()
    consensus_score = score_graph_across_splits(
        consensus_graph,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    pruned_graph, prune_rows, pruned_score = prune_consensus_graph(
        consensus_graph,
        accepted,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
        prune_tolerance=prune_tolerance,
        allow_source=include_source_mutations,
    )
    archive.append(ArchiveEntry("pruned_consensus_graph", int(steps), True, pruned_graph.clone(), pruned_score))
    ensembles = ensemble_report(archive, contexts, stability_weight=stability_weight, objective_mode=objective_mode)
    best_ensemble = max(ensembles, key=lambda row: float(row["objective"])) if ensembles else None

    consensus_path = write_graph(
        output_dir / "consensus_graph.json",
        consensus_graph,
        metadata={"benchmark": "competition_event_multisplit_benchmark", "stage": "pre_prune", "score": consensus_score.to_dict()},
    )
    pruned_path = write_graph(
        output_dir / "pruned_consensus_graph.json",
        pruned_graph,
        metadata={"benchmark": "competition_event_multisplit_benchmark", "stage": "post_prune", "score": pruned_score.to_dict()},
    )
    internal_test = internal_test_report(pruned_graph, contexts, evaluator_kwargs=evaluator_kwargs)
    read_paths = [str(path) for path in metadata.get("read_paths", [])]
    reduced_read_paths = [path for path in read_paths if "reduced" in Path(path).name]
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
    return {
        "benchmark": "competition_event_multisplit_benchmark",
        "scope": report_scope(),
        "seed": int(seed),
        "dataset": metadata,
        "dataset_config": config.dataset_config(),
        "benchmark_config": {
            "steps": int(steps),
            "folds": int(folds),
            "max_configurations": int(max_configurations),
            "split_seeds": list(split_seeds),
            "include_source_mutations": bool(include_source_mutations),
            "min_objective_improvement": float(min_objective_improvement),
            "stability_weight": float(stability_weight),
            "objective_mode": objective_mode,
            "prune_tolerance": float(prune_tolerance),
            "objective": objective_description(objective_mode),
        },
        "split_contexts": [context.to_dict() for context in contexts],
        "outer_split": outer_split,
        "split_audits": split_audits,
        "reduced_test_access": {
            "accessed": bool(reduced_read_paths),
            "read_paths": read_paths,
            "reduced_read_paths": reduced_read_paths,
            "checked_paths": ["X_test.reduced.parquet", "y_test_index.reduced.parquet"],
        },
        "source_mutations": {
            "enabled": bool(include_source_mutations),
            "templates": len(source_specs),
            "passed_repair_checks": sum(1 for check in source_checks if check.passed),
            "checks": [check.to_dict() for check in source_checks],
        },
        "seed_graph": seed_score.to_dict(),
        "consensus_graph": {
            **consensus_score.to_dict(),
            "path": str(consensus_path),
            "alternatives": sum(len(node.alternatives) for node in consensus_graph.nodes.values()),
        },
        "pruned_consensus_graph": {
            **pruned_score.to_dict(),
            "path": str(pruned_path),
            "alternatives": sum(len(node.alternatives) for node in pruned_graph.nodes.values()),
        },
        "internal_test": internal_test,
        "ensembles": {
            "rows": ensembles,
            "best": best_ensemble,
            "oof_validation_predictions": True,
        },
        "evolution": {
            "steps": int(steps),
            "accepted_mutations": sum(1 for row in rows if row.get("accepted")),
            "source_backed_candidates": sum(1 for row in rows if row.get("source_backed")),
            "failed_candidates": sum(1 for row in rows if row.get("failed")),
            "validation_selection": "multi_split_grouped",
            "candidates": rows,
            "archive": [entry.to_dict() for entry in archive],
        },
        "pruning": {
            "attempted": len(prune_rows),
            "removed": sum(1 for row in prune_rows if row.get("removed")),
            "rows": prune_rows,
        },
    }


def build_split_contexts(
    inputs: dict[str, object],
    y: np.ndarray,
    *,
    outer_seed: int,
    split_seeds: tuple[int, ...],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[tuple[SplitContext, ...], dict[str, Any]]:
    contexts: list[SplitContext] = []
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
        baseline = evaluate_structural_break_baseline(train_inputs, train_y, validation_inputs, validation_y)
        contexts.append(
            SplitContext(
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


def synthesize_unseen_document(
    engineer: EngineerAgent,
    graph: Graph,
    feedback_result: Any,
    hypotheses: Any,
    *,
    step: int,
    rng: np.random.Generator,
    attempted_signatures: set[tuple[str, str, tuple[str, ...], str]],
) -> MutationDocument:
    attempts = max(1, len(engineer.templates) * 2)
    fallback = engineer.synthesize(graph, feedback_result, hypotheses, step=step, island=None, rng=rng)
    for offset in range(attempts):
        candidate = engineer.synthesize(graph, feedback_result, hypotheses, step=step + offset, island=None, rng=rng)
        add = candidate.add[0] if candidate.add else None
        if add is None or add_signature(add) not in attempted_signatures:
            return candidate
    return fallback


def add_signature(add: Any) -> tuple[str, str, tuple[str, ...], str]:
    return (str(add.target_node), str(add.primitive), tuple(str(parent) for parent in add.parents), str(add.source).strip())


def score_graph_across_splits(
    graph: Graph,
    contexts: tuple[SplitContext, ...],
    *,
    evaluator_kwargs: dict[str, Any],
    stability_weight: float,
    objective_mode: str,
) -> MultiSplitScore:
    split_scores: list[SplitGraphScore] = []
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
            SplitGraphScore(
                seed=context.seed,
                train_auc=holdout.train_auc,
                validation_auc=holdout.validation_auc,
                delta_vs_baseline=float(holdout.validation_auc) - float(context.baseline.validation_auc),
                alpha=holdout.alpha,
                n_features=holdout.n_features,
                config=dict(train_cv.config),
                train_predictions=holdout.train_predictions,
                validation_predictions=holdout.validation_predictions,
            )
        )
    deltas = np.asarray([score.delta_vs_baseline for score in split_scores], dtype=np.float64)
    validation = np.asarray([score.validation_auc for score in split_scores], dtype=np.float64)
    mean_delta = float(np.mean(deltas)) if deltas.size else 0.0
    min_delta = float(np.min(deltas)) if deltas.size else 0.0
    mean_auc = float(np.mean(validation)) if validation.size else 0.5
    min_auc = float(np.min(validation)) if validation.size else 0.5
    return MultiSplitScore(
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


def prune_consensus_graph(
    graph: Graph,
    accepted: list[tuple[str, str, int]],
    contexts: tuple[SplitContext, ...],
    *,
    evaluator_kwargs: dict[str, Any],
    stability_weight: float,
    objective_mode: str,
    prune_tolerance: float,
    allow_source: bool,
) -> tuple[Graph, list[dict[str, Any]], MultiSplitScore]:
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
                MutationDocument(remove=(RemoveSpec(node_name, alternative_id, "Backward multi-split prune candidate."),)),
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


def ensemble_report(
    archive: list[ArchiveEntry],
    contexts: tuple[SplitContext, ...],
    *,
    stability_weight: float,
    objective_mode: str,
) -> list[dict[str, Any]]:
    ranked = sorted(archive, key=lambda entry: float(entry.score.objective), reverse=True)
    rows: list[dict[str, Any]] = []
    for k in (2, 3, 5, 10):
        selected = ranked[: min(k, len(ranked))]
        if len(selected) < 2:
            continue
        rows.append(_ensemble_row(f"top_{len(selected)}_archive", selected, contexts, stability_weight=stability_weight, objective_mode=objective_mode))
    if ranked:
        rows.append(_baseline_blend_row(ranked[0], contexts, stability_weight=stability_weight, objective_mode=objective_mode))
    return rows


def _ensemble_row(
    name: str,
    selected: list[ArchiveEntry],
    contexts: tuple[SplitContext, ...],
    *,
    stability_weight: float,
    objective_mode: str,
) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    for split_index, context in enumerate(contexts):
        predictions = np.mean(np.column_stack([entry.score.splits[split_index].validation_predictions for entry in selected]), axis=1)
        auc = roc_auc_score(context.validation_y, predictions)
        split_rows.append({"seed": context.seed, "validation_auc": float(auc), "delta_vs_baseline": float(auc) - float(context.baseline.validation_auc)})
    deltas = np.asarray([row["delta_vs_baseline"] for row in split_rows], dtype=np.float64)
    aucs = np.asarray([row["validation_auc"] for row in split_rows], dtype=np.float64)
    mean_delta = float(np.mean(deltas)) if deltas.size else 0.0
    min_delta = float(np.min(deltas)) if deltas.size else 0.0
    mean_auc = float(np.mean(aucs)) if aucs.size else 0.5
    min_auc = float(np.min(aucs)) if aucs.size else 0.5
    return {
        "name": name,
        "members": [entry.name for entry in selected],
        "mean_validation_auc": mean_auc,
        "min_validation_auc": min_auc,
        "mean_delta_vs_baseline": mean_delta,
        "min_delta_vs_baseline": min_delta,
        "objective": score_objective(
            mean_validation_auc=mean_auc,
            min_validation_auc=min_auc,
            mean_delta_vs_baseline=mean_delta,
            min_delta_vs_baseline=min_delta,
            stability_weight=stability_weight,
            objective_mode=objective_mode,
        ),
        "splits": split_rows,
    }


def _baseline_blend_row(
    entry: ArchiveEntry,
    contexts: tuple[SplitContext, ...],
    *,
    stability_weight: float,
    objective_mode: str,
) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    for split_index, context in enumerate(contexts):
        predictions = 0.5 * context.baseline.validation_predictions + 0.5 * entry.score.splits[split_index].validation_predictions
        auc = roc_auc_score(context.validation_y, predictions)
        split_rows.append({"seed": context.seed, "validation_auc": float(auc), "delta_vs_baseline": float(auc) - float(context.baseline.validation_auc)})
    deltas = np.asarray([row["delta_vs_baseline"] for row in split_rows], dtype=np.float64)
    aucs = np.asarray([row["validation_auc"] for row in split_rows], dtype=np.float64)
    mean_delta = float(np.mean(deltas)) if deltas.size else 0.0
    min_delta = float(np.min(deltas)) if deltas.size else 0.0
    mean_auc = float(np.mean(aucs)) if aucs.size else 0.5
    min_auc = float(np.min(aucs)) if aucs.size else 0.5
    return {
        "name": "baseline_plus_best_archive_blend",
        "members": ["baseline", entry.name],
        "mean_validation_auc": mean_auc,
        "min_validation_auc": min_auc,
        "mean_delta_vs_baseline": mean_delta,
        "min_delta_vs_baseline": min_delta,
        "objective": score_objective(
            mean_validation_auc=mean_auc,
            min_validation_auc=min_auc,
            mean_delta_vs_baseline=mean_delta,
            min_delta_vs_baseline=min_delta,
            stability_weight=stability_weight,
            objective_mode=objective_mode,
        ),
        "splits": split_rows,
    }


def score_objective(
    *,
    mean_validation_auc: float,
    min_validation_auc: float,
    mean_delta_vs_baseline: float,
    min_delta_vs_baseline: float,
    stability_weight: float,
    objective_mode: str,
) -> float:
    if objective_mode == "auc":
        return float(mean_validation_auc + float(stability_weight) * min_validation_auc)
    if objective_mode == "delta":
        return float(mean_delta_vs_baseline + float(stability_weight) * min_delta_vs_baseline)
    raise ValueError("objective_mode must be 'delta' or 'auc'.")


def objective_description(objective_mode: str) -> str:
    if objective_mode == "auc":
        return "mean_validation_auc + stability_weight * min_validation_auc"
    if objective_mode == "delta":
        return "mean_delta_vs_baseline + stability_weight * min_delta_vs_baseline"
    raise ValueError("objective_mode must be 'delta' or 'auc'.")


def internal_test_report(
    graph: Graph,
    contexts: tuple[SplitContext, ...],
    *,
    evaluator_kwargs: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for context in contexts:
        baseline = evaluate_structural_break_baseline(context.train_inputs, context.train_y, context.test_inputs, context.test_y)
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
    candidates = [
        [
            row["step"],
            "yes" if row.get("accepted") else "no",
            row.get("target_node", ""),
            row.get("primitive", ""),
            "yes" if row.get("source_backed") else "no",
            fmt_float(row.get("mean_validation_auc")),
            fmt_float(row.get("mean_delta_vs_baseline")),
            fmt_float(row.get("min_delta_vs_baseline")),
            fmt_float(row.get("objective")),
        ]
        for row in payload["evolution"]["candidates"]
    ]
    split_rows = [
        [
            row["seed"],
            row["split"]["sizes"]["train"],
            row["split"]["sizes"]["validation"],
            row["split"]["sizes"]["test"],
            fmt_float(row["baseline_validation_auc"]),
        ]
        for row in payload["split_contexts"]
    ]
    ensemble = payload["ensembles"]["best"] or {}
    pruned = payload["pruned_consensus_graph"]
    internal = payload["internal_test"]
    return "\n\n".join(
        [
            "# Competition Multi-Split Structural-Break Benchmark",
            str(payload["scope"]),
            (
                f"Dataset: `{payload['dataset']['name']}` from `{payload['dataset']['data_dir']}`, "
                f"ids=`{payload['dataset']['n_samples']}`, max_samples=`{payload['dataset']['max_samples']}`."
            ),
            f"Config: `{payload['benchmark_config']}`",
            f"Reduced test accessed: `{payload['reduced_test_access']['accessed']}`.",
            (
                f"Pruned consensus graph mean validation AUC `{fmt_float(pruned['mean_validation_auc'])}`, "
                f"mean delta `{fmt_float(pruned['mean_delta_vs_baseline'])}`, "
                f"minimum delta `{fmt_float(pruned['min_delta_vs_baseline'])}`, "
                f"objective `{fmt_float(pruned['objective'])}`."
            ),
            (
                f"Best archive/OOF ensemble `{ensemble.get('name', 'n/a')}` mean validation AUC "
                f"`{fmt_float(ensemble.get('mean_validation_auc'))}`, mean delta "
                f"`{fmt_float(ensemble.get('mean_delta_vs_baseline'))}`, minimum delta "
                f"`{fmt_float(ensemble.get('min_delta_vs_baseline'))}`."
            ),
            (
                f"Internal non-selection test audit mean graph AUC `{fmt_float(internal['mean_graph_auc'])}`, "
                f"mean delta `{fmt_float(internal['mean_delta_vs_baseline'])}`, "
                f"minimum delta `{fmt_float(internal['min_delta_vs_baseline'])}`."
            ),
            markdown_table(["Split Seed", "Train", "Validation", "Test", "Baseline Val AUC"], split_rows),
            markdown_table(["Step", "Accepted", "Node", "Primitive", "Source", "Mean Val AUC", "Mean Delta", "Min Delta", "Objective"], candidates),
        ]
    )


def run(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 211,
    split_seeds: tuple[int, ...] = (211, 223, 307),
    series_length: int = 160,
    max_samples: int | None = None,
    steps: int = 96,
    folds: int = 3,
    max_configurations: int = 64,
    include_source_mutations: bool = True,
    objective_mode: str = "delta",
) -> tuple[Path, Path]:
    payload = build_report(
        output_dir,
        data_dir=data_dir,
        seed=seed,
        split_seeds=split_seeds,
        series_length=series_length,
        max_samples=max_samples,
        steps=steps,
        folds=folds,
        max_configurations=max_configurations,
        include_source_mutations=include_source_mutations,
        objective_mode=objective_mode,
    )
    return write_report(output_dir, "competition_event_multisplit_benchmark", payload, markdown_report(payload))


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a multi-split id-level ADIA-style structural-break benchmark.")
    output_argument(parser)
    seed_argument(parser, default=211)
    parser.add_argument("--split-seeds", type=parse_seeds, default=(211, 223, 307), help="Comma-separated grouped split seeds.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_COMPETITION_DATA_DIR)
    parser.add_argument("--series-length", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-configurations", type=int, default=64)
    parser.add_argument("--objective-mode", choices=("delta", "auc"), default="delta")
    parser.add_argument("--disable-source-mutations", action="store_true")
    args = parser.parse_args(argv)
    print_report_paths(
        run(
            args.output,
            data_dir=args.data_dir,
            seed=args.seed,
            split_seeds=args.split_seeds,
            series_length=args.series_length,
            max_samples=args.max_samples,
            steps=args.steps,
            folds=args.folds,
            max_configurations=args.max_configurations,
            include_source_mutations=not args.disable_source_mutations,
            objective_mode=args.objective_mode,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
