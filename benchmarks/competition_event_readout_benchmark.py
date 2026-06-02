from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common import fmt_float, graph_summary, markdown_table, output_argument, print_report_paths, report_scope, seed_argument, write_report
from benchmarks.competition_event_benchmark import evaluate_structural_break_baseline, graph_holdout_score
from benchmarks.competition_event_multisplit_benchmark import build_split_contexts
from evoforest_arch.competition import COMPETITION_DATASET_NAME, DEFAULT_COMPETITION_DATA_DIR
from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import graph_from_path
from evoforest_arch.metrics import roc_auc_score
from evoforest_arch.production import ProductionConfig, load_dataset_with_metadata
from evoforest_arch.rank_readout import fit_rank_feature_expansion, select_rank_ensemble
from evoforest_arch.readout import Standardizer
from evoforest_arch.seed import build_seed_graph


def build_report(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    graph_path: Path | None = None,
    seed: int = 211,
    split_seeds: tuple[int, ...] = (211, 223, 307),
    series_length: int = 160,
    max_samples: int | None = None,
    folds: int = 3,
    max_configurations: int = 64,
    max_interaction_base: int = 12,
) -> dict[str, Any]:
    del output_dir
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
        allow_source_mutations=True,
    )
    inputs, y, metadata = load_dataset_with_metadata(config.dataset_config())
    contexts, outer_split = build_split_contexts(inputs, y, outer_seed=seed, split_seeds=split_seeds)
    graph = graph_from_path(graph_path, allow_source=True) if graph_path else build_seed_graph()
    evaluator_kwargs = {
        "n_splits": int(folds),
        "max_configurations": int(max_configurations),
        "irls_steps": 0,
        "group_key": "sample_id",
    }
    rows = [
        evaluate_split_readouts(
            graph,
            context,
            evaluator_kwargs=evaluator_kwargs,
            max_interaction_base=max_interaction_base,
        )
        for context in contexts
    ]
    readout_summary = summarize_readouts(rows)
    read_paths = [str(path) for path in metadata.get("read_paths", [])]
    reduced_read_paths = [path for path in read_paths if "reduced" in Path(path).name]
    return {
        "benchmark": "competition_event_readout_benchmark",
        "scope": report_scope(),
        "seed": int(seed),
        "dataset": metadata,
        "dataset_config": config.dataset_config(),
        "benchmark_config": {
            "graph_path": str(graph_path) if graph_path else "",
            "split_seeds": list(split_seeds),
            "series_length": int(series_length),
            "max_samples": max_samples,
            "folds": int(folds),
            "max_configurations": int(max_configurations),
            "max_interaction_base": int(max_interaction_base),
            "readouts": ["ridge", "rank_interaction", "ridge_rank_blend"],
        },
        "outer_split": outer_split,
        "graph": graph_summary(graph),
        "reduced_test_access": {
            "accessed": bool(reduced_read_paths),
            "read_paths": read_paths,
            "reduced_read_paths": reduced_read_paths,
            "checked_paths": ["X_test.reduced.parquet", "y_test_index.reduced.parquet"],
        },
        "summary": readout_summary,
        "splits": rows,
    }


def evaluate_split_readouts(
    graph: Graph,
    context: Any,
    *,
    evaluator_kwargs: dict[str, Any],
    max_interaction_base: int,
) -> dict[str, Any]:
    evaluator = RidgeEvaluator(seed=context.seed, **evaluator_kwargs)
    train_cv = evaluator.evaluate(graph, context.train_inputs, context.train_y, update_graph=False)
    ridge_validation = graph_holdout_score(
        graph,
        context.train_inputs,
        context.train_y,
        context.validation_inputs,
        context.validation_y,
        config=train_cv.config,
    )
    ridge_test = graph_holdout_score(
        graph,
        context.train_inputs,
        context.train_y,
        context.test_inputs,
        context.test_y,
        config=train_cv.config,
    )
    x_train, names, _ctx = graph.evaluate_features(context.train_inputs, config=train_cv.config)
    x_validation, validation_names, _validation_ctx = graph.evaluate_features(context.validation_inputs, config=train_cv.config)
    x_test, test_names, _test_ctx = graph.evaluate_features(context.test_inputs, config=train_cv.config)
    if names != validation_names or names != test_names:
        raise ValueError("Graph feature names differ across train/validation/test.")
    expansion = fit_rank_feature_expansion(x_train, context.train_y, names, max_interaction_base=max_interaction_base)
    x_train_rank = expansion.transform(x_train)
    x_validation_rank = expansion.transform(x_validation)
    x_test_rank = expansion.transform(x_test)
    rank_selection = select_rank_ensemble(
        x_train_rank,
        context.train_y,
        np.asarray(context.train_inputs["sample_id"]),
        n_splits=int(evaluator_kwargs["n_splits"]),
        seed=context.seed,
    )
    rank_validation_pred = rank_selection.model.predict(x_validation_rank)
    rank_test_pred = rank_selection.model.predict(x_test_rank)
    baseline_test = evaluate_structural_break_baseline(context.train_inputs, context.train_y, context.test_inputs, context.test_y)
    blend = fit_blend(
        context.train_y,
        train_cv.predictions,
        rank_selection.oof_predictions,
        ridge_validation.validation_predictions,
        rank_validation_pred,
        ridge_test.validation_predictions,
        rank_test_pred,
    )
    return {
        "seed": int(context.seed),
        "_validation_labels": np.asarray(context.validation_y, dtype=np.float64),
        "_test_labels": np.asarray(context.test_y, dtype=np.float64),
        "baseline": {
            "validation_auc": float(context.baseline.validation_auc),
            "test_auc": float(baseline_test.validation_auc),
        },
        "graph_config": dict(train_cv.config),
        "n_graph_features": int(len(names)),
        "rank_expansion": expansion.to_dict(),
        "ridge": {
            "train_oof_auc": float(train_cv.auc),
            "validation_auc": float(ridge_validation.validation_auc),
            "validation_delta_vs_baseline": float(ridge_validation.validation_auc) - float(context.baseline.validation_auc),
            "test_auc": float(ridge_test.validation_auc),
            "test_delta_vs_baseline": float(ridge_test.validation_auc) - float(baseline_test.validation_auc),
            "alpha": float(ridge_validation.alpha),
        },
        "rank_interaction": {
            "train_oof_auc": float(rank_selection.oof_auc),
            "validation_auc": float(roc_auc_score(context.validation_y, rank_validation_pred)),
            "validation_delta_vs_baseline": float(roc_auc_score(context.validation_y, rank_validation_pred)) - float(context.baseline.validation_auc),
            "test_auc": float(roc_auc_score(context.test_y, rank_test_pred)),
            "test_delta_vs_baseline": float(roc_auc_score(context.test_y, rank_test_pred)) - float(baseline_test.validation_auc),
            "selection": rank_selection.to_dict(),
        },
        "ridge_rank_blend": blend,
    }


def fit_blend(
    train_y: np.ndarray,
    ridge_oof: np.ndarray,
    rank_oof: np.ndarray,
    ridge_validation: np.ndarray,
    rank_validation: np.ndarray,
    ridge_test: np.ndarray,
    rank_test: np.ndarray,
) -> dict[str, Any]:
    ridge_std = Standardizer.fit(np.asarray(ridge_oof, dtype=np.float64).reshape(-1, 1))
    rank_std = Standardizer.fit(np.asarray(rank_oof, dtype=np.float64).reshape(-1, 1))
    ridge_oof_z = ridge_std.transform(np.asarray(ridge_oof, dtype=np.float64).reshape(-1, 1))[:, 0]
    rank_oof_z = rank_std.transform(np.asarray(rank_oof, dtype=np.float64).reshape(-1, 1))[:, 0]
    best_weight = 0.0
    best_auc = -1.0
    rows: list[dict[str, float]] = []
    for weight in np.linspace(0.0, 1.0, 9):
        pred = float(weight) * ridge_oof_z + (1.0 - float(weight)) * rank_oof_z
        auc = roc_auc_score(train_y, pred)
        rows.append({"ridge_weight": float(weight), "train_oof_auc": float(auc)})
        if auc > best_auc:
            best_auc = float(auc)
            best_weight = float(weight)

    def apply(ridge_pred: np.ndarray, rank_pred: np.ndarray) -> np.ndarray:
        ridge_z = ridge_std.transform(np.asarray(ridge_pred, dtype=np.float64).reshape(-1, 1))[:, 0]
        rank_z = rank_std.transform(np.asarray(rank_pred, dtype=np.float64).reshape(-1, 1))[:, 0]
        return best_weight * ridge_z + (1.0 - best_weight) * rank_z

    return {
        "ridge_weight": float(best_weight),
        "train_oof_auc": float(best_auc),
        "validation_predictions": apply(ridge_validation, rank_validation),
        "test_predictions": apply(ridge_test, rank_test),
        "weight_candidates": rows,
    }


def summarize_readouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    readout_names = ["ridge", "rank_interaction", "ridge_rank_blend"]
    summary: dict[str, Any] = {}
    for readout in readout_names:
        validation_aucs: list[float] = []
        validation_deltas: list[float] = []
        test_aucs: list[float] = []
        test_deltas: list[float] = []
        for row in rows:
            if readout == "ridge_rank_blend":
                validation_auc = roc_auc_score_for_row(row, "validation")
                test_auc = roc_auc_score_for_row(row, "test")
                validation_delta = validation_auc - float(row["baseline"]["validation_auc"])
                test_delta = test_auc - float(row["baseline"]["test_auc"])
                row[readout]["validation_auc"] = validation_auc
                row[readout]["validation_delta_vs_baseline"] = validation_delta
                row[readout]["test_auc"] = test_auc
                row[readout]["test_delta_vs_baseline"] = test_delta
            else:
                validation_auc = float(row[readout]["validation_auc"])
                validation_delta = float(row[readout]["validation_delta_vs_baseline"])
                test_auc = float(row[readout]["test_auc"])
                test_delta = float(row[readout]["test_delta_vs_baseline"])
            validation_aucs.append(validation_auc)
            validation_deltas.append(validation_delta)
            test_aucs.append(test_auc)
            test_deltas.append(test_delta)
        summary[readout] = {
            "mean_validation_auc": float(np.mean(validation_aucs)) if validation_aucs else 0.5,
            "min_validation_auc": float(np.min(validation_aucs)) if validation_aucs else 0.5,
            "mean_validation_delta_vs_baseline": float(np.mean(validation_deltas)) if validation_deltas else 0.0,
            "min_validation_delta_vs_baseline": float(np.min(validation_deltas)) if validation_deltas else 0.0,
            "mean_test_auc": float(np.mean(test_aucs)) if test_aucs else 0.5,
            "min_test_auc": float(np.min(test_aucs)) if test_aucs else 0.5,
            "mean_test_delta_vs_baseline": float(np.mean(test_deltas)) if test_deltas else 0.0,
            "min_test_delta_vs_baseline": float(np.min(test_deltas)) if test_deltas else 0.0,
        }
    best = max(summary, key=lambda name: float(summary[name]["mean_validation_auc"])) if summary else ""
    return {"best_by_validation_auc": best, "readouts": summary}


def roc_auc_score_for_row(row: dict[str, Any], split: str) -> float:
    key = f"{split}_predictions"
    labels_key = f"_{split}_labels"
    if labels_key not in row:
        raise KeyError(f"Missing private labels {labels_key!r} for blend scoring.")
    return float(roc_auc_score(row[labels_key], row["ridge_rank_blend"][key]))


def markdown_report(payload: dict[str, Any]) -> str:
    summary_rows = []
    for name, row in payload["summary"]["readouts"].items():
        summary_rows.append(
            [
                name,
                fmt_float(row["mean_validation_auc"]),
                fmt_float(row["mean_validation_delta_vs_baseline"]),
                fmt_float(row["mean_test_auc"]),
                fmt_float(row["mean_test_delta_vs_baseline"]),
            ]
        )
    split_rows = []
    for row in payload["splits"]:
        split_rows.append(
            [
                row["seed"],
                fmt_float(row["baseline"]["validation_auc"]),
                fmt_float(row["ridge"]["validation_auc"]),
                fmt_float(row["rank_interaction"]["validation_auc"]),
                fmt_float(row["ridge_rank_blend"]["validation_auc"]),
                fmt_float(row["baseline"]["test_auc"]),
                fmt_float(row["ridge_rank_blend"]["test_auc"]),
            ]
        )
    return "\n\n".join(
        [
            "# Competition Event Readout Benchmark",
            str(payload["scope"]),
            (
                f"Dataset: `{payload['dataset']['name']}` from `{payload['dataset']['data_dir']}`, "
                f"ids=`{payload['dataset']['n_samples']}`, max_samples=`{payload['dataset']['max_samples']}`."
            ),
            f"Graph path: `{payload['benchmark_config']['graph_path'] or 'seed_graph'}`.",
            f"Reduced test accessed: `{payload['reduced_test_access']['accessed']}`.",
            f"Best readout by validation AUC: `{payload['summary']['best_by_validation_auc']}`.",
            markdown_table(["Readout", "Mean Val AUC", "Mean Val Delta", "Mean Internal Test AUC", "Mean Test Delta"], summary_rows),
            markdown_table(["Seed", "Baseline Val", "Ridge Val", "Rank Val", "Blend Val", "Baseline Test", "Blend Test"], split_rows),
        ]
    )


def run(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    graph_path: Path | None = None,
    seed: int = 211,
    split_seeds: tuple[int, ...] = (211, 223, 307),
    series_length: int = 160,
    max_samples: int | None = None,
    folds: int = 3,
    max_configurations: int = 64,
    max_interaction_base: int = 12,
) -> tuple[Path, Path]:
    payload = build_report(
        output_dir,
        data_dir=data_dir,
        graph_path=graph_path,
        seed=seed,
        split_seeds=split_seeds,
        series_length=series_length,
        max_samples=max_samples,
        folds=folds,
        max_configurations=max_configurations,
        max_interaction_base=max_interaction_base,
    )
    for row in payload["splits"]:
        row.pop("_validation_labels", None)
        row.pop("_test_labels", None)
        row.get("ridge_rank_blend", {}).pop("validation_predictions", None)
        row.get("ridge_rank_blend", {}).pop("test_predictions", None)
    return write_report(output_dir, "competition_event_readout_benchmark", payload, markdown_report(payload))


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate stronger NumPy-only readouts on an EvoForest event graph.")
    output_argument(parser)
    seed_argument(parser, default=211)
    parser.add_argument("--graph", type=Path, default=None, help="Optional serialized graph JSON to evaluate.")
    parser.add_argument("--split-seeds", type=parse_seeds, default=(211, 223, 307))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_COMPETITION_DATA_DIR)
    parser.add_argument("--series-length", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-configurations", type=int, default=64)
    parser.add_argument("--max-interaction-base", type=int, default=12)
    args = parser.parse_args(argv)
    print_report_paths(
        run(
            args.output,
            data_dir=args.data_dir,
            graph_path=args.graph,
            seed=args.seed,
            split_seeds=args.split_seeds,
            series_length=args.series_length,
            max_samples=args.max_samples,
            folds=args.folds,
            max_configurations=args.max_configurations,
            max_interaction_base=args.max_interaction_base,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
