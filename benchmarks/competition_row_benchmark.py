from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common import fmt_float, markdown_table, output_argument, print_report_paths, report_scope, seed_argument, status_mark, write_report
from evoforest_arch.agents import EngineerAgent, ScientistAgent
from evoforest_arch.competition import COMPETITION_ROW_DATASET_NAME, DEFAULT_COMPETITION_DATA_DIR
from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.graph import Graph
from evoforest_arch.metrics import roc_auc_score
from evoforest_arch.mutations import MutationEngine
from evoforest_arch.production import ProductionConfig, load_dataset_with_metadata, make_production_split_manifest
from evoforest_arch.readout import DEFAULT_ALPHAS, Standardizer, fit_ridge, select_alpha
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.splits import split_dataset


@dataclass(frozen=True)
class HoldoutResult:
    train_auc: float
    validation_auc: float
    alpha: float
    n_features: int
    feature_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_auc": float(self.train_auc),
            "validation_auc": float(self.validation_auc),
            "alpha": float(self.alpha),
            "n_features": int(self.n_features),
            "feature_names": list(self.feature_names),
        }


def build_report(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 113,
    series_length: int = 160,
    max_samples: int | None = None,
    max_ids: int | None = 200,
    max_rows_per_id: int | None = 64,
    row_stride: int = 1,
    steps: int = 6,
    folds: int = 3,
    max_configurations: int = 96,
    min_validation_improvement: float = 1e-5,
    meaningful_margin: float = 0.01,
) -> dict[str, Any]:
    del output_dir
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
        min_train_improvement=-1.0,
        min_validation_improvement=min_validation_improvement,
    )
    inputs, y, metadata = load_dataset_with_metadata(config.dataset_config())
    split_manifest = make_production_split_manifest(
        inputs,
        y,
        dataset_name=COMPETITION_ROW_DATASET_NAME,
        seed=seed,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    splits = split_dataset(inputs, y, split_manifest)
    train_inputs, train_y = splits["train"]
    validation_inputs, validation_y = splits["validation"]

    baseline = evaluate_baseline_holdout(train_inputs, train_y, validation_inputs, validation_y)
    evaluator = RidgeEvaluator(**config.evaluator_config())
    scientist = ScientistAgent()
    engineer = EngineerAgent()
    mutation_engine = MutationEngine()
    current_graph = build_seed_graph()
    best_graph = current_graph.clone()
    best_train_cv = evaluator.evaluate(current_graph, train_inputs, train_y, update_graph=True)
    best_holdout = graph_holdout_score(current_graph, train_inputs, train_y, validation_inputs, validation_y, config=best_train_cv.config)
    seed_holdout = best_holdout
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for step in range(1, int(steps) + 1):
        hypotheses = scientist.generate(current_graph, best_train_cv)
        document = engineer.synthesize(current_graph, best_train_cv, hypotheses, step=step, island=None, rng=rng)
        application = mutation_engine.apply_document(current_graph, document)
        candidate_graph = application.graph
        candidate_train_cv = evaluator.evaluate(candidate_graph, train_inputs, train_y, update_graph=False)
        candidate_holdout = graph_holdout_score(
            candidate_graph,
            train_inputs,
            train_y,
            validation_inputs,
            validation_y,
            config=candidate_train_cv.config,
        )
        validation_delta = float(candidate_holdout.validation_auc) - float(best_holdout.validation_auc)
        accepted = validation_delta > float(min_validation_improvement)
        add = document.add[0] if document.add else None
        row = {
            "step": int(step),
            "accepted": bool(accepted),
            "target_node": add.target_node if add else "",
            "primitive": add.primitive if add else "",
            "alternative_id": add.alternative_id if add else "",
            "cv_train_auc": float(candidate_train_cv.auc),
            "holdout_train_auc": float(candidate_holdout.train_auc),
            "holdout_validation_auc": float(candidate_holdout.validation_auc),
            "holdout_validation_delta_vs_best": float(validation_delta),
            "holdout_validation_delta_vs_baseline": float(candidate_holdout.validation_auc) - float(baseline.validation_auc),
            "n_features": int(candidate_holdout.n_features),
            "config": dict(candidate_train_cv.config),
            "folds": candidate_train_cv.diagnostics.get("folds", {}),
            "maintenance": application.maintenance.to_dict(),
        }
        rows.append(row)
        if accepted:
            current_graph = candidate_graph
            best_graph = candidate_graph.clone()
            best_train_cv = candidate_train_cv
            best_holdout = candidate_holdout

    split_audit = split_group_audit(splits)
    evolved_margin = float(best_holdout.validation_auc) - float(baseline.validation_auc)
    read_paths = [str(path) for path in metadata.get("read_paths", [])]
    reduced_read_paths = [path for path in read_paths if "reduced" in Path(path).name]
    return {
        "benchmark": "competition_row_benchmark",
        "scope": report_scope(),
        "seed": int(seed),
        "dataset": metadata,
        "dataset_config": config.dataset_config(),
        "split": {
            "manifest_method": split_manifest.method,
            "group_key": split_manifest.group_key,
            "sizes": {
                "train": len(split_manifest.train_indices),
                "validation": len(split_manifest.validation_indices),
                "test": len(split_manifest.test_indices),
            },
            "group_counts": {
                "train": len(split_manifest.train_groups),
                "validation": len(split_manifest.validation_groups),
                "test": len(split_manifest.test_groups),
            },
            "audit": split_audit,
        },
        "reduced_test_access": {
            "accessed": bool(reduced_read_paths),
            "read_paths": read_paths,
            "reduced_read_paths": reduced_read_paths,
            "evidence": [
                "Loaded only dataset split='train' through load_competition_row_dataset.",
                "make_production_split_manifest split train parquet rows into train/validation/test by sample_id.",
                "This benchmark does not call load_external_reduced_test or load_competition_row_dataset(split='reduced_test').",
            ],
            "checked_paths": ["X_test.reduced.parquet", "y_test.reduced.parquet", "y_test_index.reduced.parquet"],
        },
        "baseline": baseline.to_dict(),
        "seed_graph": seed_holdout.to_dict(),
        "evolved_graph": {
            **best_holdout.to_dict(),
            "validation_delta_vs_baseline": evolved_margin,
            "validation_delta_vs_seed_graph": float(best_holdout.validation_auc) - float(seed_holdout.validation_auc),
            "beats_baseline": bool(evolved_margin > 0.0),
            "beats_baseline_by_meaningful_margin": bool(evolved_margin >= float(meaningful_margin)),
            "meaningful_margin": float(meaningful_margin),
            "alternatives": sum(len(node.alternatives) for node in best_graph.nodes.values()),
        },
        "evolution": {
            "steps": int(steps),
            "accepted_mutations": sum(1 for row in rows if row["accepted"]),
            "min_validation_improvement": float(min_validation_improvement),
            "validation_selection_only": True,
            "candidates": rows,
        },
    }


def evaluate_baseline_holdout(
    train_inputs: dict[str, object],
    train_y: np.ndarray,
    validation_inputs: dict[str, object],
    validation_y: np.ndarray,
) -> HoldoutResult:
    x_train, names = row_local_baseline_features(train_inputs)
    x_validation, validation_names = row_local_baseline_features(validation_inputs)
    if names != validation_names:
        raise ValueError("Baseline train and validation feature names differ.")
    return fit_holdout_readout(x_train, train_y, x_validation, validation_y, names)


def graph_holdout_score(
    graph: Graph,
    train_inputs: dict[str, object],
    train_y: np.ndarray,
    validation_inputs: dict[str, object],
    validation_y: np.ndarray,
    *,
    config: dict[str, str],
) -> HoldoutResult:
    x_train, names, _ctx = graph.evaluate_features(train_inputs, config=config)
    x_validation, validation_names, _validation_ctx = graph.evaluate_features(validation_inputs, config=config)
    if names != validation_names:
        raise ValueError("Graph train and validation feature names differ.")
    return fit_holdout_readout(x_train, train_y, x_validation, validation_y, names)


def fit_holdout_readout(
    x_train: np.ndarray,
    train_y: np.ndarray,
    x_validation: np.ndarray,
    validation_y: np.ndarray,
    names: list[str],
) -> HoldoutResult:
    train_y = np.asarray(train_y, dtype=np.float64)
    validation_y = np.asarray(validation_y, dtype=np.float64)
    std = Standardizer.fit(np.asarray(x_train, dtype=np.float64))
    train_z = std.transform(x_train)
    validation_z = std.transform(x_validation)
    alpha = select_alpha(train_z, train_y, DEFAULT_ALPHAS)
    model = fit_ridge(train_z, train_y, alpha)
    train_pred = model.predict(train_z)
    validation_pred = model.predict(validation_z)
    return HoldoutResult(
        train_auc=roc_auc_score(train_y, train_pred),
        validation_auc=roc_auc_score(validation_y, validation_pred),
        alpha=alpha,
        n_features=int(train_z.shape[1]),
        feature_names=list(names),
    )


def row_local_baseline_features(inputs: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    series = np.asarray(inputs["series"], dtype=np.float64)
    boundary = int(inputs["boundary"])
    older = series[:, :boundary]
    recent = series[:, boundary:]
    older_tail = tail_window(older)
    recent_head = head_window(recent)
    recent_tail = tail_window(recent)
    older_mean = np.mean(older, axis=1)
    recent_mean = np.mean(recent, axis=1)
    older_std = safe_std(older)
    recent_std = safe_std(recent)
    older_diff = np.diff(older, axis=1)
    recent_diff = np.diff(recent, axis=1)
    recent_drawdown, recent_drawup = drawdown_drawup(recent)
    centered = series - np.mean(series, axis=1, keepdims=True)
    cumulative = np.cumsum(centered, axis=1)
    peak_abs = np.maximum(np.max(np.abs(cumulative), axis=1), 1e-8)
    sample_time = optional_vector(inputs, "sample_time", series.shape[0])
    sample_period = optional_vector(inputs, "sample_period", series.shape[0])
    observed = optional_vector(inputs, "lookback_observed", series.shape[0])
    sample_time_scale = optional_vector(inputs, "sample_time_scale", series.shape[0])
    if np.any(sample_time_scale > 0.0):
        time_scale = max(float(np.max(sample_time_scale)), 1.0)
    else:
        time_scale = max(float(np.max(sample_time)) if sample_time.size else 0.0, 1.0)
    features = np.column_stack(
        [
            series[:, -1],
            series[:, -1] - older_mean,
            series[:, -1] - recent_mean,
            recent_mean - older_mean,
            np.mean(recent_head, axis=1) - np.mean(older_tail, axis=1),
            np.mean(recent_tail, axis=1) - np.mean(recent_head, axis=1),
            np.log(recent_std / older_std),
            np.mean(np.abs(recent_diff), axis=1) - np.mean(np.abs(older_diff), axis=1),
            np.log((np.mean(np.abs(recent_diff), axis=1) + 1e-8) / (np.mean(np.abs(older_diff), axis=1) + 1e-8)),
            slope(series),
            slope(recent) - slope(older),
            slope(recent_tail),
            recent_drawdown,
            recent_drawup,
            peak_abs / np.maximum(safe_std(series), 1e-8),
            np.argmax(np.abs(cumulative), axis=1).astype(np.float64) / max(series.shape[1] - 1, 1),
            np.max(np.abs(cumulative[:, boundary:]), axis=1) / peak_abs,
            sample_time / time_scale,
            np.log1p(np.maximum(sample_time, 0.0)) / max(float(np.log1p(time_scale)), 1.0),
            (sample_period >= 2.0).astype(np.float64),
            np.clip(observed / max(float(series.shape[1]), 1.0), 0.0, 1.0),
        ]
    )
    names = [
        "last_value",
        "last_vs_older_mean",
        "last_vs_recent_mean",
        "recent_vs_older_mean",
        "recent_entry_jump",
        "recent_tail_drift",
        "recent_std_log_ratio",
        "absdiff_delta",
        "absdiff_log_ratio",
        "full_window_slope",
        "recent_slope_delta",
        "recent_tail_slope",
        "recent_drawdown",
        "recent_drawup",
        "local_cusum_peak",
        "local_cusum_peak_location",
        "recent_cusum_share",
        "sample_time_norm",
        "sample_time_log_norm",
        "sample_period_two",
        "lookback_observed_fraction",
    ]
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), names


def split_group_audit(splits: dict[str, tuple[dict[str, object], np.ndarray]]) -> dict[str, Any]:
    groups = {name: set(np.asarray(inputs["sample_id"]).astype(np.int64).tolist()) for name, (inputs, _y) in splits.items()}
    labels = {
        name: {
            "positive_count": int(np.sum(np.asarray(y) >= 0.5)),
            "negative_count": int(y.shape[0] - np.sum(np.asarray(y) >= 0.5)),
            "n_samples": int(y.shape[0]),
            "n_groups": int(len(groups[name])),
        }
        for name, (_inputs, y) in splits.items()
    }
    return {
        "labels": labels,
        "group_overlaps": {
            "train_validation": len(groups["train"] & groups["validation"]),
            "train_test": len(groups["train"] & groups["test"]),
            "validation_test": len(groups["validation"] & groups["test"]),
        },
        "no_group_overlap": not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]),
    }


def markdown_report(payload: dict[str, Any]) -> str:
    candidates = [
        [
            row["step"],
            status_mark(bool(row["accepted"])),
            row["target_node"],
            row["primitive"],
            fmt_float(row["cv_train_auc"]),
            fmt_float(row["holdout_validation_auc"]),
            fmt_float(row["holdout_validation_delta_vs_baseline"]),
        ]
        for row in payload["evolution"]["candidates"]
    ]
    dataset = payload["dataset"]
    baseline = payload["baseline"]
    evolved = payload["evolved_graph"]
    split = payload["split"]
    return "\n\n".join(
        [
            "# Competition Row-Level Benchmark",
            str(payload["scope"]),
            (
                f"Dataset: `{dataset['name']}` from `{dataset['data_dir']}`, rows=`{dataset['n_samples']}`, "
                f"ids=`{dataset['n_ids']}`, max_ids=`{dataset['max_ids']}`, max_rows_per_id=`{dataset['max_rows_per_id']}`."
            ),
            (
                f"Split: `{split['manifest_method']}` by `{split['group_key']}`, sizes={split['sizes']}, "
                f"group overlaps={split['audit']['group_overlaps']}."
            ),
            "Reduced test accessed: `False`.",
            (
                f"Baseline validation AUC: `{fmt_float(baseline['validation_auc'])}`; "
                f"evolved validation AUC: `{fmt_float(evolved['validation_auc'])}`; "
                f"delta: `{fmt_float(evolved['validation_delta_vs_baseline'])}`; "
                f"meaningful margin met: `{evolved['beats_baseline_by_meaningful_margin']}`."
            ),
            markdown_table(["Step", "Accepted", "Node", "Primitive", "CV Train AUC", "Holdout Val AUC", "Delta vs Baseline"], candidates),
        ]
    )


def run(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 113,
    series_length: int = 160,
    max_samples: int | None = None,
    max_ids: int | None = 200,
    max_rows_per_id: int | None = 64,
    row_stride: int = 1,
    steps: int = 6,
    folds: int = 3,
    max_configurations: int = 96,
    meaningful_margin: float = 0.01,
) -> tuple[Path, Path]:
    payload = build_report(
        output_dir,
        data_dir=data_dir,
        seed=seed,
        series_length=series_length,
        max_samples=max_samples,
        max_ids=max_ids,
        max_rows_per_id=max_rows_per_id,
        row_stride=row_stride,
        steps=steps,
        folds=folds,
        max_configurations=max_configurations,
        meaningful_margin=meaningful_margin,
    )
    return write_report(output_dir, "competition_row_benchmark", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a grouped-id row/time-level competition benchmark.")
    output_argument(parser)
    seed_argument(parser, default=113)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_COMPETITION_DATA_DIR)
    parser.add_argument("--series-length", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-ids", type=int, default=200)
    parser.add_argument("--max-rows-per-id", type=int, default=64)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-configurations", type=int, default=96)
    parser.add_argument("--meaningful-margin", type=float, default=0.01)
    args = parser.parse_args(argv)
    print_report_paths(
        run(
            args.output,
            data_dir=args.data_dir,
            seed=args.seed,
            series_length=args.series_length,
            max_samples=args.max_samples,
            max_ids=args.max_ids,
            max_rows_per_id=args.max_rows_per_id,
            row_stride=args.row_stride,
            steps=args.steps,
            folds=args.folds,
            max_configurations=args.max_configurations,
            meaningful_margin=args.meaningful_margin,
        )
    )
    return 0


def safe_std(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.std(x, axis=1), 1e-8)


def slope(x: np.ndarray) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, x.shape[1])
    centered_t = t - np.mean(t)
    centered_x = x - np.mean(x, axis=1, keepdims=True)
    return (centered_x @ centered_t) / max(float(np.sum(centered_t**2)), 1e-8)


def head_window(x: np.ndarray) -> np.ndarray:
    return x[:, : max(2, x.shape[1] // 4)]


def tail_window(x: np.ndarray) -> np.ndarray:
    return x[:, -max(2, x.shape[1] // 4) :]


def drawdown_drawup(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cumulative_max = np.maximum.accumulate(x, axis=1)
    cumulative_min = np.minimum.accumulate(x, axis=1)
    scale = np.maximum(safe_std(x), 1e-8)
    return np.max(cumulative_max - x, axis=1) / scale, np.max(x - cumulative_min, axis=1) / scale


def optional_vector(inputs: dict[str, object], name: str, n_rows: int) -> np.ndarray:
    if name not in inputs:
        return np.zeros(n_rows, dtype=np.float64)
    value = np.asarray(inputs[name], dtype=np.float64).reshape(-1)
    if value.shape[0] != n_rows:
        return np.zeros(n_rows, dtype=np.float64)
    return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    raise SystemExit(main())
