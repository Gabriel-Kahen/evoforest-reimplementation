from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common import fmt_float, markdown_table, output_argument, print_report_paths, report_scope, seed_argument, status_mark, write_report
from evoforest_arch.agents import EngineerAgent, ScientistAgent
from evoforest_arch.competition import COMPETITION_DATASET_NAME, DEFAULT_COMPETITION_DATA_DIR
from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.graph import Graph
from evoforest_arch.metrics import roc_auc_score
from evoforest_arch.mutations import MutationEngine, built_in_mutations
from evoforest_arch.production import ProductionConfig, load_dataset_with_metadata
from evoforest_arch.readout import DEFAULT_ALPHAS, Standardizer, fit_ridge, select_alpha
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.source_mutations import structural_break_source_mutations, validate_source_mutations
from evoforest_arch.splits import make_grouped_split_manifest, split_dataset


@dataclass(frozen=True)
class PredictionHoldoutResult:
    train_auc: float
    validation_auc: float
    alpha: float
    n_features: int
    feature_names: list[str]
    train_predictions: np.ndarray
    validation_predictions: np.ndarray

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
    seed: int = 211,
    series_length: int = 160,
    max_samples: int | None = None,
    steps: int = 24,
    folds: int = 3,
    max_configurations: int = 96,
    min_validation_improvement: float = 1e-5,
    meaningful_margin: float = 0.01,
    include_source_mutations: bool = True,
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
        min_train_improvement=-1.0,
        min_validation_improvement=min_validation_improvement,
        allow_source_mutations=include_source_mutations,
    )
    inputs, y, metadata = load_dataset_with_metadata(config.dataset_config())
    split_manifest = make_grouped_split_manifest(
        inputs,
        y,
        groups=np.asarray(inputs["sample_id"]),
        group_key="sample_id",
        seed=seed,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    splits = split_dataset(inputs, y, split_manifest)
    train_inputs, train_y = splits["train"]
    validation_inputs, validation_y = splits["validation"]

    baseline = evaluate_structural_break_baseline(train_inputs, train_y, validation_inputs, validation_y)
    evaluator = RidgeEvaluator(
        n_splits=folds,
        seed=seed,
        max_configurations=max_configurations,
        irls_steps=0,
        group_key="sample_id",
    )
    scientist = ScientistAgent()
    source_specs = structural_break_source_mutations() if include_source_mutations else ()
    source_checks = validate_source_mutations(build_seed_graph(), source_specs, train_inputs) if source_specs else []
    passed_source_specs = tuple(spec for spec, check in zip(source_specs, source_checks) if check.passed)
    templates = passed_source_specs + tuple(built_in_mutations())
    engineer = EngineerAgent(templates=templates)
    mutation_engine = MutationEngine(allow_source=include_source_mutations)

    current_graph = build_seed_graph()
    best_graph = current_graph.clone()
    best_train_cv = evaluator.evaluate(current_graph, train_inputs, train_y, update_graph=True)
    seed_holdout = graph_holdout_score(current_graph, train_inputs, train_y, validation_inputs, validation_y, config=best_train_cv.config)
    best_holdout = seed_holdout
    rows: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = [
        {
            "name": "seed_graph",
            "step": 0,
            "accepted": True,
            "validation_auc": float(seed_holdout.validation_auc),
            "train_auc": float(seed_holdout.train_auc),
            "train_predictions": seed_holdout.train_predictions,
            "validation_predictions": seed_holdout.validation_predictions,
        }
    ]
    rng = np.random.default_rng(seed)
    for step in range(1, int(steps) + 1):
        hypotheses = scientist.generate(current_graph, best_train_cv)
        document = engineer.synthesize(current_graph, best_train_cv, hypotheses, step=step, island=None, rng=rng)
        add = document.add[0] if document.add else None
        try:
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
            row = {
                "step": int(step),
                "accepted": bool(accepted),
                "failed": False,
                "target_node": add.target_node if add else "",
                "primitive": add.primitive if add else "",
                "alternative_id": add.alternative_id if add else "",
                "source_backed": bool(add and add.source),
                "cv_train_auc": float(candidate_train_cv.auc),
                "holdout_train_auc": float(candidate_holdout.train_auc),
                "holdout_validation_auc": float(candidate_holdout.validation_auc),
                "holdout_validation_delta_vs_best": float(validation_delta),
                "holdout_validation_delta_vs_baseline": float(candidate_holdout.validation_auc) - float(baseline.validation_auc),
                "n_features": int(candidate_holdout.n_features),
                "config": dict(candidate_train_cv.config),
                "fold_group_overlap_count": int(candidate_train_cv.diagnostics.get("folds", {}).get("fold_group_overlap_count", 0)),
                "maintenance": application.maintenance.to_dict(),
            }
            archive.append(
                {
                    "name": add.alternative_id if add else f"step_{step}",
                    "step": int(step),
                    "accepted": bool(accepted),
                    "validation_auc": float(candidate_holdout.validation_auc),
                    "train_auc": float(candidate_holdout.train_auc),
                    "train_predictions": candidate_holdout.train_predictions,
                    "validation_predictions": candidate_holdout.validation_predictions,
                }
            )
            if accepted:
                current_graph = candidate_graph
                best_graph = candidate_graph.clone()
                best_train_cv = candidate_train_cv
                best_holdout = candidate_holdout
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

    ensembles = ensemble_report(archive, train_y, validation_y, baseline)
    split_audit = split_group_audit(splits)
    read_paths = [str(path) for path in metadata.get("read_paths", [])]
    reduced_read_paths = [path for path in read_paths if "reduced" in Path(path).name]
    evolved_margin = float(best_holdout.validation_auc) - float(baseline.validation_auc)
    best_ensemble = max(ensembles, key=lambda row: float(row["validation_auc"])) if ensembles else None
    return {
        "benchmark": "competition_event_benchmark",
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
            "checked_paths": ["X_test.reduced.parquet", "y_test_index.reduced.parquet"],
        },
        "source_mutations": {
            "enabled": bool(include_source_mutations),
            "templates": len(source_specs),
            "passed_repair_checks": sum(1 for check in source_checks if check.passed),
            "checks": [check.to_dict() for check in source_checks],
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
        "ensembles": {
            "rows": ensembles,
            "best": best_ensemble,
            "beats_baseline_by_meaningful_margin": bool(best_ensemble and float(best_ensemble["validation_auc"]) - float(baseline.validation_auc) >= float(meaningful_margin)),
        },
        "evolution": {
            "steps": int(steps),
            "accepted_mutations": sum(1 for row in rows if row.get("accepted")),
            "source_backed_candidates": sum(1 for row in rows if row.get("source_backed")),
            "failed_candidates": sum(1 for row in rows if row.get("failed")),
            "min_validation_improvement": float(min_validation_improvement),
            "validation_selection_only": True,
            "candidates": rows,
        },
    }


def evaluate_structural_break_baseline(
    train_inputs: dict[str, object],
    train_y: np.ndarray,
    validation_inputs: dict[str, object],
    validation_y: np.ndarray,
) -> PredictionHoldoutResult:
    x_train, names = structural_break_baseline_features(train_inputs)
    x_validation, validation_names = structural_break_baseline_features(validation_inputs)
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
) -> PredictionHoldoutResult:
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
) -> PredictionHoldoutResult:
    train_y = np.asarray(train_y, dtype=np.float64)
    validation_y = np.asarray(validation_y, dtype=np.float64)
    std = Standardizer.fit(np.asarray(x_train, dtype=np.float64))
    train_z = std.transform(x_train)
    validation_z = std.transform(x_validation)
    alpha = select_alpha(train_z, train_y, DEFAULT_ALPHAS)
    model = fit_ridge(train_z, train_y, alpha)
    train_pred = model.predict(train_z)
    validation_pred = model.predict(validation_z)
    return PredictionHoldoutResult(
        train_auc=roc_auc_score(train_y, train_pred),
        validation_auc=roc_auc_score(validation_y, validation_pred),
        alpha=alpha,
        n_features=int(train_z.shape[1]),
        feature_names=list(names),
        train_predictions=train_pred,
        validation_predictions=validation_pred,
    )


def structural_break_baseline_features(inputs: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    series = np.asarray(inputs["series"], dtype=np.float64)
    boundary = int(inputs["boundary"])
    pre = series[:, :boundary]
    post = series[:, boundary:]
    pre_tail = tail_window(pre)
    post_head = head_window(post)
    post_tail = tail_window(post)
    pre_d = np.diff(pre, axis=1)
    post_d = np.diff(post, axis=1)
    pre_mean = np.mean(pre, axis=1)
    post_mean = np.mean(post, axis=1)
    pre_std = safe_std(pre)
    post_std = safe_std(post)
    pre_iqr = np.quantile(pre, 0.75, axis=1) - np.quantile(pre, 0.25, axis=1)
    post_iqr = np.quantile(post, 0.75, axis=1) - np.quantile(post, 0.25, axis=1)
    pre_c = pre - pre_mean.reshape(-1, 1)
    post_c = post - post_mean.reshape(-1, 1)
    pre_cusum = np.cumsum(pre_c, axis=1)
    post_cusum = np.cumsum(post_c, axis=1)
    pre_fft = np.abs(np.fft.rfft(pre, axis=1))
    post_fft = np.abs(np.fft.rfft(post, axis=1))
    pre_low = np.mean(pre_fft[:, 1:4], axis=1)
    post_low = np.mean(post_fft[:, 1:4], axis=1)
    pre_high = np.mean(pre_fft[:, 4:], axis=1)
    post_high = np.mean(post_fft[:, 4:], axis=1)
    quantiles = np.linspace(0.1, 0.9, 9)
    pre_q = np.quantile(pre, quantiles, axis=1).T
    post_q = np.quantile(post, quantiles, axis=1).T
    post_drawdown, post_drawup = drawdown_drawup(post)
    pre_drawdown, pre_drawup = drawdown_drawup(pre)
    features = np.column_stack(
        [
            post_mean - pre_mean,
            np.abs(post_mean - pre_mean),
            (post_mean - pre_mean) / pre_std,
            np.median(post, axis=1) - np.median(pre, axis=1),
            np.log((post_std + 1e-8) / (pre_std + 1e-8)),
            post_iqr - pre_iqr,
            np.log((post_iqr + 1e-8) / (pre_iqr + 1e-8)),
            np.mean(np.abs(post_d), axis=1) - np.mean(np.abs(pre_d), axis=1),
            np.log((safe_std(post_d) + 1e-8) / (safe_std(pre_d) + 1e-8)),
            autocorr1(post) - autocorr1(pre),
            slope(post) - slope(pre),
            slope(post_tail) - slope(pre_tail),
            np.mean(post_head, axis=1) - np.mean(pre_tail, axis=1),
            np.mean(post_tail, axis=1) - np.mean(post_head, axis=1),
            np.mean(np.abs(post_q - pre_q), axis=1),
            np.max(np.abs(post_q - pre_q), axis=1),
            np.mean(np.sort(post, axis=1) - np.sort(pre, axis=1), axis=1),
            np.mean(np.abs(np.sort(post, axis=1) - np.sort(pre, axis=1)), axis=1),
            np.max(np.abs(post_cusum), axis=1) / post_std,
            np.max(np.abs(pre_cusum), axis=1) / pre_std,
            np.max(np.abs(post_cusum), axis=1) / np.maximum(np.max(np.abs(pre_cusum), axis=1), 1e-8),
            np.log((post_low + 1e-8) / (pre_low + 1e-8)),
            np.log((post_high + 1e-8) / (pre_high + 1e-8)),
            np.log((post_high + 1e-8) / (post_low + 1e-8)) - np.log((pre_high + 1e-8) / (pre_low + 1e-8)),
            post_drawdown - pre_drawdown,
            post_drawup - pre_drawup,
            series[:, -1] - pre_mean,
            series[:, -1] - np.mean(pre_tail, axis=1),
        ]
    )
    names = [
        "mean_delta",
        "mean_delta_abs",
        "mean_delta_scaled",
        "median_delta",
        "std_log_ratio",
        "iqr_delta",
        "iqr_log_ratio",
        "absdiff_delta",
        "diff_std_log_ratio",
        "autocorr_delta",
        "slope_delta",
        "tail_slope_delta",
        "boundary_jump",
        "post_tail_drift",
        "quantile_l1_distance",
        "quantile_max_distance",
        "sorted_signed_distance",
        "sorted_l1_distance",
        "post_cusum_peak",
        "pre_cusum_peak",
        "cusum_peak_ratio",
        "low_freq_log_ratio",
        "high_freq_log_ratio",
        "spectral_shape_delta",
        "drawdown_delta",
        "drawup_delta",
        "last_vs_pre_mean",
        "last_vs_pre_tail",
    ]
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), names


def ensemble_report(
    archive: list[dict[str, Any]],
    train_y: np.ndarray,
    validation_y: np.ndarray,
    baseline: PredictionHoldoutResult,
) -> list[dict[str, Any]]:
    ranked = sorted(archive, key=lambda row: float(row["validation_auc"]), reverse=True)
    rows: list[dict[str, Any]] = []
    for k in (2, 3, 5):
        selected = ranked[: min(k, len(ranked))]
        if len(selected) < 2:
            continue
        train_pred = np.mean(np.column_stack([row["train_predictions"] for row in selected]), axis=1)
        validation_pred = np.mean(np.column_stack([row["validation_predictions"] for row in selected]), axis=1)
        rows.append(
            {
                "name": f"top_{len(selected)}_graph_archive",
                "members": [row["name"] for row in selected],
                "train_auc": float(roc_auc_score(train_y, train_pred)),
                "validation_auc": float(roc_auc_score(validation_y, validation_pred)),
                "validation_delta_vs_baseline": float(roc_auc_score(validation_y, validation_pred)) - float(baseline.validation_auc),
            }
        )
    if ranked:
        best = ranked[0]
        train_pred = 0.5 * baseline.train_predictions + 0.5 * best["train_predictions"]
        validation_pred = 0.5 * baseline.validation_predictions + 0.5 * best["validation_predictions"]
        rows.append(
            {
                "name": "baseline_plus_best_graph_blend",
                "members": ["baseline", best["name"]],
                "train_auc": float(roc_auc_score(train_y, train_pred)),
                "validation_auc": float(roc_auc_score(validation_y, validation_pred)),
                "validation_delta_vs_baseline": float(roc_auc_score(validation_y, validation_pred)) - float(baseline.validation_auc),
            }
        )
    return rows


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
            status_mark(bool(row.get("accepted"))),
            row.get("target_node", ""),
            row.get("primitive", ""),
            "yes" if row.get("source_backed") else "no",
            fmt_float(row.get("cv_train_auc")),
            fmt_float(row.get("holdout_validation_auc")),
            fmt_float(row.get("holdout_validation_delta_vs_baseline")),
        ]
        for row in payload["evolution"]["candidates"]
    ]
    dataset = payload["dataset"]
    baseline = payload["baseline"]
    evolved = payload["evolved_graph"]
    ensemble = payload["ensembles"]["best"] or {}
    split = payload["split"]
    return "\n\n".join(
        [
            "# Competition Id-Level Structural-Break Benchmark",
            str(payload["scope"]),
            (
                f"Dataset: `{dataset['name']}` from `{dataset['data_dir']}`, ids=`{dataset['n_samples']}`, "
                f"max_samples=`{dataset['max_samples']}`."
            ),
            (
                f"Split: `{split['manifest_method']}` by `{split['group_key']}`, sizes={split['sizes']}, "
                f"group overlaps={split['audit']['group_overlaps']}."
            ),
            f"Reduced test accessed: `{payload['reduced_test_access']['accessed']}`.",
            (
                f"Baseline validation AUC: `{fmt_float(baseline['validation_auc'])}`; "
                f"best evolved graph validation AUC: `{fmt_float(evolved['validation_auc'])}`; "
                f"delta: `{fmt_float(evolved['validation_delta_vs_baseline'])}`."
            ),
            (
                f"Best ensemble: `{ensemble.get('name', 'n/a')}` validation AUC `{fmt_float(ensemble.get('validation_auc'))}`, "
                f"delta vs baseline `{fmt_float(ensemble.get('validation_delta_vs_baseline'))}`."
            ),
            markdown_table(["Step", "Accepted", "Node", "Primitive", "Source", "CV Train AUC", "Holdout Val AUC", "Delta vs Baseline"], candidates),
        ]
    )


def run(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 211,
    series_length: int = 160,
    max_samples: int | None = None,
    steps: int = 24,
    folds: int = 3,
    max_configurations: int = 96,
    include_source_mutations: bool = True,
) -> tuple[Path, Path]:
    payload = build_report(
        output_dir,
        data_dir=data_dir,
        seed=seed,
        series_length=series_length,
        max_samples=max_samples,
        steps=steps,
        folds=folds,
        max_configurations=max_configurations,
        include_source_mutations=include_source_mutations,
    )
    return write_report(output_dir, "competition_event_benchmark", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an id-level ADIA-style structural-break benchmark.")
    output_argument(parser)
    seed_argument(parser, default=211)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_COMPETITION_DATA_DIR)
    parser.add_argument("--series-length", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-configurations", type=int, default=96)
    parser.add_argument("--disable-source-mutations", action="store_true")
    args = parser.parse_args(argv)
    print_report_paths(
        run(
            args.output,
            data_dir=args.data_dir,
            seed=args.seed,
            series_length=args.series_length,
            max_samples=args.max_samples,
            steps=args.steps,
            folds=args.folds,
            max_configurations=args.max_configurations,
            include_source_mutations=not args.disable_source_mutations,
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


def autocorr1(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=1, keepdims=True)
    return np.sum(centered[:, 1:] * centered[:, :-1], axis=1) / np.maximum(np.sum(centered[:, :-1] ** 2, axis=1), 1e-8)


def head_window(x: np.ndarray) -> np.ndarray:
    return x[:, : max(2, x.shape[1] // 4)]


def tail_window(x: np.ndarray) -> np.ndarray:
    return x[:, -max(2, x.shape[1] // 4) :]


def drawdown_drawup(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cumulative_max = np.maximum.accumulate(x, axis=1)
    cumulative_min = np.minimum.accumulate(x, axis=1)
    scale = np.maximum(safe_std(x), 1e-8)
    return np.max(cumulative_max - x, axis=1) / scale, np.max(x - cumulative_min, axis=1) / scale


if __name__ == "__main__":
    raise SystemExit(main())
