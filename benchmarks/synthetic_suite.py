from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.seed import build_structural_break_seed_graph
from evoforest_arch.synthetic import TimeSeriesDataset, make_structural_break_data

from benchmarks.common import (
    evaluation_summary,
    fmt_float,
    markdown_table,
    output_argument,
    print_report_paths,
    quick_argument,
    report_scope,
    seed_argument,
    status_mark,
    write_report,
)


def build_report(output_dir: Path, seed: int = 23, quick: bool = False) -> dict[str, Any]:
    del output_dir
    n_series = 54 if quick else 108
    length = 64 if quick else 96
    max_configurations = 8 if quick else 24
    clean = make_structural_break_data(n_series=n_series, length=length, seed=seed)
    outlier_heavy = add_target_independent_outliers(clean, seed=seed + 100)

    rows = [
        run_case(
            name="structural_break_full_search",
            mechanism="capped configuration search over reusable graph computations",
            dataset=clean,
            seed=seed,
            max_configurations=max_configurations,
            config=None,
            threshold=0.60,
        ),
        run_case(
            name="sample_weight_boundary_energy",
            mechanism="ridge_w nonuniform sample weights",
            dataset=clean,
            seed=seed,
            max_configurations=max_configurations,
            config={"ridge_w": "boundary_energy"},
            baseline_config={"ridge_w": "uniform"},
            extra_pass=lambda row: float(row["evidence"]["ridge_w"].get("std", 0.0)) > 0.0,
            threshold=0.60,
        ),
        run_case(
            name="residual_huber_irls",
            mechanism="ridge_g iterative residual reweighting on heavy-tailed data",
            dataset=outlier_heavy,
            seed=seed,
            max_configurations=max_configurations,
            config={"ridge_g": "huber"},
            baseline_config={"ridge_g": "identity"},
            extra_pass=lambda row: bool(row["evidence"]["global_ridge"].get("residual_reweighted", False)),
            threshold=0.35,
        ),
        run_case(
            name="callable_sigmoid_gate",
            mechanism="callable-node activation family selected by configuration",
            dataset=clean,
            seed=seed,
            max_configurations=max_configurations,
            config={"activation": "sigmoid_gate"},
            baseline_config={"activation": "identity"},
            extra_pass=lambda row: any("sigmoid_gate" in name for name in row["feature_names"]),
            threshold=0.55,
        ),
        run_case(
            name="global_projection_feature",
            mechanism="output feature backed by persistent trainable globals",
            dataset=clean,
            seed=seed,
            max_configurations=max_configurations,
            config={"activation": "sigmoid_gate"},
            extra_pass=lambda row: any("global_projection" in name for name in row["feature_names"]),
            threshold=0.55,
        ),
    ]
    return {
        "benchmark": "synthetic_suite",
        "scope": report_scope(),
        "seed": seed,
        "quick": quick,
        "summary": {
            "passed": sum(1 for row in rows if row["passed"]),
            "total": len(rows),
            "all_passed": all(row["passed"] for row in rows),
        },
        "datasets": {
            "clean": dataset_summary(clean),
            "outlier_heavy": dataset_summary(outlier_heavy),
        },
        "cases": rows,
    }


def add_target_independent_outliers(dataset: TimeSeriesDataset, seed: int) -> TimeSeriesDataset:
    rng = np.random.default_rng(seed)
    values = dataset.values.copy()
    n_spikes = max(1, values.shape[0] // 5)
    rows = rng.choice(values.shape[0], size=n_spikes, replace=False)
    cols = rng.integers(0, values.shape[1], size=n_spikes)
    signs = rng.choice([-1.0, 1.0], size=n_spikes)
    values[rows, cols] += signs * rng.uniform(5.0, 8.0, size=n_spikes)
    return replace(dataset, values=values)


def run_case(
    *,
    name: str,
    mechanism: str,
    dataset: TimeSeriesDataset,
    seed: int,
    max_configurations: int,
    config: dict[str, str] | None,
    threshold: float,
    baseline_config: dict[str, str] | None = None,
    extra_pass: Any = None,
) -> dict[str, Any]:
    graph = build_structural_break_seed_graph()
    evaluator = RidgeEvaluator(n_splits=3, seed=seed, max_configurations=max_configurations, irls_steps=2)
    result = evaluator.evaluate(graph, dataset.inputs(), dataset.y, config=config)
    baseline_score = None
    if baseline_config is not None:
        baseline = evaluator.evaluate(build_structural_break_seed_graph(), dataset.inputs(), dataset.y, config=baseline_config)
        baseline_score = float(baseline.score)
    row = {
        "name": name,
        "mechanism": mechanism,
        "passed": float(result.score) >= threshold,
        "threshold_score": threshold,
        "score": float(result.score),
        "baseline_score": baseline_score,
        "delta_vs_baseline": None if baseline_score is None else float(result.score - baseline_score),
        "config": result.config,
        "feature_names": result.feature_names,
        "evidence": {
            **evaluation_summary(result),
            "global_ridge": result.diagnostics.get("global_ridge", {}),
            "linear_shap": result.diagnostics.get("linear_shap", {}),
        },
    }
    if extra_pass is not None:
        row["passed"] = bool(row["passed"] and extra_pass(row))
    return row


def dataset_summary(dataset: TimeSeriesDataset) -> dict[str, object]:
    return {
        "n_series": int(dataset.values.shape[0]),
        "length": int(dataset.values.shape[1]),
        "boundary": int(dataset.boundary),
        "target_mean": float(np.mean(dataset.y)),
        "target_std": float(np.std(dataset.y)),
        "value_std": float(np.std(dataset.values)),
    }


def markdown_report(payload: dict[str, Any]) -> str:
    rows = [
        [
            row["name"],
            status_mark(bool(row["passed"])),
            row["mechanism"],
            fmt_float(row["score"]),
            fmt_float(row["baseline_score"]),
            fmt_float(row["delta_vs_baseline"]),
        ]
        for row in payload["cases"]
    ]
    return "\n\n".join(
        [
            "# Synthetic Mechanism Benchmark",
            str(payload["scope"]),
            f"Seed: `{payload['seed']}`",
            f"Passed: `{payload['summary']['passed']}/{payload['summary']['total']}`",
            markdown_table(["Case", "Status", "Mechanism", "Score", "Baseline Score", "Delta"], rows),
        ]
    )


def run(output_dir: Path, seed: int = 23, quick: bool = False) -> tuple[Path, Path]:
    payload = build_report(output_dir, seed=seed, quick=quick)
    return write_report(output_dir, "synthetic_suite", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic architecture-mechanism benchmarks.")
    output_argument(parser)
    seed_argument(parser, default=23)
    quick_argument(parser)
    args = parser.parse_args(argv)
    print_report_paths(run(args.output, seed=args.seed, quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
