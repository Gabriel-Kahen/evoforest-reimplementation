from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common import fmt_float, graph_summary, markdown_table, output_argument, print_report_paths, report_scope, seed_argument, write_report
from benchmarks.competition_event_multisplit_benchmark import parse_seeds, score_objective
from benchmarks.competition_row_benchmark import evaluate_baseline_holdout, graph_holdout_score, split_group_audit
from benchmarks.competition_row_multisplit_benchmark import (
    ROW_BASELINE_SPEC,
    ROW_MULTISCALE_TAIL_SPEC,
    ROW_TIME_BASIS_SPEC,
    build_output_only_graph,
    build_row_split_contexts,
    validation_split_group_overlap,
)
from evoforest_arch.competition import COMPETITION_ROW_DATASET_NAME, DEFAULT_COMPETITION_DATA_DIR
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import write_graph
from evoforest_arch.production import ProductionConfig, load_dataset_with_metadata
from evoforest_arch.seed import build_seed_graph


def build_report(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 113,
    split_seeds: tuple[int, ...] = (113, 127, 149),
    series_length: int = 480,
    max_samples: int | None = None,
    max_ids: int | None = 10000,
    max_rows_per_id: int | None = 32,
    row_stride: int = 1,
    stability_weight: float = 0.5,
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
        folds=3,
        max_configurations=1,
        irls_steps=0,
    )
    inputs, y, metadata = load_dataset_with_metadata(config.dataset_config())
    contexts, outer_split = build_row_split_contexts(inputs, y, outer_seed=seed, split_seeds=split_seeds)
    seed_graph = build_seed_graph()
    graph_specs = {
        "row_baseline_only_graph": (ROW_BASELINE_SPEC,),
        "row_baseline_time_graph": (ROW_BASELINE_SPEC, ROW_TIME_BASIS_SPEC),
        "row_baseline_tail_graph": (ROW_BASELINE_SPEC, ROW_MULTISCALE_TAIL_SPEC),
        "row_baseline_time_tail_graph": (ROW_BASELINE_SPEC, ROW_TIME_BASIS_SPEC, ROW_MULTISCALE_TAIL_SPEC),
    }
    graphs = {
        name: build_output_only_graph(seed_graph, specs, allow_source=False)
        for name, specs in graph_specs.items()
    }
    graph_reports = {
        name: score_output_graph(graph, contexts, stability_weight=stability_weight)
        for name, graph in graphs.items()
    }
    graph_paths = {
        name: write_graph(
            output_dir / f"{name}.json",
            graph,
            metadata={"benchmark": "competition_row_focused_graph_benchmark", "stage": name, "score": graph_reports[name]},
        )
        for name, graph in graphs.items()
    }
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
    best_by_validation = max(graph_reports, key=lambda name: float(graph_reports[name]["objective"]))
    best_by_internal_test = max(graph_reports, key=lambda name: float(graph_reports[name]["test"]["mean_auc"]))
    return {
        "benchmark": "competition_row_focused_graph_benchmark",
        "scope": report_scope(),
        "seed": int(seed),
        "dataset": metadata,
        "dataset_config": config.dataset_config(),
        "benchmark_config": {
            "split_seeds": list(split_seeds),
            "stability_weight": float(stability_weight),
            "graphs": {name: [spec.primitive for spec in specs] for name, specs in graph_specs.items()},
            "selection_policy": "best graph by grouped multi-split validation objective; internal test is reported for audit only",
        },
        "outer_split": outer_split,
        "split_contexts": [context.to_dict() for context in contexts],
        "split_audits": split_audits,
        "validation_split_group_overlap": validation_split_group_overlap(contexts),
        "reduced_test_access": {
            "accessed": bool(reduced_read_paths),
            "read_paths": read_paths,
            "reduced_read_paths": reduced_read_paths,
            "checked_paths": ["X_test.reduced.parquet", "y_test.reduced.parquet", "y_test_index.reduced.parquet"],
        },
        "graphs": {
            name: {
                **graph_reports[name],
                "path": str(graph_paths[name]),
                "graph": graph_summary(graphs[name]),
            }
            for name in graphs
        },
        "summary": {
            "best_graph_by_validation": best_by_validation,
            "best_validation_objective": float(graph_reports[best_by_validation]["objective"]),
            "best_mean_validation_auc": float(graph_reports[best_by_validation]["validation"]["mean_auc"]),
            "best_mean_validation_delta_vs_baseline": float(graph_reports[best_by_validation]["validation"]["mean_delta_vs_baseline"]),
            "best_internal_test_mean_auc": float(graph_reports[best_by_validation]["test"]["mean_auc"]),
            "best_internal_test_mean_delta_vs_baseline": float(graph_reports[best_by_validation]["test"]["mean_delta_vs_baseline"]),
            "best_graph_by_internal_test": best_by_internal_test,
            "best_internal_test_audit_auc": float(graph_reports[best_by_internal_test]["test"]["mean_auc"]),
        },
    }


def score_output_graph(graph: Graph, contexts: tuple[Any, ...], *, stability_weight: float) -> dict[str, Any]:
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    config = graph.default_config()
    for context in contexts:
        validation_score = graph_holdout_score(
            graph,
            context.train_inputs,
            context.train_y,
            context.validation_inputs,
            context.validation_y,
            config=config,
        )
        test_score = graph_holdout_score(
            graph,
            context.train_inputs,
            context.train_y,
            context.test_inputs,
            context.test_y,
            config=config,
        )
        test_baseline = evaluate_baseline_holdout(context.train_inputs, context.train_y, context.test_inputs, context.test_y)
        validation_rows.append(
            {
                "seed": int(context.seed),
                "train_auc": float(validation_score.train_auc),
                "auc": float(validation_score.validation_auc),
                "baseline_auc": float(context.baseline.validation_auc),
                "delta_vs_baseline": float(validation_score.validation_auc) - float(context.baseline.validation_auc),
                "alpha": float(validation_score.alpha),
                "n_features": int(validation_score.n_features),
            }
        )
        test_rows.append(
            {
                "seed": int(context.seed),
                "auc": float(test_score.validation_auc),
                "baseline_auc": float(test_baseline.validation_auc),
                "delta_vs_baseline": float(test_score.validation_auc) - float(test_baseline.validation_auc),
                "alpha": float(test_score.alpha),
                "n_features": int(test_score.n_features),
            }
        )
    validation = summarize_rows(validation_rows)
    test = summarize_rows(test_rows)
    return {
        "validation": validation,
        "test": test,
        "objective": score_objective(
            mean_validation_auc=validation["mean_auc"],
            min_validation_auc=validation["min_auc"],
            mean_delta_vs_baseline=validation["mean_delta_vs_baseline"],
            min_delta_vs_baseline=validation["min_delta_vs_baseline"],
            stability_weight=stability_weight,
            objective_mode="auc",
        ),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = np.asarray([row["auc"] for row in rows], dtype=np.float64)
    deltas = np.asarray([row["delta_vs_baseline"] for row in rows], dtype=np.float64)
    return {
        "mean_auc": float(np.mean(aucs)) if aucs.size else 0.5,
        "min_auc": float(np.min(aucs)) if aucs.size else 0.5,
        "mean_delta_vs_baseline": float(np.mean(deltas)) if deltas.size else 0.0,
        "min_delta_vs_baseline": float(np.min(deltas)) if deltas.size else 0.0,
        "splits": rows,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    graph_rows = []
    for name, report in payload["graphs"].items():
        graph_rows.append(
            [
                name,
                fmt_float(report["validation"]["mean_auc"]),
                fmt_float(report["validation"]["mean_delta_vs_baseline"]),
                fmt_float(report["validation"]["min_auc"]),
                fmt_float(report["test"]["mean_auc"]),
                fmt_float(report["test"]["mean_delta_vs_baseline"]),
                fmt_float(report["objective"]),
            ]
        )
    dataset = payload["dataset"]
    summary = payload["summary"]
    return "\n\n".join(
        [
            "# Competition Row Focused Graph Benchmark",
            str(payload["scope"]),
            (
                f"Dataset: `{dataset['name']}` from `{dataset['data_dir']}`, rows=`{dataset['n_samples']}`, "
                f"ids=`{dataset['n_ids']}`, max_ids=`{dataset['max_ids']}`, max_rows_per_id=`{dataset['max_rows_per_id']}`."
            ),
            f"Config: `{payload['benchmark_config']}`",
            f"Reduced test accessed: `{payload['reduced_test_access']['accessed']}`.",
            (
                f"Best validation graph: `{summary['best_graph_by_validation']}` with mean validation AUC "
                f"`{fmt_float(summary['best_mean_validation_auc'])}` and internal test mean AUC "
                f"`{fmt_float(summary['best_internal_test_mean_auc'])}`."
            ),
            markdown_table(
                ["Graph", "Mean Val AUC", "Mean Val Delta", "Min Val AUC", "Mean Test AUC", "Mean Test Delta", "Objective"],
                graph_rows,
            ),
        ]
    )


def run(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 113,
    split_seeds: tuple[int, ...] = (113, 127, 149),
    series_length: int = 480,
    max_samples: int | None = None,
    max_ids: int | None = 10000,
    max_rows_per_id: int | None = 32,
    row_stride: int = 1,
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
    )
    return write_report(output_dir, "competition_row_focused_graph_benchmark", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a focused graph audit for row-level ADIA-style output primitives.")
    output_argument(parser)
    seed_argument(parser, default=113)
    parser.add_argument("--split-seeds", type=parse_seeds, default=(113, 127, 149), help="Comma-separated grouped split seeds.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_COMPETITION_DATA_DIR)
    parser.add_argument("--series-length", type=int, default=480)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-ids", type=int, default=10000)
    parser.add_argument("--max-rows-per-id", type=int, default=32)
    parser.add_argument("--row-stride", type=int, default=1)
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
