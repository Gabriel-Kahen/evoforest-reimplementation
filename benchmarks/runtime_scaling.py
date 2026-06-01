from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import median
from typing import Any

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.graph import Graph
from evoforest_arch.primitives import PrimitiveRegistry
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.synthetic import make_structural_break_data

from benchmarks.common import (
    evaluation_summary,
    fmt_float,
    markdown_table,
    output_argument,
    print_report_paths,
    quick_argument,
    report_scope,
    seed_argument,
    write_report,
)


def build_report(output_dir: Path, seed: int = 37, quick: bool = False) -> dict[str, Any]:
    del output_dir
    repeats = 1 if quick else 3
    rows: list[dict[str, Any]] = []
    rows.extend(configuration_cap_rows(seed=seed, quick=quick, repeats=repeats))
    rows.extend(dataset_scale_rows(seed=seed, quick=quick, repeats=repeats))
    rows.extend(output_feature_rows(seed=seed, quick=quick, repeats=repeats))
    return {
        "benchmark": "runtime_scaling",
        "scope": report_scope(),
        "seed": seed,
        "quick": quick,
        "summary": {
            "passed": all(row["seconds_median"] > 0.0 and row["evaluation"]["n_configs_evaluated"] > 0 for row in rows),
            "scenarios": len(rows),
            "repeats": repeats,
        },
        "scenarios": rows,
    }


def configuration_cap_rows(seed: int, quick: bool, repeats: int) -> list[dict[str, Any]]:
    caps = (1, 4, 8) if quick else (1, 8, 16, 32)
    dataset = make_structural_break_data(n_series=48 if quick else 96, length=64 if quick else 96, seed=seed)
    rows = []
    for cap in caps:
        rows.append(
            time_scenario(
                family="configuration_cap",
                setting=f"max_configurations={cap}",
                graph=build_seed_graph(),
                dataset_inputs=dataset.inputs(),
                y=dataset.y,
                evaluator=RidgeEvaluator(n_splits=3, seed=seed, max_configurations=cap),
                repeats=repeats,
            )
        )
    return rows


def dataset_scale_rows(seed: int, quick: bool, repeats: int) -> list[dict[str, Any]]:
    sizes = ((36, 48), (72, 80)) if quick else ((48, 64), (96, 96), (144, 128))
    rows = []
    for n_series, length in sizes:
        dataset = make_structural_break_data(n_series=n_series, length=length, seed=seed)
        rows.append(
            time_scenario(
                family="dataset_scale",
                setting=f"n_series={n_series}, length={length}",
                graph=build_seed_graph(),
                dataset_inputs=dataset.inputs(),
                y=dataset.y,
                evaluator=RidgeEvaluator(n_splits=3, seed=seed, max_configurations=8 if quick else 16),
                repeats=repeats,
            )
        )
    return rows


def output_feature_rows(seed: int, quick: bool, repeats: int) -> list[dict[str, Any]]:
    extra_counts = (0, 2, 4) if quick else (0, 4, 8, 12)
    dataset = make_structural_break_data(n_series=48 if quick else 96, length=64 if quick else 96, seed=seed)
    rows = []
    for extra in extra_counts:
        rows.append(
            time_scenario(
                family="output_feature_growth",
                setting=f"extra_output_alternatives={extra}",
                graph=graph_with_extra_outputs(extra),
                dataset_inputs=dataset.inputs(),
                y=dataset.y,
                evaluator=RidgeEvaluator(n_splits=3, seed=seed, max_configurations=8 if quick else 16),
                repeats=repeats,
            )
        )
    return rows


def graph_with_extra_outputs(extra_outputs: int) -> Graph:
    graph = build_seed_graph()
    registry = PrimitiveRegistry.default()
    parents = ("segment_stats", "trend_stats", "shape_stats")
    for index in range(extra_outputs):
        graph.nodes["output"].add_alternative(registry.build("projection_outputs", f"projection_extra_{index}", parents))
    return graph


def time_scenario(
    *,
    family: str,
    setting: str,
    graph: Graph,
    dataset_inputs: dict[str, object],
    y: Any,
    evaluator: RidgeEvaluator,
    repeats: int,
) -> dict[str, Any]:
    durations = []
    last_result = None
    for _ in range(repeats):
        trial_graph = graph.clone()
        start = time.perf_counter()
        last_result = evaluator.evaluate(trial_graph, dataset_inputs, y)
        durations.append(time.perf_counter() - start)
    if last_result is None:
        raise RuntimeError("Runtime scenario did not execute.")
    evaluation = evaluation_summary(last_result)
    cache = evaluation.get("cache", {})
    return {
        "family": family,
        "setting": setting,
        "seconds": durations,
        "seconds_median": float(median(durations)),
        "auc": float(last_result.auc),
        "evaluation": evaluation,
        "cache_hits": int(cache.get("hits", 0)) if isinstance(cache, dict) else 0,
        "cache_misses": int(cache.get("misses", 0)) if isinstance(cache, dict) else 0,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    rows = [
        [
            row["family"],
            row["setting"],
            fmt_float(row["seconds_median"], digits=5),
            fmt_float(row["auc"]),
            row["evaluation"]["n_configs_evaluated"],
            row["evaluation"]["n_configs_total"],
            row["evaluation"]["n_features"],
            row["cache_hits"],
            row["cache_misses"],
        ]
        for row in payload["scenarios"]
    ]
    return "\n\n".join(
        [
            "# Runtime Scaling Benchmark",
            str(payload["scope"]),
            f"Seed: `{payload['seed']}`",
            f"Repeats per scenario: `{payload['summary']['repeats']}`",
            markdown_table(["Family", "Setting", "Median Seconds", "AUC", "Configs Eval", "Configs Total", "Features", "Cache Hits", "Cache Misses"], rows),
        ]
    )


def run(output_dir: Path, seed: int = 37, quick: bool = False) -> tuple[Path, Path]:
    payload = build_report(output_dir, seed=seed, quick=quick)
    return write_report(output_dir, "runtime_scaling", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EvoForest runtime scaling benchmarks.")
    output_argument(parser)
    seed_argument(parser, default=37)
    quick_argument(parser)
    args = parser.parse_args(argv)
    print_report_paths(run(args.output, seed=args.seed, quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
