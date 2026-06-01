from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.graph import Graph
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.synthetic import make_structural_break_data

from benchmarks.common import (
    evaluation_summary,
    fmt_float,
    graph_summary,
    markdown_table,
    output_argument,
    print_report_paths,
    quick_argument,
    report_scope,
    seed_argument,
    status_mark,
    write_report,
)


def build_report(output_dir: Path, seed: int = 29, quick: bool = False) -> dict[str, Any]:
    del output_dir
    n_series = 54 if quick else 108
    length = 64 if quick else 96
    max_configurations = 8 if quick else 24
    dataset = make_structural_break_data(n_series=n_series, length=length, seed=seed)
    evaluator = RidgeEvaluator(n_splits=3, seed=seed, max_configurations=max_configurations, irls_steps=2)

    full_graph = build_seed_graph()
    full_result = evaluator.evaluate(full_graph, dataset.inputs(), dataset.y)
    full_eval = evaluation_summary(full_result)
    full_graph_summary = graph_summary(full_graph)

    ablations = [
        ("full", "No ablation; capped configuration search over the full seed graph.", build_seed_graph(), None),
        ("default_path_only", "Disable configuration search by scoring only the default selected path.", build_seed_graph(), build_seed_graph().default_config()),
        ("raw_output_only", "Remove output ensemble diversity by keeping only raw concatenated outputs.", output_raw_only(), None),
        ("no_callable_choice", "Collapse callable family search to the identity callable.", only_alternatives("activation", ("identity",)), None),
        ("no_fitting_choice", "Collapse fitting-rule search to uniform sample weights and identity residual weights.", no_fitting_choice(), None),
        ("no_global_projection", "Remove the output alternative that uses persistent global projection parameters.", without_alternatives("output", ("projection",)), None),
        ("no_spectral_shape", "Remove spectral shape alternatives from the intermediate search space.", without_alternatives("shape_stats", ("spectral",)), None),
    ]

    rows = []
    for name, description, graph, config in ablations:
        result = evaluator.evaluate(graph, dataset.inputs(), dataset.y, config=config)
        eval_summary = evaluation_summary(result)
        row = {
            "name": name,
            "description": description,
            "passed": ablation_surface_changed(name, eval_summary, graph_summary(graph), full_eval, full_graph_summary),
            "auc": float(result.auc),
            "delta_auc_vs_full": float(result.auc - full_result.auc),
            "evaluation": eval_summary,
            "graph": graph_summary(graph),
        }
        rows.append(row)

    return {
        "benchmark": "ablation_suite",
        "scope": report_scope(),
        "seed": seed,
        "quick": quick,
        "summary": {
            "passed": sum(1 for row in rows if row["passed"]),
            "total": len(rows),
            "all_passed": all(row["passed"] for row in rows),
            "full_auc": float(full_result.auc),
        },
        "ablations": rows,
    }


def output_raw_only() -> Graph:
    graph = build_seed_graph()
    return only_alternatives_in_graph(graph, "output", ("raw_concat",))


def no_fitting_choice() -> Graph:
    graph = build_seed_graph()
    only_alternatives_in_graph(graph, "ridge_w", ("uniform",))
    only_alternatives_in_graph(graph, "ridge_g", ("identity",))
    return graph


def only_alternatives(node_name: str, alternative_ids: tuple[str, ...]) -> Graph:
    graph = build_seed_graph()
    return only_alternatives_in_graph(graph, node_name, alternative_ids)


def without_alternatives(node_name: str, alternative_ids: tuple[str, ...]) -> Graph:
    graph = build_seed_graph()
    remove = set(alternative_ids)
    graph.nodes[node_name].alternatives = [alt for alt in graph.nodes[node_name].alternatives if alt.id not in remove]
    return graph


def only_alternatives_in_graph(graph: Graph, node_name: str, alternative_ids: tuple[str, ...]) -> Graph:
    keep = set(alternative_ids)
    graph.nodes[node_name].alternatives = [alt for alt in graph.nodes[node_name].alternatives if alt.id in keep]
    if not graph.nodes[node_name].alternatives:
        raise ValueError(f"Ablation removed every alternative from {node_name}.")
    return graph


def ablation_surface_changed(
    name: str,
    evaluation: dict[str, object],
    graph: dict[str, object],
    full_evaluation: dict[str, object],
    full_graph: dict[str, object],
) -> bool:
    if name == "full":
        return bool(evaluation["n_configs_evaluated"] > 1 and evaluation["n_features"] > 1)
    if name == "default_path_only":
        return int(evaluation["n_configs_evaluated"]) == 1
    return bool(
        int(evaluation["n_configs_total"]) != int(full_evaluation["n_configs_total"])
        or int(evaluation["n_features"]) != int(full_evaluation["n_features"])
        or int(graph["alternatives"]) != int(full_graph["alternatives"])
    )


def markdown_report(payload: dict[str, Any]) -> str:
    rows = [
        [
            row["name"],
            status_mark(bool(row["passed"])),
            fmt_float(row["auc"]),
            fmt_float(row["delta_auc_vs_full"]),
            row["evaluation"]["n_configs_evaluated"],
            row["evaluation"]["n_configs_total"],
            row["evaluation"]["n_features"],
        ]
        for row in payload["ablations"]
    ]
    return "\n\n".join(
        [
            "# Ablation Benchmark",
            str(payload["scope"]),
            f"Seed: `{payload['seed']}`",
            f"Full graph AUC: `{fmt_float(payload['summary']['full_auc'])}`",
            f"Passed: `{payload['summary']['passed']}/{payload['summary']['total']}`",
            markdown_table(["Ablation", "Status", "AUC", "Delta", "Configs Eval", "Configs Total", "Features"], rows),
        ]
    )


def run(output_dir: Path, seed: int = 29, quick: bool = False) -> tuple[Path, Path]:
    payload = build_report(output_dir, seed=seed, quick=quick)
    return write_report(output_dir, "ablation_suite", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EvoForest architecture ablation benchmarks.")
    output_argument(parser)
    seed_argument(parser, default=29)
    quick_argument(parser)
    args = parser.parse_args(argv)
    print_report_paths(run(args.output, seed=args.seed, quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
