from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmarks.common import fmt_float, graph_summary, markdown_table, output_argument, print_report_paths, report_scope, seed_argument, write_report
from benchmarks.competition_event_benchmark import split_group_audit
from benchmarks.competition_event_multisplit_benchmark import (
    build_split_contexts,
    internal_test_report,
    objective_description,
    parse_seeds,
    prune_consensus_graph,
    score_graph_across_splits,
)
from evoforest_arch.competition import COMPETITION_DATASET_NAME, DEFAULT_COMPETITION_DATA_DIR
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import write_graph
from evoforest_arch.mutations import MutationDocument, MutationEngine, MutationSpec, built_in_mutations
from evoforest_arch.production import ProductionConfig, load_dataset_with_metadata
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.source_mutations import structural_break_source_mutations, validate_source_mutations


def build_report(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 211,
    split_seeds: tuple[int, ...] = (211, 223, 307),
    series_length: int = 160,
    max_samples: int | None = None,
    folds: int = 3,
    max_configurations: int = 64,
    stability_weight: float = 0.5,
    objective_mode: str = "delta",
    prune_tolerance: float = 0.001,
    screen_sources: bool = True,
    include_builtin_templates: bool = False,
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
        allow_source_mutations=True,
    )
    inputs, y, metadata = load_dataset_with_metadata(config.dataset_config())
    contexts, outer_split = build_split_contexts(inputs, y, outer_seed=seed, split_seeds=split_seeds)
    evaluator_kwargs = {
        "n_splits": int(folds),
        "max_configurations": int(max_configurations),
        "irls_steps": 0,
        "group_key": "sample_id",
    }

    seed_graph = build_seed_graph()
    source_specs = structural_break_source_mutations()
    source_checks = validate_source_mutations(seed_graph, source_specs, contexts[0].train_inputs)
    passed_specs = tuple(spec for spec, check in zip(source_specs, source_checks, strict=True) if check.passed)
    builtin_specs = tuple(built_in_mutations()) if include_builtin_templates else ()
    suite_specs = (*builtin_specs, *passed_specs)
    source_suite_graph = build_source_suite_graph(seed_graph, suite_specs)
    accepted_templates = [(spec.target_node, spec.alternative_id, 0) for spec in suite_specs]

    seed_score = score_graph_across_splits(
        seed_graph,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    template_screens = (
        screen_template_specs(
            seed_graph,
            suite_specs,
            contexts,
            evaluator_kwargs=evaluator_kwargs,
            stability_weight=stability_weight,
            objective_mode=objective_mode,
        )
        if screen_sources
        else []
    )
    suite_score = score_graph_across_splits(
        source_suite_graph,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
    )
    pruned_graph, prune_rows, pruned_score = prune_consensus_graph(
        source_suite_graph,
        accepted_templates,
        contexts,
        evaluator_kwargs=evaluator_kwargs,
        stability_weight=stability_weight,
        objective_mode=objective_mode,
        prune_tolerance=prune_tolerance,
        allow_source=True,
    )
    suite_path = write_graph(
        output_dir / "source_suite_graph.json",
        source_suite_graph,
        metadata={"benchmark": "competition_event_source_suite_benchmark", "stage": "pre_prune", "score": suite_score.to_dict()},
    )
    pruned_path = write_graph(
        output_dir / "pruned_source_suite_graph.json",
        pruned_graph,
        metadata={"benchmark": "competition_event_source_suite_benchmark", "stage": "post_prune", "score": pruned_score.to_dict()},
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
    source_screens = [row for row in template_screens if row["source_backed"]]
    best_source_screen = max(source_screens, key=lambda row: float(row["score"]["objective"])) if source_screens else None
    best_template_screen = max(template_screens, key=lambda row: float(row["score"]["objective"])) if template_screens else None
    best_graph = max(
        (
            ("seed_graph", seed_score),
            ("source_suite_graph", suite_score),
            ("pruned_source_suite_graph", pruned_score),
        ),
        key=lambda row: float(row[1].objective),
    )
    return {
        "benchmark": "competition_event_source_suite_benchmark",
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
            "screen_sources": bool(screen_sources),
            "include_builtin_templates": bool(include_builtin_templates),
            "objective": objective_description(objective_mode),
        },
        "outer_split": outer_split,
        "split_audits": split_audits,
        "reduced_test_access": {
            "accessed": bool(reduced_read_paths),
            "read_paths": read_paths,
            "reduced_read_paths": reduced_read_paths,
            "checked_paths": ["X_test.reduced.parquet", "y_test_index.reduced.parquet"],
        },
        "source_mutations": {
            "enabled": True,
            "templates": len(source_specs),
            "passed_repair_checks": sum(1 for check in source_checks if check.passed),
            "checks": [check.to_dict() for check in source_checks],
            "best_isolated_source": best_source_screen,
            "isolated_sources": source_screens,
        },
        "template_suite": {
            "builtin_templates": len(builtin_specs),
            "source_templates": len(passed_specs),
            "added_templates": len(suite_specs),
            "include_builtin_templates": bool(include_builtin_templates),
            "best_isolated_template": best_template_screen,
            "isolated_templates": template_screens,
        },
        "seed_graph": seed_score.to_dict(),
        "source_suite_graph": {
            **suite_score.to_dict(),
            "path": str(suite_path),
            "graph": graph_summary(source_suite_graph),
        },
        "pruned_source_suite_graph": {
            **pruned_score.to_dict(),
            "path": str(pruned_path),
            "graph": graph_summary(pruned_graph),
        },
        "internal_test": internal_test,
        "summary": {
            "best_graph_by_objective": best_graph[0],
            "best_objective": float(best_graph[1].objective),
            "best_mean_validation_auc": float(best_graph[1].mean_validation_auc),
            "best_mean_delta_vs_baseline": float(best_graph[1].mean_delta_vs_baseline),
            "source_suite_delta_vs_seed": float(suite_score.objective) - float(seed_score.objective),
            "pruned_delta_vs_suite": float(pruned_score.objective) - float(suite_score.objective),
        },
        "pruning": {
            "attempted": len(prune_rows),
            "removed": sum(1 for row in prune_rows if row.get("removed")),
            "rows": prune_rows,
        },
    }


def build_source_suite_graph(seed_graph: Graph, specs: tuple[MutationSpec, ...]) -> Graph:
    if not specs:
        return seed_graph.clone()
    return MutationEngine(allow_source=True).apply_document(seed_graph, MutationDocument(add=specs)).graph


def screen_template_specs(
    seed_graph: Graph,
    specs: tuple[MutationSpec, ...],
    contexts: tuple[Any, ...],
    *,
    evaluator_kwargs: dict[str, Any],
    stability_weight: float,
    objective_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    engine = MutationEngine(allow_source=True)
    for spec in specs:
        graph = engine.apply(seed_graph, spec)
        score = score_graph_across_splits(
            graph,
            contexts,
            evaluator_kwargs=evaluator_kwargs,
            stability_weight=stability_weight,
            objective_mode=objective_mode,
        )
        rows.append(
            {
                "alternative_id": spec.alternative_id,
                "target_node": spec.target_node,
                "primitive": spec.primitive,
                "source_backed": bool(spec.source),
                "description": spec.description,
                "score": score.to_dict(),
            }
        )
    return sorted(rows, key=lambda row: float(row["score"]["objective"]), reverse=True)


def markdown_report(payload: dict[str, Any]) -> str:
    template_rows = [
        [
            row["alternative_id"],
            row["target_node"],
            row["primitive"],
            "yes" if row["source_backed"] else "no",
            fmt_float(row["score"]["mean_validation_auc"]),
            fmt_float(row["score"]["mean_delta_vs_baseline"]),
            fmt_float(row["score"]["min_delta_vs_baseline"]),
            fmt_float(row["score"]["objective"]),
        ]
        for row in payload["template_suite"]["isolated_templates"]
    ]
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
    seed = payload["seed_graph"]
    suite = payload["source_suite_graph"]
    pruned = payload["pruned_source_suite_graph"]
    internal = payload["internal_test"]
    summary_rows = [
        ["seed_graph", fmt_float(seed["mean_validation_auc"]), fmt_float(seed["mean_delta_vs_baseline"]), fmt_float(seed["objective"])],
        ["source_suite_graph", fmt_float(suite["mean_validation_auc"]), fmt_float(suite["mean_delta_vs_baseline"]), fmt_float(suite["objective"])],
        ["pruned_source_suite_graph", fmt_float(pruned["mean_validation_auc"]), fmt_float(pruned["mean_delta_vs_baseline"]), fmt_float(pruned["objective"])],
    ]
    return "\n\n".join(
        [
            "# Competition Source-Suite Structural-Break Benchmark",
            str(payload["scope"]),
            (
                f"Dataset: `{payload['dataset']['name']}` from `{payload['dataset']['data_dir']}`, "
                f"ids=`{payload['dataset']['n_samples']}`, max_samples=`{payload['dataset']['max_samples']}`."
            ),
            f"Config: `{payload['benchmark_config']}`",
            f"Reduced test accessed: `{payload['reduced_test_access']['accessed']}`.",
            (
                f"Best graph by objective: `{payload['summary']['best_graph_by_objective']}` with mean validation AUC "
                f"`{fmt_float(payload['summary']['best_mean_validation_auc'])}` and mean delta "
                f"`{fmt_float(payload['summary']['best_mean_delta_vs_baseline'])}`."
            ),
            (
                f"Internal non-selection test audit for pruned suite: mean graph AUC `{fmt_float(internal['mean_graph_auc'])}`, "
                f"mean delta `{fmt_float(internal['mean_delta_vs_baseline'])}`, minimum delta `{fmt_float(internal['min_delta_vs_baseline'])}`."
            ),
            markdown_table(["Graph", "Mean Val AUC", "Mean Delta", "Objective"], summary_rows),
            markdown_table(["Isolated Template", "Node", "Primitive", "Source", "Mean Val AUC", "Mean Delta", "Min Delta", "Objective"], template_rows),
            markdown_table(["Pruned Alternative", "Removed", "Objective Before", "Objective After", "Delta"], prune_rows),
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
    folds: int = 3,
    max_configurations: int = 64,
    objective_mode: str = "delta",
    screen_sources: bool = True,
    include_builtin_templates: bool = False,
) -> tuple[Path, Path]:
    payload = build_report(
        output_dir,
        data_dir=data_dir,
        seed=seed,
        split_seeds=split_seeds,
        series_length=series_length,
        max_samples=max_samples,
        folds=folds,
        max_configurations=max_configurations,
        objective_mode=objective_mode,
        screen_sources=screen_sources,
        include_builtin_templates=include_builtin_templates,
    )
    return write_report(output_dir, "competition_event_source_suite_benchmark", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate all trusted structural-break source mutations as a suite, then prune under grouped multi-split validation.")
    output_argument(parser)
    seed_argument(parser, default=211)
    parser.add_argument("--split-seeds", type=parse_seeds, default=(211, 223, 307), help="Comma-separated grouped split seeds.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_COMPETITION_DATA_DIR)
    parser.add_argument("--series-length", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-configurations", type=int, default=64)
    parser.add_argument("--objective-mode", choices=("delta", "auc"), default="delta")
    parser.add_argument("--disable-source-screen", action="store_true")
    parser.add_argument("--include-builtins", action="store_true", help="Add built-in mutation templates to the assembled suite before pruning.")
    args = parser.parse_args(argv)
    print_report_paths(
        run(
            args.output,
            data_dir=args.data_dir,
            seed=args.seed,
            split_seeds=args.split_seeds,
            series_length=args.series_length,
            max_samples=args.max_samples,
            folds=args.folds,
            max_configurations=args.max_configurations,
            objective_mode=args.objective_mode,
            screen_sources=not args.disable_source_screen,
            include_builtin_templates=args.include_builtins,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
