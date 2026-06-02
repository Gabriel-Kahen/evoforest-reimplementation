from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from evoforest_arch.agents import EngineerAgent, ScientistAgent
from evoforest_arch.competition import COMPETITION_DATASET_NAME, DEFAULT_COMPETITION_DATA_DIR
from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.mutations import MutationEngine
from evoforest_arch.production import ProductionConfig, load_dataset_with_metadata
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.splits import make_split_manifest, split_dataset

from benchmarks.common import fmt_float, markdown_table, output_argument, print_report_paths, report_scope, seed_argument, status_mark, write_report


def build_report(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 91,
    max_samples: int | None = 1000,
    series_length: int = 160,
    steps: int = 6,
    folds: int = 3,
    max_configurations: int = 96,
    min_train_improvement: float = -0.005,
    min_validation_improvement: float = 1e-6,
) -> dict[str, Any]:
    del output_dir
    config = ProductionConfig(
        output_dir=Path("unused"),
        dataset_name=COMPETITION_DATASET_NAME,
        data_dir=data_dir,
        seed=seed,
        max_samples=max_samples,
        competition_series_length=series_length,
        folds=folds,
        max_configurations=max_configurations,
        irls_steps=0,
        min_train_improvement=min_train_improvement,
        min_validation_improvement=min_validation_improvement,
    )
    inputs, y, metadata = load_dataset_with_metadata(config.dataset_config())
    split_manifest = make_split_manifest(inputs, y, seed=seed, validation_fraction=0.2, test_fraction=0.2)
    splits = split_dataset(inputs, y, split_manifest)
    train_inputs, train_y = splits["train"]
    validation_inputs, validation_y = splits["validation"]
    evaluator = RidgeEvaluator(**config.evaluator_config())
    scientist = ScientistAgent()
    engineer = EngineerAgent()
    mutation_engine = MutationEngine()
    current_graph = build_seed_graph()
    best_graph = current_graph.clone()
    best_train = evaluator.evaluate(current_graph, train_inputs, train_y, update_graph=True)
    best_validation = evaluator.evaluate(current_graph, validation_inputs, validation_y, config=best_train.config, update_graph=False)
    seed_train_auc = float(best_train.auc)
    seed_validation_auc = float(best_validation.auc)
    seed_feature_count = len(best_train.feature_names)
    seed_alternatives = sum(len(node.alternatives) for node in current_graph.nodes.values())
    rows: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        hypotheses = scientist.generate(current_graph, best_train)
        document = engineer.synthesize(current_graph, best_train, hypotheses, step=step, island=None, rng=np.random.default_rng(seed))
        application = mutation_engine.apply_document(current_graph, document)
        candidate_graph = application.graph
        candidate_train = evaluator.evaluate(candidate_graph, train_inputs, train_y, update_graph=False)
        candidate_validation = evaluator.evaluate(candidate_graph, validation_inputs, validation_y, config=candidate_train.config, update_graph=False)
        train_delta = float(candidate_train.auc) - float(best_train.auc)
        validation_delta = float(candidate_validation.auc) - float(best_validation.auc)
        accepted = train_delta > min_train_improvement and validation_delta > min_validation_improvement
        add = document.add[0] if document.add else None
        used_by_selected_config = bool(
            add
            and (
                add.target_node == "output"
                or candidate_train.config.get(add.target_node) == add.alternative_id
            )
        )
        row = {
            "step": step,
            "accepted": accepted,
            "target_node": add.target_node if add else "",
            "primitive": add.primitive if add else "",
            "alternative_id": add.alternative_id if add else "",
            "train_auc": float(candidate_train.auc),
            "validation_auc": float(candidate_validation.auc),
            "train_delta": train_delta,
            "validation_delta": validation_delta,
            "n_features": len(candidate_train.feature_names),
            "n_alternatives": sum(len(node.alternatives) for node in candidate_graph.nodes.values()),
            "used_by_selected_config": used_by_selected_config,
            "maintenance": application.maintenance.to_dict(),
            "config": candidate_train.config,
        }
        rows.append(row)
        if accepted:
            current_graph = candidate_graph
            best_graph = candidate_graph.clone()
            best_train = candidate_train
            best_validation = candidate_validation

    primitives = [row["primitive"] for row in rows]
    unique_primitives = sorted(set(primitives))
    best_row = max(rows, key=lambda row: float(row["validation_auc"])) if rows else None
    best_validation_auc = max([seed_validation_auc, *(float(row["validation_auc"]) for row in rows)])
    validation_delta = best_validation_auc - seed_validation_auc
    duplicate_proposals = len(unique_primitives) != len(primitives)
    accepted_count = sum(1 for row in rows if row["accepted"])
    feature_growth = max((int(row["n_features"]) for row in rows), default=seed_feature_count) > seed_feature_count
    alternative_growth = max((int(row["n_alternatives"]) for row in rows), default=seed_alternatives) > seed_alternatives
    validation_improved = validation_delta > 0.0
    graph_alternatives = sum(len(node.alternatives) for node in best_graph.nodes.values())
    passed = bool(rows) and not duplicate_proposals and (feature_growth or alternative_growth)
    return {
        "benchmark": "competition_mutation_usefulness",
        "scope": report_scope(),
        "seed": seed,
        "dataset": metadata,
        "split_sizes": {
            "train": len(split_manifest.train_indices),
            "validation": len(split_manifest.validation_indices),
            "test": len(split_manifest.test_indices),
        },
        "reduced_test_accessed": False,
        "acceptance": {
            "policy": "validation_improvement_with_train_regression_floor",
            "min_train_improvement": min_train_improvement,
            "min_validation_improvement": min_validation_improvement,
        },
        "summary": {
            "passed": passed,
            "steps": steps,
            "seed_train_auc": seed_train_auc,
            "seed_validation_auc": seed_validation_auc,
            "best_validation_auc": best_validation_auc,
            "best_validation_delta": validation_delta,
            "validation_improved": validation_improved,
            "accepted_mutations": accepted_count,
            "unique_primitives": unique_primitives,
            "duplicate_proposals": duplicate_proposals,
            "feature_growth": feature_growth,
            "alternative_growth": alternative_growth,
            "best_candidate": best_row,
            "final_graph_alternatives": graph_alternatives,
        },
        "candidates": rows,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    rows = [
        [
            row["step"],
            status_mark(bool(row["accepted"])),
            row["target_node"],
            row["primitive"],
            "yes" if row["used_by_selected_config"] else "no",
            fmt_float(row["train_auc"]),
            fmt_float(row["validation_auc"]),
            fmt_float(row["validation_delta"]),
        ]
        for row in payload["candidates"]
    ]
    summary = payload["summary"]
    dataset = payload["dataset"]
    return "\n\n".join(
        [
            "# Competition Mutation Usefulness Benchmark",
            str(payload["scope"]),
            (
                f"Dataset: `{dataset['name']}` from `{dataset['data_dir']}`, "
                f"n=`{dataset['n_samples']}`, max_samples=`{dataset['max_samples']}`."
            ),
            "Reduced test accessed: `False`.",
            (
                f"Seed validation AUC: `{fmt_float(summary['seed_validation_auc'])}`; "
                f"best validation AUC: `{fmt_float(summary['best_validation_auc'])}`; "
                f"delta: `{fmt_float(summary['best_validation_delta'])}`; "
                f"validation improved: `{summary['validation_improved']}`; "
                f"accepted mutations: `{summary['accepted_mutations']}`."
            ),
            markdown_table(["Step", "Accepted", "Node", "Primitive", "Used", "Train AUC", "Validation AUC", "Validation Delta"], rows),
        ]
    )


def run(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seed: int = 91,
    max_samples: int | None = 1000,
    series_length: int = 160,
    steps: int = 6,
    folds: int = 3,
    max_configurations: int = 96,
) -> tuple[Path, Path]:
    payload = build_report(
        output_dir,
        data_dir=data_dir,
        seed=seed,
        max_samples=max_samples,
        series_length=series_length,
        steps=steps,
        folds=folds,
        max_configurations=max_configurations,
    )
    return write_report(output_dir, "competition_mutation_usefulness", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a validation-focused capped parquet mutation usefulness benchmark.")
    output_argument(parser)
    seed_argument(parser, default=91)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_COMPETITION_DATA_DIR)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--series-length", type=int, default=160)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-configurations", type=int, default=96)
    args = parser.parse_args(argv)
    print_report_paths(
        run(
            args.output,
            data_dir=args.data_dir,
            seed=args.seed,
            max_samples=args.max_samples,
            series_length=args.series_length,
            steps=args.steps,
            folds=args.folds,
            max_configurations=args.max_configurations,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
