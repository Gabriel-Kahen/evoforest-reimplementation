from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from benchmarks.common import markdown_table, output_argument, print_report_paths, quick_argument, seed_argument, write_report
from benchmarks.research_suite.baselines import RandomFeatureRidge, RawRidge
from benchmarks.research_suite.compositional_dags import BenchmarkDataset, DatasetSplit, generate_benchmark, task_catalog
from benchmarks.research_suite.evoforest_model import fit_frozen_evoforest_regressor
from benchmarks.research_suite.protocol import BudgetUsage, DatasetPartition, EvaluationProtocol, ExperimentResultRow
from evoforest_arch.seed import build_seed_graph


@dataclass(frozen=True)
class ControlledRunConfig:
    task_names: tuple[str, ...]
    seeds: tuple[int, ...]
    n_train: int = 512
    n_validation: int = 256
    n_test: int = 512
    random_features: int = 256


def run_controlled_benchmark(config: ControlledRunConfig) -> list[ExperimentResultRow]:
    rows: list[ExperimentResultRow] = []
    for task_name in config.task_names:
        for seed in config.seeds:
            dataset = generate_benchmark(
                task_name,
                seed=seed,
                n_train=config.n_train,
                n_validation=config.n_validation,
                n_test=config.n_test,
            )
            rows.extend(_run_dataset(dataset, config.random_features))
    return rows


def _run_dataset(dataset: BenchmarkDataset, random_features: int) -> list[ExperimentResultRow]:
    train_x = np.vstack((dataset.train.X, dataset.validation.X))
    train_y = np.concatenate((dataset.train.y, dataset.validation.y))
    methods: tuple[tuple[str, Callable[[], Callable[[np.ndarray], np.ndarray]]], ...] = (
        ("raw_ridge", lambda: _fit_baseline(RawRidge(), train_x, train_y)),
        (
            "random_features_ridge",
            lambda: _fit_baseline(
                RandomFeatureRidge(n_random_features=random_features, seed=dataset.seed),
                train_x,
                train_y,
            ),
        ),
        ("evoforest_seed", lambda: _fit_seed_evoforest(train_x, train_y)),
    )
    rows: list[ExperimentResultRow] = []
    for method_name, fit in methods:
        predict = fit()
        for regime, split in (
            ("interpolation", dataset.test_interpolation),
            ("extrapolation", dataset.test_extrapolation),
        ):
            protocol = _protocol(dataset, split, regime)
            token = protocol.finalize(f"{method_name}:{dataset.spec.name}:{dataset.seed}")
            evaluation = protocol.evaluate_test(token, predict)
            rows.append(
                ExperimentResultRow(
                    task_id=dataset.spec.name,
                    task_family=dataset.spec.name,
                    method=method_name,
                    seed=dataset.seed,
                    split_id=regime,
                    status="completed",
                    metrics=evaluation.metrics,
                    usage=BudgetUsage(exact_evaluations=1),
                    metadata={
                        "active_variables": list(dataset.spec.active_variables),
                        "motifs": [motif.name for motif in dataset.spec.motifs],
                    },
                )
            )
    return rows


def _fit_baseline(model: RawRidge, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    model.fit(X, y)
    return model.predict


def _fit_seed_evoforest(X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    model = fit_frozen_evoforest_regressor(build_seed_graph(), {}, {"x": X}, y)
    return lambda values: model.predict({"x": values})


def _protocol(dataset: BenchmarkDataset, test: DatasetSplit, regime: str) -> EvaluationProtocol:
    return EvaluationProtocol(
        _partition("search_train", dataset.train, f"{dataset.spec.name}:{dataset.seed}:train"),
        _partition("selection_validation", dataset.validation, f"{dataset.spec.name}:{dataset.seed}:validation"),
        _partition(f"test_{regime}", test, f"{dataset.spec.name}:{dataset.seed}:test:{regime}"),
    )


def _partition(name: str, split: DatasetSplit, prefix: str) -> DatasetPartition:
    return DatasetPartition(name, split.X, split.y, tuple(f"{prefix}:{index}" for index in range(split.y.shape[0])))


def build_report(seed: int = 17, quick: bool = False) -> dict[str, object]:
    names = tuple(task_catalog())
    config = ControlledRunConfig(
        task_names=names[:2] if quick else names,
        seeds=(seed,) if quick else (seed, seed + 1, seed + 2),
        n_train=96 if quick else 512,
        n_validation=48 if quick else 256,
        n_test=96 if quick else 512,
        random_features=48 if quick else 256,
    )
    rows = run_controlled_benchmark(config)
    return {
        "benchmark": "controlled_compositional_dags",
        "protocol": "train-fitted models with one-use sealed interpolation and extrapolation tests",
        "config": {
            "tasks": list(config.task_names),
            "seeds": list(config.seeds),
            "n_train": config.n_train,
            "n_validation": config.n_validation,
            "n_test": config.n_test,
        },
        "results": [row.to_dict() for row in rows],
    }


def markdown_report(payload: dict[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, list)
    table = [
        [
            row["task_id"],
            row["method"],
            row["split_id"],
            f"{row['metrics']['nrmse']:.4f}",
            f"{row['metrics']['r2']:.4f}",
        ]
        for row in results
    ]
    return "\n\n".join(
        [
            "# Controlled Compositional DAG Benchmark",
            str(payload["protocol"]),
            markdown_table(["Task", "Method", "Regime", "NRMSE", "R2"], table),
        ]
    )


def run(output_dir: Path, seed: int = 17, quick: bool = False) -> tuple[Path, Path]:
    payload = build_report(seed=seed, quick=quick)
    return write_report(output_dir, "controlled_compositional_dags", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run leakage-safe controlled computation-discovery baselines.")
    output_argument(parser)
    seed_argument(parser)
    quick_argument(parser)
    args = parser.parse_args(argv)
    print_report_paths(run(args.output, seed=args.seed, quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
