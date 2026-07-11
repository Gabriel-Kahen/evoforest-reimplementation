from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.common import markdown_table, output_argument, print_report_paths, write_report
from benchmarks.research_suite.baselines import RandomFeatureRidge, RawRidge
from benchmarks.research_suite.evoforest_model import fit_frozen_evoforest_regressor
from benchmarks.research_suite.external_datasets import FrozenRegressionDataset, load_regression_dataset
from benchmarks.research_suite.protocol import BudgetUsage, DatasetPartition, EvaluationProtocol, ExperimentResultRow
from evoforest_arch.seed import build_seed_graph


def evaluate_external_dataset(dataset: FrozenRegressionDataset, *, seed: int = 0) -> list[ExperimentResultRow]:
    train_x = np.vstack((dataset.train.X, dataset.validation.X))
    train_y = np.concatenate((dataset.train.y, dataset.validation.y))

    raw = RawRidge().fit(train_x, train_y)
    random = RandomFeatureRidge(seed=seed).fit(train_x, train_y)
    evoforest = fit_frozen_evoforest_regressor(build_seed_graph(), {}, {"x": train_x}, train_y)
    methods = {
        "raw_ridge": raw.predict,
        "random_features_ridge": random.predict,
        "evoforest_seed": lambda X: evoforest.predict({"x": X}),
    }

    rows: list[ExperimentResultRow] = []
    for method, predict in methods.items():
        protocol = EvaluationProtocol(
            _partition(dataset, "search_train"),
            _partition(dataset, "selection_validation"),
            _partition(dataset, "sealed_test"),
        )
        token = protocol.finalize(f"{dataset.manifest.dataset_id}:{method}:{seed}")
        result = protocol.evaluate_test(token, predict)
        rows.append(
            ExperimentResultRow(
                task_id=dataset.manifest.dataset_id,
                task_family=str(dataset.manifest.metadata.get("family", "external_regression")),
                method=method,
                seed=seed,
                split_id="sealed_test",
                status="completed",
                metrics=result.metrics,
                usage=BudgetUsage(exact_evaluations=1),
                metadata={"manifest": str(dataset.manifest.manifest_path)},
            )
        )
    return rows


def _partition(dataset: FrozenRegressionDataset, name: str) -> DatasetPartition:
    source = {
        "search_train": dataset.train,
        "selection_validation": dataset.validation,
        "sealed_test": dataset.test,
    }[name]
    ids = tuple(f"{dataset.manifest.dataset_id}:{int(index)}" for index in source.indices)
    return DatasetPartition(name, source.X, source.y, ids)


def build_report(manifests: list[Path], seed: int = 0) -> dict[str, object]:
    rows = [
        row
        for manifest in manifests
        for row in evaluate_external_dataset(load_regression_dataset(manifest), seed=seed)
    ]
    return {
        "benchmark": "external_regression_boundaries",
        "manifests": [str(path) for path in manifests],
        "results": [row.to_dict() for row in rows],
    }


def markdown_report(payload: dict[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, list)
    rows = [
        [row["task_id"], row["method"], f"{row['metrics']['nrmse']:.4f}", f"{row['metrics']['r2']:.4f}"]
        for row in results
    ]
    return "\n\n".join(
        [
            "# External Regression Boundary Benchmark",
            "Every row uses frozen manifest splits and a one-use sealed test.",
            markdown_table(["Task", "Method", "NRMSE", "R2"], rows),
        ]
    )


def run(output_dir: Path, manifests: list[Path], seed: int = 0) -> tuple[Path, Path]:
    payload = build_report(manifests, seed)
    return write_report(output_dir, "external_regression_boundaries", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen local symbolic/real regression manifests.")
    output_argument(parser)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print_report_paths(run(args.output, args.manifest, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
