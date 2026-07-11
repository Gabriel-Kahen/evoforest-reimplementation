from __future__ import annotations

import json

import numpy as np

from benchmarks.research_suite.external_benchmark import build_report


def test_external_benchmark_uses_frozen_manifest_split(tmp_path) -> None:
    rng = np.random.default_rng(71)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] * X[:, 1] + 0.2 * X[:, 2]
    np.savez(tmp_path / "task.npz", X=X, y=y)
    manifest = {
        "version": 1,
        "dataset_id": "local/composition",
        "data": "task.npz",
        "splits": {
            "train": list(range(30)),
            "validation": list(range(30, 45)),
            "test": list(range(45, 60)),
        },
        "metadata": {"family": "test"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = build_report([path], seed=71)

    assert len(report["results"]) == 3
    assert {row["method"] for row in report["results"]} == {
        "raw_ridge",
        "random_features_ridge",
        "evoforest_seed",
    }
