from __future__ import annotations

import json

import numpy as np

from benchmarks.research_suite import curate_srsd
from benchmarks.research_suite.external_datasets import load_regression_dataset


def test_srsd_curator_preserves_source_splits_and_equation_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(curate_srsd, "SELECTED_EASY_TASKS", ("task",))
    source = tmp_path / "source"
    for split in ("train", "val", "test"):
        (source / split).mkdir(parents=True, exist_ok=True)
        np.savetxt(source / split / "task.txt", np.array([[1.0, 2.0], [2.0, 4.0]]))
    (source / "supp_info.json").write_text(json.dumps({"task": {"sympy_eq_str": "2*x0"}}), encoding="utf-8")

    manifests = curate_srsd.curate(source, tmp_path / "out")
    dataset = load_regression_dataset(manifests[0])

    assert len(dataset.train.y) == len(dataset.validation.y) == len(dataset.test.y) == 2
    assert dataset.manifest.metadata["true_equation"] == "2*x0"
