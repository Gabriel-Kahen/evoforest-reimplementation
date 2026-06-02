from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.competition_mutation_usefulness import build_report as build_mutation_usefulness_report
from evoforest_arch.competition import COMPETITION_DATASET_NAME, competition_data_summary, load_competition_event_dataset
from evoforest_arch.production import ProductionConfig, ProductionEvolutionRunner, recheck_run


def test_competition_loader_maps_parquet_ids_to_graph_inputs(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=True)

    dataset = load_competition_event_dataset(data_dir, split="train", series_length=20)

    assert dataset.inputs["series"].shape == (12, 20)
    assert dataset.inputs["boundary"] == 10
    assert dataset.inputs["sample_id"].tolist() == list(range(12))
    assert dataset.y.tolist() == [1.0, 0.0] * 6
    assert dataset.metadata["official_metric_note"].startswith("This is an id-level event-detection surrogate")


def test_competition_production_ignores_reduced_test_until_explicit_recheck(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=False)
    config = competition_config(tmp_path, data_dir, steps=1)

    summary = ProductionEvolutionRunner(config).run()
    assert summary["step"] == 1
    assert summary["test_recheck_count"] == 0

    recheck = recheck_run(config.output_dir)
    assert set(recheck["splits"]) == {"train", "validation"}
    state = json.loads((config.output_dir / "state.json").read_text(encoding="utf-8"))
    assert state["test_recheck_count"] == 0


def test_data_summary_is_train_only_unless_reduced_test_is_explicit(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=True)

    train_only = competition_data_summary(data_dir, series_length=20)
    with_reduced = competition_data_summary(data_dir, series_length=20, include_reduced_test=True)

    assert set(train_only) == {"train"}
    assert set(with_reduced) == {"train", "reduced_test"}


def test_competition_include_test_recheck_reads_reduced_test_explicitly(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=True)
    config = competition_config(tmp_path, data_dir, steps=0)
    ProductionEvolutionRunner(config).run()

    recheck = recheck_run(config.output_dir, include_test=True)

    assert {"train", "validation", "test", "reduced_test"} <= set(recheck["splits"])
    assert recheck["splits"]["reduced_test"]["n_samples"] == 4
    state = json.loads((config.output_dir / "state.json").read_text(encoding="utf-8"))
    assert state["test_recheck_count"] == 1


def test_competition_mutation_usefulness_benchmark_uses_diverse_train_only_candidates(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=True, n_train=48, n_reduced=8)

    report = build_mutation_usefulness_report(
        tmp_path,
        data_dir=data_dir,
        seed=9,
        max_samples=None,
        series_length=20,
        steps=4,
        folds=2,
        max_configurations=4,
    )

    primitives = [row["primitive"] for row in report["candidates"]]
    assert report["reduced_test_accessed"] is False
    assert len(primitives) == len(set(primitives))
    assert "reduced_test" not in report["dataset"]["split"]
    assert report["summary"]["duplicate_proposals"] is False


def competition_config(tmp_path: Path, data_dir: Path, *, steps: int) -> ProductionConfig:
    return ProductionConfig(
        output_dir=tmp_path / f"run_{steps}",
        steps=steps,
        seed=7,
        dataset_name=COMPETITION_DATASET_NAME,
        data_dir=data_dir,
        competition_series_length=20,
        folds=2,
        max_configurations=3,
        irls_steps=0,
        validation_fraction=0.25,
        test_fraction=0.25,
    )


def write_competition_bundle(tmp_path: Path, *, include_reduced: bool, n_train: int = 12, n_reduced: int = 4) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_x(data_dir / "X_train.parquet", ids=range(n_train))
    write_index(data_dir / "y_train_index.parquet", ids=range(n_train))
    if include_reduced:
        write_x(data_dir / "X_test.reduced.parquet", ids=range(100, 100 + n_reduced))
        write_index(data_dir / "y_test_index.reduced.parquet", ids=range(100, 100 + n_reduced))
    return data_dir


def write_x(path: Path, *, ids: range) -> None:
    rows = {"value": [], "period": [], "id": [], "time": []}
    for id_ in ids:
        label_boost = 0.5 if id_ % 2 == 0 else -0.5
        for time in range(10):
            rows["value"].append(float(np.sin(time / 3.0) + id_ * 0.01))
            rows["period"].append(1)
            rows["id"].append(id_)
            rows["time"].append(time)
        for offset, time in enumerate(range(10, 20)):
            rows["value"].append(float(label_boost + np.cos(offset / 3.0) + id_ * 0.01))
            rows["period"].append(2)
            rows["id"].append(id_)
            rows["time"].append(time)
    pq.write_table(pa.table(rows), path)


def write_index(path: Path, *, ids: range) -> None:
    rows = {"tau_index": [], "tau": [], "id": []}
    for id_ in ids:
        positive = id_ % 2 == 0
        rows["tau_index"].append(2 if positive else -1)
        rows["tau"].append(14 if positive else -1)
        rows["id"].append(id_)
    pq.write_table(pa.table(rows), path)
