from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.competition_mutation_usefulness import build_report as build_mutation_usefulness_report
from benchmarks.competition_event_benchmark import build_report as build_event_benchmark_report
from benchmarks.competition_event_campaign import build_campaign_report as build_event_campaign_report
from benchmarks.competition_row_benchmark import build_report as build_row_benchmark_report
from evoforest_arch.competition import COMPETITION_DATASET_NAME, COMPETITION_ROW_DATASET_NAME, competition_data_summary, load_competition_event_dataset, load_competition_row_dataset
from evoforest_arch.production import ProductionConfig, ProductionEvolutionRunner, recheck_run
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.source_mutations import structural_break_source_mutations, validate_source_mutations
from evoforest_arch.splits import make_grouped_split_manifest, split_dataset


def test_competition_loader_maps_parquet_ids_to_graph_inputs(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=True)

    dataset = load_competition_event_dataset(data_dir, split="train", series_length=20)

    assert dataset.inputs["series"].shape == (12, 20)
    assert dataset.inputs["boundary"] == 10
    assert dataset.inputs["sample_id"].tolist() == list(range(12))
    assert dataset.y.tolist() == [1.0, 0.0] * 6
    assert dataset.metadata["official_metric_note"].startswith("This is an id-level event-detection surrogate")


def test_competition_row_loader_maps_target_rows_to_causal_graph_inputs(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=True)

    dataset = load_competition_row_dataset(data_dir, split="train", series_length=20, max_ids=6, max_rows_per_id=4)

    assert dataset.inputs["series"].shape == (24, 20)
    assert dataset.inputs["boundary"] == 10
    assert dataset.metadata["name"] == COMPETITION_ROW_DATASET_NAME
    assert dataset.metadata["n_ids"] == 6
    assert dataset.metadata["target_source"] == "target from y_train.parquet"
    assert [Path(path).name for path in dataset.metadata["read_paths"]] == ["X_train.parquet", "y_train.parquet"]
    assert dataset.metadata["official_metric_note"].startswith("This is row/time-level")
    assert dataset.inputs["sample_id"][:4].tolist() == [0, 0, 0, 0]
    assert dataset.inputs["sample_time"][:4].tolist() == [10, 13, 16, 19]
    first_series = dataset.inputs["series"][0]
    assert np.allclose(first_series[10:], first_series[10])
    assert first_series[-1] == 0.5 + np.cos(0.0)
    assert set(dataset.y.tolist()) == {0.0, 1.0}


def test_competition_row_grouped_split_keeps_ids_disjoint(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=False, n_train=18)
    dataset = load_competition_row_dataset(data_dir, split="train", series_length=20, max_rows_per_id=4)

    manifest = make_grouped_split_manifest(dataset.inputs, dataset.y, groups=dataset.inputs["sample_id"], seed=3, validation_fraction=0.2, test_fraction=0.2)
    splits = split_dataset(dataset.inputs, dataset.y, manifest)

    groups = {name: set(inputs["sample_id"].tolist()) for name, (inputs, _y) in splits.items()}
    assert manifest.group_key == "sample_id"
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])
    assert set(manifest.train_groups) == groups["train"]


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


def test_competition_row_production_uses_grouped_splits_and_ignores_reduced_test(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=False, n_train=18)
    config = row_competition_config(tmp_path, data_dir, steps=0)

    summary = ProductionEvolutionRunner(config).run()
    assert summary["step"] == 0
    assert summary["test_recheck_count"] == 0

    splits = json.loads((config.output_dir / "splits.json").read_text(encoding="utf-8"))
    assert splits["method"] == "group_stratified_random"
    assert splits["group_key"] == "sample_id"
    assert not (set(splits["train_groups"]) & set(splits["validation_groups"]))
    assert not (set(splits["train_groups"]) & set(splits["test_groups"]))

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

    row_train_only = competition_data_summary(data_dir, series_length=20, row_level=True, max_rows_per_id=2)
    row_with_reduced = competition_data_summary(data_dir, series_length=20, row_level=True, max_rows_per_id=2, include_reduced_test=True)

    assert set(row_train_only) == {"train"}
    assert set(row_with_reduced) == {"train", "reduced_test"}
    assert row_train_only["train"]["name"] == COMPETITION_ROW_DATASET_NAME


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


def test_structural_break_source_mutations_pass_repair_checks(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=False, n_train=24)
    dataset = load_competition_event_dataset(data_dir, split="train", series_length=20)

    checks = validate_source_mutations(build_seed_graph(), structural_break_source_mutations(), dataset.inputs)

    assert checks
    assert all(check.passed for check in checks)
    assert all(check.n_features > 0 for check in checks)


def test_competition_event_benchmark_reports_source_candidates_and_no_reduced_access(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=False, n_train=30)

    report = build_event_benchmark_report(
        tmp_path,
        data_dir=data_dir,
        seed=13,
        series_length=20,
        max_samples=None,
        steps=4,
        folds=2,
        max_configurations=4,
    )

    assert report["dataset"]["name"] == COMPETITION_DATASET_NAME
    assert report["split"]["group_key"] == "sample_id"
    assert report["split"]["audit"]["no_group_overlap"] is True
    assert report["source_mutations"]["passed_repair_checks"] == report["source_mutations"]["templates"]
    assert any(row["source_backed"] for row in report["evolution"]["candidates"])
    assert report["reduced_test_access"]["accessed"] is False
    assert all("reduced" not in Path(path).name for path in report["reduced_test_access"]["read_paths"])
    assert "validation_auc" in report["baseline"]
    assert "best" in report["ensembles"]


def test_competition_event_campaign_aggregates_multiple_seeds(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=False, n_train=30)

    report = build_event_campaign_report(
        tmp_path / "campaign",
        data_dir=data_dir,
        seeds=(13, 17),
        series_length=20,
        max_samples=None,
        steps=4,
        folds=2,
        max_configurations=4,
    )

    assert report["summary"]["seed_count"] == 2
    assert report["summary"]["any_reduced_test_accessed"] is False
    assert len(report["seeds_detail"]) == 2
    assert all(row["source_backed_candidates"] >= 1 for row in report["seeds_detail"])
    assert all(row["group_overlap"] == {"train_validation": 0, "train_test": 0, "validation_test": 0} for row in report["seeds_detail"])


def test_competition_row_benchmark_reports_grouped_holdout_and_no_reduced_access(tmp_path) -> None:
    data_dir = write_competition_bundle(tmp_path, include_reduced=False, n_train=24)

    report = build_row_benchmark_report(
        tmp_path,
        data_dir=data_dir,
        seed=11,
        series_length=20,
        max_ids=None,
        max_rows_per_id=4,
        steps=2,
        folds=2,
        max_configurations=4,
    )

    assert report["dataset"]["name"] == COMPETITION_ROW_DATASET_NAME
    assert report["split"]["group_key"] == "sample_id"
    assert report["split"]["audit"]["no_group_overlap"] is True
    assert report["reduced_test_access"]["accessed"] is False
    assert all("reduced" not in Path(path).name for path in report["reduced_test_access"]["read_paths"])
    assert "validation_auc" in report["baseline"]
    assert "validation_delta_vs_baseline" in report["evolved_graph"]


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


def row_competition_config(tmp_path: Path, data_dir: Path, *, steps: int) -> ProductionConfig:
    return ProductionConfig(
        output_dir=tmp_path / f"row_run_{steps}",
        steps=steps,
        seed=7,
        dataset_name=COMPETITION_ROW_DATASET_NAME,
        data_dir=data_dir,
        competition_series_length=20,
        competition_max_rows_per_id=4,
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
    write_y(data_dir / "y_train.parquet", ids=range(n_train))
    if include_reduced:
        write_x(data_dir / "X_test.reduced.parquet", ids=range(100, 100 + n_reduced))
        write_index(data_dir / "y_test_index.reduced.parquet", ids=range(100, 100 + n_reduced))
        write_y(data_dir / "y_test.reduced.parquet", ids=range(100, 100 + n_reduced))
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


def write_y(path: Path, *, ids: range) -> None:
    rows = {"target": [], "id": [], "time": []}
    for id_ in ids:
        positive = id_ % 2 == 0
        for time in range(10, 20):
            rows["target"].append(1 if positive and time >= 14 else 0)
            rows["id"].append(id_)
            rows["time"].append(time)
    pq.write_table(pa.table(rows), path)
