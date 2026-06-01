from __future__ import annotations

from dataclasses import replace
import json

from evoforest_arch.graph_io import graph_from_path
from evoforest_arch.production import ProductionConfig, ProductionEvolutionRunner, export_best_graph, inspect_run, recheck_run


def small_config(tmp_path, *, steps: int = 1, seed: int = 41, **kwargs: object) -> ProductionConfig:
    return ProductionConfig(
        output_dir=tmp_path / "production_run",
        steps=steps,
        seed=seed,
        n_series=60,
        length=70,
        folds=2,
        max_configurations=4,
        irls_steps=1,
        **kwargs,
    )


def test_production_evolve_writes_fixed_splits_and_resumes(tmp_path) -> None:
    config = small_config(tmp_path, steps=1, seed=41)
    summary = ProductionEvolutionRunner(config).run()

    run_dir = config.output_dir
    assert summary["step"] == 1
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "splits.json").exists()
    assert (run_dir / "state.json").exists()
    assert (run_dir / "best_graph.json").exists()
    assert (run_dir / "current_graph.json").exists()
    assert (run_dir / "archive" / "index.jsonl").exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    assert manifest["dataset_fingerprint"] == splits["dataset_fingerprint"]
    assert manifest["test_policy"].startswith("test split is not evaluated")
    train = set(splits["train_indices"])
    validation = set(splits["validation_indices"])
    test = set(splits["test_indices"])
    assert train
    assert validation
    assert test
    assert not (train & validation)
    assert not (train & test)
    assert not (validation & test)
    assert train | validation | test == set(range(splits["n_samples"]))

    resumed = ProductionEvolutionRunner(replace(config, steps=1)).run(resume=True)
    assert resumed["step"] == 2
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 2


def test_export_inspect_and_recheck_keep_test_explicit(tmp_path) -> None:
    config = small_config(tmp_path, steps=1, seed=43)
    ProductionEvolutionRunner(config).run()
    run_dir = config.output_dir

    inspected = inspect_run(run_dir)
    assert inspected["split_sizes"]["test"] > 0
    assert inspected["test_recheck_count"] == 0

    exported_path = export_best_graph(run_dir, tmp_path / "best_export.json")
    exported_graph = graph_from_path(exported_path)
    assert exported_graph.output_nodes() == ["output"]

    validation_recheck = recheck_run(run_dir)
    assert set(validation_recheck["splits"]) == {"train", "validation"}
    state_after_validation = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state_after_validation["test_recheck_count"] == 0

    test_recheck = recheck_run(run_dir, include_test=True)
    assert "test" in test_recheck["splits"]
    state_after_test = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state_after_test["test_recheck_count"] == 1


def test_strict_production_promotion_does_not_archive_ties_or_small_deltas(tmp_path) -> None:
    config = small_config(
        tmp_path,
        steps=1,
        seed=45,
        min_train_improvement=10.0,
        min_validation_improvement=10.0,
    )
    ProductionEvolutionRunner(config).run()
    run_dir = config.output_dir

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    archive_rows = (run_dir / "archive" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))

    assert events[0]["accepted"] is False
    assert len(archive_rows) == 1
    assert state["archive_version"] == 0
