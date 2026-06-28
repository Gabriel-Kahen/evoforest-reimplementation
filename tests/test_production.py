from __future__ import annotations

from dataclasses import replace
import json

import pytest

from evoforest_arch.graph_io import graph_from_path
from evoforest_arch.llm import LLMEngineerAgent, LLMMemorandumAgent, LLMScientistAgent, StaticLLMClient
from evoforest_arch.mutations import MutationDocument, MutationSpec
from evoforest_arch.production import ProductionConfig, ProductionEvolutionRunner, export_best_graph, inspect_run, recheck_run


def small_config(tmp_path, *, steps: int = 1, seed: int = 41, **kwargs: object) -> ProductionConfig:
    production_kwargs = {"islands": 1, "async_islands": False, **kwargs}
    return ProductionConfig(
        output_dir=tmp_path / "production_run",
        steps=steps,
        seed=seed,
        n_series=60,
        length=70,
        folds=2,
        max_configurations=4,
        irls_steps=1,
        **production_kwargs,
    )


def read_jsonl(path) -> list[dict[str, object]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def paper_memorandum(label: str) -> str:
    return "\n".join(
        [
            "[OUTCOME HISTORY]",
            f"- {label} outcome.",
            "[STATE]",
            f"- {label} state.",
            "[WHAT WORKS]",
            "- Valid graph candidates are retained.",
            "[WHAT FAILED]",
            "- Invalid candidates are recorded.",
            "[ERROR LOG]",
            "- No runtime errors recorded.",
        ]
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


def test_production_llm_repairs_failed_source_and_uses_memorandum_agent(tmp_path) -> None:
    config = small_config(tmp_path, steps=1, seed=47, allow_source_mutations=True)
    bad_document = MutationDocument(
        hypotheses=("Try a source alternative that fails at runtime.",),
        rationale="Exercise production repair feedback.",
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="source",
                alternative_id="production_bad_source_output",
                parents=("segment_stats",),
                source="lambda ctx, values: values['missing_parent']",
                description="Intentional production runtime failure.",
            ),
        ),
    )
    repair_document = MutationDocument(
        hypotheses=("Repair with a registry-backed spectral alternative.",),
        rationale="Use a known primitive after the source failure.",
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="shape_stats",
                primitive="spectral_basic",
                alternative_id="production_spectral_after_error",
                parents=("series",),
                description="Valid production repair after runtime failure.",
            ),
        ),
    )
    client = StaticLLMClient(
        (
            paper_memorandum("initial"),
            "Hypothesis: Add a failing source output.\nRationale: Exercise failures.\nExpected Improvement: none.\nRisk Mode: Risky.",
            bad_document.to_yaml(),
            "Hypothesis: Use a safer shape_stats primitive.\nRationale: Prior source failed.\nExpected Improvement: recover valid search.\nRisk Mode: Conservative.",
            repair_document.to_yaml(),
            paper_memorandum("final"),
        )
    )

    ProductionEvolutionRunner(
        config,
        scientist=LLMScientistAgent(client),
        engineer=LLMEngineerAgent(client, allow_source=True),
        memorandum_agent=LLMMemorandumAgent(client),
    ).run()

    run_dir = config.output_dir
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0].get("failed") is not True
    assert events[0]["mutation"]["add"][0]["alternative_id"] == "production_spectral_after_error"
    assert (run_dir / "mutations" / "step_0001_repair_01.yaml").exists()
    prompt_names = {path.name for path in (run_dir / "prompts").glob("*.md")}
    assert any("memorandum" in name for name in prompt_names)
    assert "KeyError" in str(client.requests[4]["user_prompt"])


def test_production_async_islands_write_durable_artifacts_and_resume(tmp_path) -> None:
    config = small_config(
        tmp_path,
        steps=4,
        seed=4,
        islands=4,
        async_islands=True,
        island_workers=4,
        island_devices=("cpu:0", "cpu:1", "cpu:2", "cpu:3"),
        migration_interval=1,
        refine_globals=False,
        min_train_improvement=-1.0,
        min_validation_improvement=-1.0,
    )
    summary = ProductionEvolutionRunner(config).run()
    run_dir = config.output_dir

    assert summary["step"] == 4
    assert summary["event_count"] == 4
    assert summary["test_recheck_count"] == 0
    assert summary["islands"]["mode"] == "async"
    assert summary["islands"]["topology"] == "paper_dedicated_gpu"
    assert summary["islands"]["count"] == 4
    assert summary["islands"]["workers"] == 4
    assert summary["islands"]["devices"] == ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
    assert summary["islands"]["scientist_temperature_schedule"] == [0.35, 0.5, 0.6, 0.75]
    assert summary["islands"]["migration_count"] >= 1
    assert (run_dir / "migrations.jsonl").exists()
    assert (run_dir / "jobs.jsonl").exists()
    events = read_jsonl(run_dir / "events.jsonl")
    assert [event["step"] for event in events] == [1, 2, 3, 4]
    assert len({event["job_id"] for event in events}) == 4
    assert any(event.get("stale") for event in events)
    jobs = read_jsonl(run_dir / "jobs.jsonl")
    submitted = {str(row["job_id"]): row for row in jobs if row["status"] == "submitted"}
    terminal = {str(row["job_id"]): row for row in jobs if row["status"] in {"completed", "failed", "stale", "abandoned_on_resume"}}
    assert set(submitted) == set(terminal)
    assert {str(row["device"]) for row in submitted.values()} == {"cpu:0", "cpu:1", "cpu:2", "cpu:3"}
    for island_id in range(4):
        island_dir = run_dir / "islands" / f"island_{island_id}"
        assert (island_dir / "state.json").exists()
        assert (island_dir / "current_graph.json").exists()
        assert (island_dir / "best_graph.json").exists()
        assert (island_dir / "checkpoint.json").exists()
        assert (island_dir / "memorandum.md").exists()
        assert (island_dir / "events.jsonl").exists()
        assert (island_dir / "archive" / "index.jsonl").exists()
        island_jobs = read_jsonl(island_dir / "jobs.jsonl")
        assert all(row["island"] == island_id for row in island_jobs)
        assert all(row["device"] == f"cpu:{island_id}" for row in island_jobs)

    resumed = ProductionEvolutionRunner(ProductionConfig(output_dir=run_dir, steps=1)).run(resume=True)
    assert resumed["step"] == 5
    assert resumed["event_count"] == 5
    assert resumed["islands"]["mode"] == "async"
    assert resumed["islands"]["count"] == 4
    assert resumed["islands"]["devices"] == ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
    assert len((run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    assert len((run_dir / "migrations.jsonl").read_text(encoding="utf-8").splitlines()) >= summary["islands"]["migration_count"]


def test_production_defaults_to_paper_four_gpu_island_topology(tmp_path) -> None:
    config = ProductionConfig(
        output_dir=tmp_path / "paper_default_run",
        steps=0,
        seed=5,
        n_series=60,
        length=70,
        folds=2,
        max_configurations=4,
        irls_steps=1,
        refine_globals=False,
    )
    summary = ProductionEvolutionRunner(config).run()

    assert summary["islands"]["mode"] == "async"
    assert summary["islands"]["topology"] == "paper_dedicated_gpu"
    assert summary["islands"]["count"] == 4
    assert summary["islands"]["workers"] == 4
    assert summary["islands"]["devices"] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert {item["device"] for item in summary["islands"]["items"]} == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}


def test_production_islands_require_async_mode(tmp_path) -> None:
    config = small_config(tmp_path, steps=1, islands=2, async_islands=False)

    with pytest.raises(ValueError, match="async_islands=True"):
        ProductionEvolutionRunner(config).run()


def test_production_island_native_topology_requires_four_dedicated_devices(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly 4 islands"):
        ProductionEvolutionRunner(small_config(tmp_path, steps=1, islands=2, async_islands=True)).run()

    with pytest.raises(ValueError, match="unique dedicated devices"):
        ProductionEvolutionRunner(
            small_config(
                tmp_path,
                steps=1,
                islands=4,
                async_islands=True,
                island_workers=4,
                island_devices=("cpu:0", "cpu:0", "cpu:2", "cpu:3"),
            )
        ).run()
