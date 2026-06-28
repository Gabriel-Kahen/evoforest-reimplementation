from __future__ import annotations

from dataclasses import replace
import json

import pytest

from evoforest_arch.cli import main as cli_main
from evoforest_arch.evaluator import EvaluationResult
from evoforest_arch.graph_io import graph_from_path
from evoforest_arch.llm import LLMEngineerAgent, LLMMemorandumAgent, LLMScientistAgent, StaticLLMClient
from evoforest_arch.mutations import MutationDocument, MutationSpec
from evoforest_arch.production import (
    CV_AUC_PROMOTION_POLICY,
    PAPER_GPU_DEVICES,
    PAPER_PROFILE,
    PAPER_PROFILE_STEPS,
    ProductionConfig,
    ProductionContext,
    ProductionEvolutionRunner,
    RunState,
    export_best_graph,
    inspect_run,
    recheck_run,
)


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


def dummy_result(auc: float) -> EvaluationResult:
    return EvaluationResult(
        auc=auc,
        config={},
        feature_names=[],
        predictions=[],
        alphas=[],
        diagnostics={},
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
    assert summary["islands"]["worker_execution"] == "process_actor"
    assert summary["islands"]["devices"] == ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
    assert summary["islands"]["scientist_temperature_schedule"] == [0.35, 0.5, 0.6, 0.75]
    assert summary["islands"]["migration_count"] >= 1
    assert {item["worker_execution"] for item in summary["islands"]["items"]} == {"process_actor"}
    assert (run_dir / "migrations.jsonl").exists()
    assert (run_dir / "jobs.jsonl").exists()
    events = read_jsonl(run_dir / "events.jsonl")
    assert [event["step"] for event in events] == [1, 2, 3, 4]
    assert len({event["job_id"] for event in events}) == 4
    assert {str(event["worker_execution"]) for event in events} == {"process_actor"}
    assert all(int(event["actor_pid"]) > 0 for event in events)
    assert any(event.get("stale") for event in events)
    migrations = read_jsonl(run_dir / "migrations.jsonl")
    assert migrations
    assert {str(row["worker_execution"]) for row in migrations} == {"process_actor"}
    assert all(int(row["target_actor_pid"]) > 0 for row in migrations)
    jobs = read_jsonl(run_dir / "jobs.jsonl")
    submitted = {str(row["job_id"]): row for row in jobs if row["status"] == "submitted"}
    terminal = {str(row["job_id"]): row for row in jobs if row["status"] in {"completed", "failed", "stale", "abandoned_on_resume"}}
    assert set(submitted) == set(terminal)
    assert {str(row["device"]) for row in submitted.values()} == {"cpu:0", "cpu:1", "cpu:2", "cpu:3"}
    assert {str(row["worker_execution"]) for row in submitted.values()} == {"process_actor"}
    assert len({int(row["actor_pid"]) for row in submitted.values()}) == 4
    assert all(int(row["actor_pid"]) > 0 for row in terminal.values())
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
        assert {str(row["worker_execution"]) for row in island_jobs} == {"process_actor"}
        assert len({int(row["actor_pid"]) for row in island_jobs}) == 1
        island_migrations_path = island_dir / "migrations.jsonl"
        if island_migrations_path.exists():
            island_migrations = read_jsonl(island_migrations_path)
            assert all(row["target_island"] == island_id for row in island_migrations)
            assert {str(row["worker_execution"]) for row in island_migrations} == {"process_actor"}
            assert all(int(row["target_actor_pid"]) > 0 for row in island_migrations)

    resumed = ProductionEvolutionRunner(ProductionConfig(output_dir=run_dir, steps=1)).run(resume=True)
    assert resumed["step"] == 5
    assert resumed["event_count"] == 5
    assert resumed["islands"]["mode"] == "async"
    assert resumed["islands"]["count"] == 4
    assert resumed["islands"]["worker_execution"] == "process_actor"
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
    assert summary["islands"]["worker_execution"] == "process_actor"
    assert summary["islands"]["devices"] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert {item["device"] for item in summary["islands"]["items"]} == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}


def test_paper_profile_constructor_encodes_long_run_contract(tmp_path) -> None:
    config = ProductionConfig.paper_profile(tmp_path / "paper_profile")

    assert config.profile == PAPER_PROFILE
    assert config.steps == PAPER_PROFILE_STEPS
    assert config.islands == 4
    assert config.async_islands is True
    assert config.island_workers == 4
    assert config.island_devices == PAPER_GPU_DEVICES
    assert config.max_configurations == 64
    assert config.refine_globals is True
    assert config.refine_backend == "torch"
    assert config.promotion_policy == CV_AUC_PROMOTION_POLICY
    assert config.min_train_improvement == 0.0


def test_cli_paper_profile_writes_paper_manifest_without_long_run(tmp_path) -> None:
    output = tmp_path / "paper_profile_cli"
    result = cli_main(
        [
            "evolve",
            "--profile",
            "paper",
            "--steps",
            "0",
            "--n-series",
            "60",
            "--length",
            "70",
            "--folds",
            "2",
            "--no-refine-globals",
            "--island-devices",
            "cpu:0,cpu:1,cpu:2,cpu:3",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "paper"
    assert manifest["profile_spec"]["target_steps"] == 600
    assert manifest["profile_spec"]["promotion_metric"] == "train_cv_roc_auc"
    assert manifest["evaluator"]["max_configurations"] == 64
    assert manifest["evaluator"]["refine_backend"] == "torch"
    assert manifest["evaluator"]["refine_globals"] is False
    assert manifest["acceptance"]["policy"] == "paper_cv_auc_improvement"
    assert manifest["acceptance"]["validation_gate"] is False
    assert manifest["islands"]["count"] == 4
    assert manifest["islands"]["workers"] == 4
    assert manifest["islands"]["worker_execution"] == "process_actor"
    assert manifest["islands"]["devices"] == ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
    assert manifest["islands"]["scientist_temperature_schedule"] == [0.35, 0.5, 0.6, 0.75]
    inspected = inspect_run(output)
    assert inspected["profile"] == "paper"
    assert inspected["acceptance"]["policy"] == "paper_cv_auc_improvement"


def test_paper_profile_promotion_uses_cv_auc_not_validation_gate(tmp_path) -> None:
    runner = ProductionEvolutionRunner(
        ProductionConfig.paper_profile(
            tmp_path / "unused",
            steps=0,
            refine_globals=False,
            island_devices=("cpu:0", "cpu:1", "cpu:2", "cpu:3"),
        )
    )
    context = ProductionContext(
        run_dir=tmp_path,
        manifest={
            "acceptance": {
                "policy": "paper_cv_auc_improvement",
                "metric": "train_cv_roc_auc",
                "min_train_improvement": 0.0,
                "min_validation_improvement": 1.0,
            }
        },
        split_manifest=None,  # type: ignore[arg-type]
        splits={},
        state=RunState(
            run_id="dummy",
            step=0,
            archive_version=0,
            best_train_auc=0.5,
            best_validation_auc=0.9,
            best_config={},
            current_graph_path="current_graph.json",
            best_graph_path="best_graph.json",
            rng_state={},
        ),
        current_graph=None,  # type: ignore[arg-type]
        best_graph=None,  # type: ignore[arg-type]
        best_train_result=dummy_result(0.5),
        best_validation_result=dummy_result(0.9),
    )

    assert runner._promotes(dummy_result(0.51), dummy_result(0.1), context) is True


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
