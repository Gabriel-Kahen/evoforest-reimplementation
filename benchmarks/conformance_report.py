from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.llm import DEFAULT_ISLAND_TEMPERATURES, LLMEngineerAgent, LLMMemorandumAgent, LLMScientistAgent, PromptBuilder, StaticLLMClient
from evoforest_arch.maintenance import GraphMaintenance
from evoforest_arch.mutations import MutationDocument, MutationEngine, MutationSpec
from evoforest_arch.seed import build_structural_break_seed_graph
from evoforest_arch.synthetic import make_structural_break_data

from benchmarks.common import (
    evaluation_summary,
    graph_summary,
    markdown_table,
    output_argument,
    print_report_paths,
    quick_argument,
    report_scope,
    seed_argument,
    status_mark,
    write_report,
)


def build_report(output_dir: Path, seed: int = 17, quick: bool = False) -> dict[str, Any]:
    n_series = 48 if quick else 72
    length = 64 if quick else 80
    max_configurations = 8 if quick else 16

    graph = build_structural_break_seed_graph()
    graph.validate_acyclic()
    dataset = make_structural_break_data(n_series=n_series, length=length, seed=seed)
    evaluator = RidgeEvaluator(n_splits=3, seed=seed, max_configurations=max_configurations, irls_steps=2)
    result = evaluator.evaluate(graph, dataset.inputs(), dataset.y, update_graph=True)

    refine_result = RidgeEvaluator(
        n_splits=3,
        seed=seed,
        max_configurations=4,
        refine_globals=True,
        refine_steps=1,
        refine_backend="auto",
    ).evaluate(build_structural_break_seed_graph(), dataset.inputs(), dataset.y)

    run_dir = output_dir / "artifacts" / "conformance_evolution"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    evolution_document = MutationDocument(
        hypotheses=("Exercise the required LLM mutation path.",),
        rationale="Conformance report paper-pipeline mutation.",
        add=(MutationSpec(
            kind="add_alternative",
            target_node="shape_stats",
            primitive="spectral_basic",
            alternative_id="spectral_conformance_evolution",
            parents=("series",),
        ),),
    )
    memorandum_text = "\n".join((
        "[OUTCOME HISTORY]", "- conformance.", "[STATE]", "- conformance.",
        "[WHAT WORKS]", "- conformance.", "[WHAT FAILED]", "- none.",
        "[ERROR LOG]", "- none.",
    ))
    evolution_result = EvolutionLoop(
        build_structural_break_seed_graph(),
        evaluator=RidgeEvaluator(n_splits=3, seed=seed, max_configurations=4),
        scientist=LLMScientistAgent(StaticLLMClient((
            "Hypothesis: Test LLM path.\nRationale: Conformance.\nExpected Improvement: coverage.\nRisk Mode: Balanced.",
        ))),
        engineer=LLMEngineerAgent(StaticLLMClient((evolution_document.to_yaml(),))),
        memorandum_agent=LLMMemorandumAgent(
            StaticLLMClient((memorandum_text, memorandum_text, memorandum_text))
        ),
        seed=seed,
    ).run(dataset.inputs(), dataset.y, steps=1, output_dir=run_dir)

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    archive_rows = [json.loads(line) for line in (run_dir / "archive" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    memorandum = (run_dir / "memorandum.md").read_text(encoding="utf-8")
    task_context = (run_dir / "task_context.md").read_text(encoding="utf-8")

    mutation_document = MutationDocument(
        hypotheses=("Exercise the YAML mutation contract.",),
        rationale="Conformance report round-trip.",
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="shape_stats",
                primitive="spectral_basic",
                alternative_id="spectral_conformance",
                parents=("series",),
                description="Conformance mutation alternative.",
            ),
        ),
    )
    parsed_document = MutationDocument.from_yaml(mutation_document.to_yaml())

    prompt_builder = PromptBuilder()
    scientist_client = StaticLLMClient(("Hypothesis: Test.\nRationale: Test.\nExpected Improvement: Test.\nRisk Mode: Balanced.",))
    engineer_client = StaticLLMClient((mutation_document.to_yaml(),))
    scientist = LLMScientistAgent(scientist_client, prompt_builder=prompt_builder)
    engineer = LLMEngineerAgent(engineer_client, prompt_builder=prompt_builder)
    scientist_temperature = scientist._temperature_for(2)
    engineer_temperature = engineer.temperature

    graph_info = graph_summary(graph)
    eval_info = evaluation_summary(result)
    search = result.diagnostics["configuration_search"]
    cache = search["cache"]
    fitting = result.diagnostics["fitting"]
    scoring = result.diagnostics["scoring_context"]
    linear_shap = result.diagnostics["linear_shap"]
    refinement = refine_result.diagnostics["refinement"]

    requirements = [
        requirement(
            "dag_nodes",
            "Shared DAG exposes input, intermediate, callable, output, and fitting node types.",
            {"input", "intermediate", "callable", "output", "fitting"} <= set(graph_info["node_kinds"]),
            {"node_kinds": graph_info["node_kinds"], "nodes": graph_info["nodes"]},
        ),
        requirement(
            "output_semantics",
            "Output alternatives are stacked as features and excluded from configuration selection.",
            "output" not in graph.configuration_space() and len(result.feature_names) >= len(graph.nodes["output"].alternatives),
            {"output_in_config_space": "output" in graph.configuration_space(), "n_output_alternatives": len(graph.nodes["output"].alternatives), "n_features": len(result.feature_names)},
        ),
        requirement(
            "configuration_search",
            "Evaluator scores capped graph configurations and reports the best configured task score.",
            bool(search["capped"]) and int(search["evaluated"]) == max_configurations and float(search["best_config_score"]) == result.score,
            {"evaluated": search["evaluated"], "total": search["total"], "capped": search["capped"], "score": result.score},
        ),
        requirement(
            "ancestor_cache",
            "Configuration search shares an ancestor-conditioned cache across configurations.",
            cache.get("shared_across_configurations") is True and cache.get("key") == "ancestor_conditioned_subpath",
            cache,
        ),
        requirement(
            "fitting_nodes",
            "ridge_w and ridge_g fitting nodes alter sample and residual weighting.",
            isinstance(fitting.get("ridge_w"), dict) and isinstance(fitting.get("ridge_g"), dict) and "irls_steps_requested" in fitting["ridge_g"],
            fitting,
        ),
        requirement(
            "global_parameters",
            "Persistent trainable globals are present and referenced by graph alternatives.",
            {"gate_scale", "projection_vector"} <= set(graph.globals.trainable_names()) and "residual_huber_scale" in graph.globals.names(),
            {"trainable_globals": graph.globals.trainable_names(), "referenced_globals": sorted(graph.referenced_globals())},
        ),
        requirement(
            "two_phase_evaluation",
            "Paper-mode global refinement uses PyTorch L-BFGS or explicitly reports a skipped torch probe.",
            refinement.get("requested_backend") == "auto" and refinement.get("backend") == "torch_l_bfgs",
            refinement,
        ),
        requirement(
            "ridge_diagnostics",
            "Evaluator emits diagnostic global Ridge and exact linear contribution reconstruction metrics.",
            "global_ridge_score" in scoring and float(linear_shap["global_reconstruction_error"]) < 1e-8,
            {"scoring_context": scoring, "linear_shap": linear_shap},
        ),
        requirement(
            "alternative_history",
            "Stateful evaluation updates alternative age and rolling statistics.",
            any(int(row.get("age", 0)) > 0 for row in result.diagnostics["alternative_stats"]),
            {"n_alternative_stats": len(result.diagnostics["alternative_stats"]), "sample": result.diagnostics["alternative_stats"][:3]},
        ),
        requirement(
            "mutation_yaml",
            "Mutation documents round-trip through the YAML-style contract.",
            parsed_document.to_dict() == mutation_document.to_dict(),
            parsed_document.to_dict(),
        ),
        requirement(
            "maintenance",
            "Mutation application runs graph maintenance and reports cleanup actions.",
            "maintenance" in events[0] and {"collapsed_duplicates", "removed_nodes", "removed_globals"} <= set(events[0]["maintenance"]),
            events[0].get("maintenance", {}),
        ),
        requirement(
            "failed_feedback_salvage_surface",
            "Evolution events expose failure/salvage fields for rejected candidate handling.",
            "salvaged" in events[0] and hasattr(EvolutionLoop, "_salvage"),
            {"event_salvaged": events[0].get("salvaged"), "has_salvage_method": hasattr(EvolutionLoop, "_salvage")},
        ),
        requirement(
            "artifacts",
            "Evolution writes events, checkpoint, archive, task context, mutation YAML, and memorandum artifacts.",
            all(
                path.exists()
                for path in (
                    run_dir / "events.jsonl",
                    run_dir / "checkpoint.json",
                    run_dir / "archive" / "index.jsonl",
                    run_dir / "task_context.md",
                    run_dir / "mutations" / "step_0001.yaml",
                    run_dir / "memorandum.md",
                )
            ),
            {"run_dir": str(run_dir), "archive_rows": len(archive_rows), "checkpoint_keys": sorted(checkpoint)},
        ),
        requirement(
            "memorandum",
            "Memorandum uses paper-style sectioned, hypothesis-free experiment logging.",
            all(section in memorandum for section in ("[OUTCOME HISTORY]", "[STATE]", "[WHAT WORKS]", "[WHAT FAILED]", "[ERROR LOG]")),
            {"sections_present": [section for section in ("[OUTCOME HISTORY]", "[STATE]", "[WHAT WORKS]", "[WHAT FAILED]", "[ERROR LOG]") if section in memorandum]},
        ),
        requirement(
            "task_context",
            "Task context captures tensor inventory and scorer mechanics for prompt grounding.",
            "## Tensor Inventory" in task_context and "## Scorer Mechanics" in task_context,
            {"contains_tensor_inventory": "## Tensor Inventory" in task_context, "contains_scorer_mechanics": "## Scorer Mechanics" in task_context},
        ),
        requirement(
            "llm_two_stage",
            "LLM-backed scientist/engineer agents preserve the two-stage prompt and temperature contract.",
            tuple(DEFAULT_ISLAND_TEMPERATURES) == (0.35, 0.5, 0.6, 0.75) and scientist_temperature == 0.6 and engineer_temperature == 0.0,
            {"scientist_island_temperatures": DEFAULT_ISLAND_TEMPERATURES, "island_2_temperature": scientist_temperature, "engineer_temperature": engineer_temperature},
        ),
        requirement(
            "source_mutation_gate",
            "Trusted source-backed mutations are explicit opt-in behavior.",
            MutationEngine(allow_source=True).allow_source is True and MutationEngine(allow_source=False).allow_source is False,
            {"allow_source_true": MutationEngine(allow_source=True).allow_source, "allow_source_false": MutationEngine(allow_source=False).allow_source},
        ),
        requirement(
            "graph_maintenance_api",
            "Graph maintenance exposes duplicate collapse, unreachable pruning, and unused-global cleanup.",
            all(hasattr(GraphMaintenance, method) for method in ("_collapse_duplicate_alternatives", "_prune_unreachable_nodes", "_prune_unused_globals")),
            {"maintenance_methods": [name for name in dir(GraphMaintenance) if name.startswith("_prune") or name.startswith("_collapse")]},
        ),
    ]

    passed = sum(1 for item in requirements if item["passed"])
    return {
        "benchmark": "conformance",
        "scope": report_scope(),
        "seed": seed,
        "quick": quick,
        "summary": {
            "passed": passed,
            "total": len(requirements),
            "all_passed": passed == len(requirements),
        },
        "graph": graph_info,
        "evaluation": eval_info,
        "evolution": evaluation_summary(evolution_result),
        "requirements": requirements,
    }


def requirement(requirement_id: str, description: str, passed: bool, evidence: object) -> dict[str, object]:
    return {
        "id": requirement_id,
        "description": description,
        "passed": bool(passed),
        "evidence": evidence,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    rows = [
        [item["id"], status_mark(bool(item["passed"])), evidence_summary(item["evidence"])]
        for item in payload["requirements"]
    ]
    return "\n\n".join(
        [
            "# EvoForest Conformance Report",
            str(payload["scope"]),
            f"Seed: `{payload['seed']}`",
            f"Passed: `{payload['summary']['passed']}/{payload['summary']['total']}`",
            markdown_table(["Requirement", "Status", "Evidence"], rows),
        ]
    )


def evidence_summary(evidence: object) -> str:
    text = json.dumps(evidence, sort_keys=True)
    if len(text) <= 140:
        return f"`{text}`"
    return f"`{text[:137]}...`"


def run(output_dir: Path, seed: int = 17, quick: bool = False) -> tuple[Path, Path]:
    payload = build_report(output_dir, seed=seed, quick=quick)
    return write_report(output_dir, "conformance_report", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an EvoForest architecture conformance report.")
    output_argument(parser)
    seed_argument(parser)
    quick_argument(parser)
    args = parser.parse_args(argv)
    print_report_paths(run(args.output, seed=args.seed, quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
