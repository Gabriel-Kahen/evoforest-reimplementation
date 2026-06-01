from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.synthetic import make_structural_break_data

from benchmarks.common import (
    evaluation_summary,
    fmt_float,
    markdown_table,
    output_argument,
    print_report_paths,
    quick_argument,
    report_scope,
    seed_argument,
    status_mark,
    write_report,
)


def build_report(output_dir: Path, seed: int = 31, quick: bool = False) -> dict[str, Any]:
    steps = 3 if quick else 8
    n_series = 54 if quick else 108
    length = 64 if quick else 96
    max_configurations = 6 if quick else 16
    run_dir = output_dir / "artifacts" / "search_dynamics_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    dataset = make_structural_break_data(n_series=n_series, length=length, seed=seed)
    result = EvolutionLoop(
        build_seed_graph(),
        evaluator=RidgeEvaluator(n_splits=3, seed=seed, max_configurations=max_configurations, irls_steps=2),
        seed=seed,
    ).run(dataset.inputs(), dataset.y, steps=steps, output_dir=run_dir)

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    archive_rows = [json.loads(line) for line in (run_dir / "archive" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    final_graph = checkpoint["graph"]
    final_result = checkpoint["result"]
    event_rows = [event_summary(event) for event in events]
    initial_auc = float(archive_rows[0]["auc"]) if archive_rows else float(result.auc)
    final_auc = float(final_result["auc"])
    accepted_count = sum(1 for event in events if bool(event.get("accepted", False)))
    failed_count = sum(1 for event in events if bool(event.get("failed", False)))
    salvaged_count = sum(len(event.get("salvaged", []) or []) for event in events)
    graph_complexity = graph_payload_summary(final_graph, final_result)

    passed = len(events) == steps and bool(archive_rows) and (run_dir / "memorandum.md").exists()
    return {
        "benchmark": "search_dynamics",
        "scope": report_scope(),
        "seed": seed,
        "quick": quick,
        "summary": {
            "passed": passed,
            "steps": steps,
            "initial_auc": initial_auc,
            "final_auc": final_auc,
            "delta_auc": final_auc - initial_auc,
            "accepted_mutations": accepted_count,
            "failed_mutations": failed_count,
            "salvaged_alternatives": salvaged_count,
            "global_best_versions": len(archive_rows),
            "run_dir": str(run_dir),
        },
        "graph_complexity": graph_complexity,
        "final_evaluation": evaluation_summary(result),
        "archive": archive_rows,
        "events": event_rows,
    }


def event_summary(event: dict[str, Any]) -> dict[str, Any]:
    mutation = event.get("mutation", {})
    add = mutation.get("add", []) if isinstance(mutation, dict) else []
    remove = mutation.get("remove", []) if isinstance(mutation, dict) else []
    return {
        "step": int(event.get("step", 0)),
        "accepted": bool(event.get("accepted", False)),
        "failed": bool(event.get("failed", False)),
        "score": event.get("score"),
        "best_score": event.get("best_score"),
        "delta_to_best": None if event.get("score") is None else float(event["score"]) - float(event["best_score"]),
        "added": [f"{row.get('target_node')}.{row.get('alternative_id')}" for row in add if isinstance(row, dict)],
        "removed": [f"{row.get('target_node')}.{row.get('alternative_id')}" for row in remove if isinstance(row, dict)],
        "salvaged": event.get("salvaged", []),
        "config": event.get("config", {}),
    }


def graph_payload_summary(graph: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    alternatives = sum(len(node.get("alternatives", [])) for node in nodes if isinstance(node, dict))
    output_alternatives = sum(
        len(node.get("alternatives", []))
        for node in nodes
        if isinstance(node, dict) and node.get("kind") == "output"
    )
    diagnostics = result.get("diagnostics", {}) if isinstance(result, dict) else {}
    search = diagnostics.get("configuration_search", {}) if isinstance(diagnostics, dict) else {}
    return {
        "nodes": len(nodes),
        "alternatives": alternatives,
        "output_alternatives": output_alternatives,
        "n_features_best_config": len(result.get("feature_names", [])) if isinstance(result, dict) else 0,
        "n_configs_evaluated": search.get("evaluated", 0) if isinstance(search, dict) else 0,
        "n_configs_total": search.get("total", 0) if isinstance(search, dict) else 0,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    rows = [
        [
            event["step"],
            status_mark(bool(event["accepted"])),
            fmt_float(event["score"]),
            fmt_float(event["best_score"]),
            ", ".join(event["added"]) or "none",
            ", ".join(event["salvaged"]) or "none",
        ]
        for event in payload["events"]
    ]
    summary = payload["summary"]
    complexity = payload["graph_complexity"]
    return "\n\n".join(
        [
            "# Search Dynamics Benchmark",
            str(payload["scope"]),
            f"Seed: `{payload['seed']}`",
            (
                f"Initial AUC: `{fmt_float(summary['initial_auc'])}`; final AUC: `{fmt_float(summary['final_auc'])}`; "
                f"global-best versions: `{summary['global_best_versions']}`; run artifacts: `{summary['run_dir']}`"
            ),
            (
                f"Final graph: `{complexity['nodes']}` nodes, `{complexity['alternatives']}` alternatives, "
                f"`{complexity['output_alternatives']}` output alternatives, `{complexity['n_configs_total']}` total configs."
            ),
            markdown_table(["Step", "Accepted", "Score", "Best", "Added", "Salvaged"], rows),
        ]
    )


def run(output_dir: Path, seed: int = 31, quick: bool = False) -> tuple[Path, Path]:
    payload = build_report(output_dir, seed=seed, quick=quick)
    return write_report(output_dir, "search_dynamics", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic EvoForest search-dynamics benchmark.")
    output_argument(parser)
    seed_argument(parser, default=31)
    quick_argument(parser)
    args = parser.parse_args(argv)
    print_report_paths(run(args.output, seed=args.seed, quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
