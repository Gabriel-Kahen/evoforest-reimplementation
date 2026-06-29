from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evoforest_arch.evaluator import EvaluationResult
from evoforest_arch.graph import Graph


DEFAULT_OUTPUT_DIR = Path("benchmark_reports")


def output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON and Markdown reports.")


def seed_argument(parser: argparse.ArgumentParser, default: int = 17) -> None:
    parser.add_argument("--seed", type=int, default=default, help="Deterministic random seed for generated data and search.")


def quick_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true", help="Use smaller workloads for smoke tests and CI.")


def write_report(output_dir: Path, name: str, payload: dict[str, Any], markdown: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    markdown_path = output_dir / f"{name}.md"
    json_path.write_text(json.dumps(as_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return json_path, markdown_path


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return as_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def status_mark(passed: bool) -> str:
    return "PASS" if passed else "CHECK"


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def evaluation_summary(result: EvaluationResult) -> dict[str, object]:
    search = result.diagnostics.get("configuration_search", {})
    fitting = result.diagnostics.get("fitting", {})
    scoring = result.diagnostics.get("scoring_context", {})
    return {
        "score": float(result.score),
        "config": result.config,
        "n_features": len(result.feature_names),
        "n_configs_evaluated": int(search.get("evaluated", 1)) if isinstance(search, dict) else 1,
        "n_configs_total": int(search.get("total", 1)) if isinstance(search, dict) else 1,
        "configuration_search_capped": bool(search.get("capped", False)) if isinstance(search, dict) else False,
        "cache": search.get("cache", {}) if isinstance(search, dict) else {},
        "ridge_w": fitting.get("ridge_w", {}) if isinstance(fitting, dict) else {},
        "ridge_g": fitting.get("ridge_g", {}) if isinstance(fitting, dict) else {},
        "global_ridge_score": scoring.get("global_ridge_score", 0.0) if isinstance(scoring, dict) else 0.0,
        "shap_reconstruction_error": scoring.get("shap_reconstruction_error", 0.0) if isinstance(scoring, dict) else 0.0,
    }


def graph_summary(graph: Graph) -> dict[str, object]:
    kinds: dict[str, int] = {}
    alternatives_by_kind: dict[str, int] = {}
    for node in graph.nodes.values():
        kinds[node.kind] = kinds.get(node.kind, 0) + 1
        alternatives_by_kind[node.kind] = alternatives_by_kind.get(node.kind, 0) + len(node.alternatives)
    space = graph.configuration_space()
    total_configs = 1
    for alternatives in space.values():
        total_configs *= max(1, len(alternatives))
    return {
        "name": graph.name,
        "nodes": len(graph.nodes),
        "node_kinds": kinds,
        "alternatives": sum(len(node.alternatives) for node in graph.nodes.values()),
        "alternatives_by_kind": alternatives_by_kind,
        "configuration_nodes": sorted(space),
        "total_configurations": int(total_configs),
        "output_nodes": graph.output_nodes(),
        "fitting_nodes": graph.fitting_nodes(),
        "globals": graph.globals.to_dict(),
    }


def report_scope() -> str:
    return (
        "These benchmarks validate architecture-level behavior of this clean-room "
        "reimplementation. They do not claim to reproduce the authors' private evolved "
        "graph, private code-generation stack, competition pipeline, or reported task score."
    )


def print_report_paths(paths: tuple[Path, Path]) -> None:
    json_path, markdown_path = paths
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2), flush=True)
