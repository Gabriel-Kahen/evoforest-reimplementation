from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from typing import Any

from evoforest_arch.graph import Graph
from evoforest_arch.primitives import PrimitiveRegistry
from evoforest_arch.source import build_source_alternative


def graph_hash(graph: Graph) -> str:
    payload = json.dumps(graph.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def graph_from_dict(
    payload: dict[str, Any],
    *,
    registry: PrimitiveRegistry | None = None,
    allow_source: bool = False,
) -> Graph:
    """Rehydrate a serialized graph whose alternatives reference known primitives.

    Source-backed alternatives are intentionally opt-in because they execute local
    Python code when evaluated.
    """

    registry = registry or PrimitiveRegistry.default()
    graph_payload = payload.get("graph", payload)
    if not isinstance(graph_payload, dict):
        raise TypeError("Graph payload must be a mapping.")
    task_schema = graph_payload.get("task_schema")
    graph = Graph(name=str(graph_payload.get("name", "evoforest")), task_schema=task_schema if isinstance(task_schema, dict) else None)

    node_rows = graph_payload.get("nodes", [])
    if not isinstance(node_rows, list):
        raise TypeError("Graph payload field 'nodes' must be a list.")
    for row in node_rows:
        if not isinstance(row, dict):
            raise TypeError("Graph node entries must be mappings.")
        name = str(row["name"])
        kind = str(row["kind"])
        description = str(row.get("description", ""))
        if kind == "input":
            graph.add_input(name, description)
        else:
            graph.add_node(name, kind, description)

    globals_payload = graph_payload.get("globals", {})
    if isinstance(globals_payload, dict):
        for key, row in globals_payload.items():
            if not isinstance(row, dict):
                raise TypeError(f"Global payload for {key!r} must be a mapping.")
            graph.globals.add(
                name=str(row.get("name", key)),
                value=row.get("value", []),
                trainable=bool(row.get("trainable", True)),
                description=str(row.get("description", "")),
            )

    for row in node_rows:
        assert isinstance(row, dict)
        node_name = str(row["name"])
        if str(row["kind"]) == "input":
            continue
        alternatives = row.get("alternatives", [])
        if not isinstance(alternatives, list):
            raise TypeError(f"Node {node_name!r} alternatives must be a list.")
        for alt_row in alternatives:
            if not isinstance(alt_row, dict):
                raise TypeError(f"Node {node_name!r} alternatives must be mappings.")
            primitive = alt_row.get("primitive")
            source = str(alt_row.get("source", "") or "")
            alt_id = str(alt_row["id"])
            parents = tuple(str(parent) for parent in alt_row.get("parents", []))
            global_refs = tuple(str(name) for name in alt_row.get("global_refs", []))
            output_contract = dict(alt_row.get("output_contract", {})) if isinstance(alt_row.get("output_contract", {}), dict) else {}
            torch_source = str(alt_row.get("torch_source", "") or "")
            if source or primitive == "source":
                if not allow_source:
                    raise ValueError("Deserializing source-backed alternatives requires allow_source=True.")
                alternative = build_source_alternative(
                    alt_id,
                    parents,
                    source,
                    description=str(alt_row.get("description", "")),
                    global_refs=global_refs,
                    tags=tuple(str(tag) for tag in alt_row.get("tags", [])),
                    node_kind=str(row["kind"]),
                    output_contract=output_contract,
                    torch_source=torch_source,
                )
            elif primitive:
                alternative = registry.build(str(primitive), alt_id, parents)
            else:
                raise ValueError(f"Alternative {node_name}.{alt_id} has no primitive or source to rebuild from.")

            alternative.description = str(alt_row.get("description", alternative.description))
            alternative.tags = tuple(str(tag) for tag in alt_row.get("tags", alternative.tags))
            alternative.global_refs = global_refs or alternative.global_refs
            alternative.source = source
            alternative.age = int(alt_row.get("age", 0))
            stats = alt_row.get("stats", {})
            alternative.stats = copy.deepcopy(stats) if isinstance(stats, dict) else {}
            graph.nodes[node_name].add_alternative(alternative)

    graph.validate_paper_architecture()
    return graph


def graph_from_path(path: str | pathlib.Path, *, registry: PrimitiveRegistry | None = None, allow_source: bool = False) -> Graph:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return graph_from_dict(payload, registry=registry, allow_source=allow_source)


def write_graph(path: str | pathlib.Path, graph: Graph, *, metadata: dict[str, Any] | None = None) -> pathlib.Path:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graph_hash": graph_hash(graph),
        "graph": graph.to_dict(),
        "metadata": metadata or {},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
