from __future__ import annotations

from dataclasses import dataclass, field
import json

from evoforest_arch.graph import Graph, NodeAlternative


@dataclass
class MaintenanceReport:
    removed_alternatives: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    removed_globals: list[str] = field(default_factory=list)
    collapsed_duplicates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "removed_alternatives": self.removed_alternatives,
            "removed_nodes": self.removed_nodes,
            "removed_globals": self.removed_globals,
            "collapsed_duplicates": self.collapsed_duplicates,
        }


class GraphMaintenance:
    """Conservative maintenance pass modeled on the paper's cleanup stage."""

    def clean(self, graph: Graph) -> tuple[Graph, MaintenanceReport]:
        cleaned = graph.clone()
        report = MaintenanceReport()
        cleaned.validate_acyclic()
        self._collapse_duplicate_alternatives(cleaned, report)
        self._remove_empty_nodes(cleaned, report)
        self._prune_unreachable_nodes(cleaned, report)
        self._prune_unused_globals(cleaned, report)
        cleaned.validate_acyclic()
        return cleaned, report

    @staticmethod
    def _collapse_duplicate_alternatives(graph: Graph, report: MaintenanceReport) -> None:
        for node in graph.nodes.values():
            seen: set[tuple[object, ...]] = set()
            kept: list[NodeAlternative] = []
            for alternative in node.alternatives:
                signature = (
                    alternative.primitive,
                    alternative.parents,
                    alternative.source,
                    alternative.torch_source,
                    json.dumps(alternative.output_contract, sort_keys=True, separators=(",", ":")),
                    alternative.description,
                    alternative.tags,
                    alternative.global_refs,
                )
                if signature in seen:
                    report.collapsed_duplicates.append(f"{node.name}.{alternative.id}")
                    report.removed_alternatives.append(f"{node.name}.{alternative.id}")
                    continue
                seen.add(signature)
                kept.append(alternative)
            node.alternatives = kept

    @staticmethod
    def _remove_empty_nodes(graph: Graph, report: MaintenanceReport) -> None:
        roots = set(graph.output_nodes() + graph.fitting_nodes())
        removable = [
            name
            for name, node in graph.nodes.items()
            if node.kind not in {"input"} and name not in roots and not node.alternatives
        ]
        for name in removable:
            del graph.nodes[name]
            report.removed_nodes.append(name)

    @staticmethod
    def _prune_unreachable_nodes(graph: Graph, report: MaintenanceReport) -> None:
        reachable = graph.reachable_nodes()
        for name in list(graph.nodes):
            if graph.nodes[name].kind == "input":
                continue
            if name not in reachable:
                del graph.nodes[name]
                report.removed_nodes.append(name)

    @staticmethod
    def _prune_unused_globals(graph: Graph, report: MaintenanceReport) -> None:
        referenced = graph.referenced_globals()
        for name in list(graph.globals.names()):
            if name not in referenced:
                graph.globals.remove(name)
                report.removed_globals.append(name)
