"""Safe transfer of reusable alternatives between compatible EvoForest graphs."""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

from evoforest_arch.evaluation_cache import fingerprint_callable, fingerprint_value
from evoforest_arch.graph import Graph, GraphNode, NodeAlternative


NodeFilter = Callable[[str, GraphNode], bool]
AlternativeFilter = Callable[[str, NodeAlternative], bool]


@dataclass(frozen=True)
class TransferDecision:
    node: str
    alternative: str
    status: str
    reason: str

    @property
    def qualified_name(self) -> str:
        return f"{self.node}.{self.alternative}"


@dataclass
class TransferReport:
    decisions: list[TransferDecision] = field(default_factory=list)
    copied_globals: list[str] = field(default_factory=list)

    @property
    def transferred(self) -> list[str]:
        return [decision.qualified_name for decision in self.decisions if decision.status == "transferred"]

    @property
    def skipped(self) -> list[str]:
        return [decision.qualified_name for decision in self.decisions if decision.status == "skipped"]

    @property
    def reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            if decision.status == "skipped":
                counts[decision.reason] = counts.get(decision.reason, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "transferred": self.transferred,
            "skipped": self.skipped,
            "copied_globals": list(self.copied_globals),
            "reasons": self.reasons,
            "decisions": [
                {
                    "node": decision.node,
                    "alternative": decision.alternative,
                    "status": decision.status,
                    "reason": decision.reason,
                }
                for decision in self.decisions
            ],
        }


def transfer_alternatives(
    source: Graph,
    target: Graph,
    *,
    node_names: Collection[str] | None = None,
    node_kinds: Collection[str] | None = None,
    node_filter: NodeFilter | None = None,
    alternative_filter: AlternativeFilter | None = None,
    copy_missing_globals: bool = False,
    preserve_statistics: bool = False,
) -> TransferReport:
    """Copy compatible alternatives from ``source`` into ``target``.

    Nodes and parents are matched by name and kind. Global references must exist
    with matching shapes and trainability, unless ``copy_missing_globals`` copies
    a missing parameter from the source. Existing alternative IDs and semantically
    equivalent alternatives are never duplicated.
    """

    selected_names = set(node_names) if node_names is not None else None
    selected_kinds = set(node_kinds) if node_kinds is not None else None
    report = TransferReport()

    for node_name, source_node in source.nodes.items():
        for alternative in source_node.alternatives:
            reason = _filter_reason(
                node_name,
                source_node,
                alternative,
                selected_names,
                selected_kinds,
                node_filter,
                alternative_filter,
            )
            if reason:
                _skip(report, node_name, alternative.id, reason)
                continue

            reason = _compatibility_reason(
                source,
                target,
                node_name,
                source_node,
                alternative,
                allow_missing_globals=copy_missing_globals,
            )
            if reason:
                _skip(report, node_name, alternative.id, reason)
                continue

            target_node = target.nodes[node_name]
            if any(existing.id == alternative.id for existing in target_node.alternatives):
                _skip(report, node_name, alternative.id, "duplicate_id")
                continue
            signature = _semantic_signature(alternative)
            if any(_semantic_signature(existing) == signature for existing in target_node.alternatives):
                _skip(report, node_name, alternative.id, "duplicate_semantics")
                continue
            if _would_create_cycle(target, node_name, alternative.parents):
                _skip(report, node_name, alternative.id, "would_create_cycle")
                continue

            missing_globals = [name for name in alternative.global_refs if name not in target.globals.names()]
            for global_name in missing_globals:
                source_parameter = source.globals.to_dict()[global_name]
                target.globals.add(
                    global_name,
                    source.globals.get(global_name).copy(),
                    trainable=bool(source_parameter["trainable"]),
                    description=str(source_parameter["description"]),
                )
                if global_name not in report.copied_globals:
                    report.copied_globals.append(global_name)

            transplanted = copy.deepcopy(alternative)
            if not preserve_statistics:
                transplanted.stats = {}
                transplanted.age = 0
            target_node.add_alternative(transplanted)
            report.decisions.append(TransferDecision(node_name, alternative.id, "transferred", "compatible"))

    return report


def _filter_reason(
    node_name: str,
    node: GraphNode,
    alternative: NodeAlternative,
    node_names: set[str] | None,
    node_kinds: set[str] | None,
    node_filter: NodeFilter | None,
    alternative_filter: AlternativeFilter | None,
) -> str:
    if node_names is not None and node_name not in node_names:
        return "node_filtered"
    if node_kinds is not None and node.kind not in node_kinds:
        return "node_filtered"
    if node_filter is not None and not node_filter(node_name, node):
        return "node_filtered"
    if alternative_filter is not None and not alternative_filter(node_name, alternative):
        return "alternative_filtered"
    return ""


def _compatibility_reason(
    source: Graph,
    target: Graph,
    node_name: str,
    source_node: GraphNode,
    alternative: NodeAlternative,
    *,
    allow_missing_globals: bool,
) -> str:
    if node_name not in target.nodes:
        return "target_node_missing"
    if target.nodes[node_name].kind != source_node.kind:
        return "node_kind_mismatch"

    for parent in alternative.parents:
        if parent not in target.nodes:
            return "target_parent_missing"
        if target.nodes[parent].kind != source.nodes[parent].kind:
            return "parent_kind_mismatch"
        if target.nodes[parent].kind != "input" and not target.nodes[parent].alternatives:
            return "target_parent_has_no_alternatives"

    source_globals = source.globals.to_dict()
    target_globals = target.globals.to_dict()
    for global_name in alternative.global_refs:
        if global_name not in source_globals:
            return "source_global_missing"
        if global_name not in target_globals:
            if allow_missing_globals:
                continue
            return "target_global_missing"
        if source_globals[global_name]["shape"] != target_globals[global_name]["shape"]:
            return "global_shape_mismatch"
        if source_globals[global_name]["trainable"] != target_globals[global_name]["trainable"]:
            return "global_trainability_mismatch"
    return ""


def _semantic_signature(alternative: NodeAlternative) -> str:
    return fingerprint_value(
        (
            alternative.parents,
            alternative.primitive,
            alternative.global_refs,
            alternative.source,
            alternative.torch_source,
            alternative.output_contract,
            fingerprint_callable(alternative.fn),
            fingerprint_callable(alternative.torch_fn) if alternative.torch_fn is not None else None,
        )
    )


def _would_create_cycle(target: Graph, node_name: str, parents: tuple[str, ...]) -> bool:
    def reaches_node(start: str) -> bool:
        pending = [start]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == node_name:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(
                parent
                for alternative in target.nodes[current].alternatives
                for parent in alternative.parents
            )
        return False

    return any(reaches_node(parent) for parent in parents)


def _skip(report: TransferReport, node: str, alternative: str, reason: str) -> None:
    report.decisions.append(TransferDecision(node, alternative, "skipped", reason))
