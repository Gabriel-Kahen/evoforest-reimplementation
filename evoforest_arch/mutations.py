from __future__ import annotations

from dataclasses import dataclass, field
import json

from evoforest_arch.graph import Graph
from evoforest_arch.maintenance import GraphMaintenance, MaintenanceReport
from evoforest_arch.primitives import PrimitiveRegistry
from evoforest_arch.source import build_source_alternative


@dataclass(frozen=True)
class MutationSpec:
    kind: str
    target_node: str
    primitive: str
    alternative_id: str
    parents: tuple[str, ...]
    description: str = ""
    source: str = ""
    global_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "target_node": self.target_node,
            "primitive": self.primitive,
            "alternative_id": self.alternative_id,
            "parents": list(self.parents),
            "description": self.description,
            "source": self.source,
            "global_refs": list(self.global_refs),
        }


@dataclass(frozen=True)
class RemoveSpec:
    target_node: str
    alternative_id: str
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "target_node": self.target_node,
            "alternative_id": self.alternative_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GlobalSpec:
    name: str
    value: list[float]
    trainable: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "trainable": self.trainable,
            "description": self.description,
        }


@dataclass(frozen=True)
class NodeSpec:
    name: str
    kind: str
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
        }


@dataclass(frozen=True)
class MutationDocument:
    hypotheses: tuple[str, ...] = ()
    rationale: str = ""
    nodes: tuple[NodeSpec, ...] = ()
    add: tuple[MutationSpec, ...] = ()
    remove: tuple[RemoveSpec, ...] = ()
    globals: tuple[GlobalSpec, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "hypotheses": list(self.hypotheses),
            "rationale": self.rationale,
            "nodes": [spec.to_dict() for spec in self.nodes],
            "add": [spec.to_dict() for spec in self.add],
            "remove": [spec.to_dict() for spec in self.remove],
            "globals": [spec.to_dict() for spec in self.globals],
        }

    def to_yaml(self) -> str:
        rows = [f"rationale: {json.dumps(self.rationale)}"]
        for key, values in (
            ("hypotheses", list(self.hypotheses)),
            ("nodes", [spec.to_dict() for spec in self.nodes]),
            ("remove", [spec.to_dict() for spec in self.remove]),
            ("globals", [spec.to_dict() for spec in self.globals]),
            ("add", [spec.to_dict() for spec in self.add]),
        ):
            rows.append(f"{key}:")
            if values:
                rows.extend(f"  - {json.dumps(value, sort_keys=True)}" for value in values)
            else:
                rows.append("  []")
        return "\n".join(rows) + "\n"

    @classmethod
    def from_yaml(cls, text: str) -> "MutationDocument":
        text = extract_mutation_yaml(text)
        data: dict[str, object] = {"hypotheses": [], "nodes": [], "remove": [], "globals": [], "add": [], "rationale": ""}
        section: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" "):
                key, _, value = line.partition(":")
                section = key.strip()
                if value.strip():
                    data[section] = json.loads(value.strip())
                continue
            if section is None:
                raise ValueError(f"Indented YAML item without a section: {line!r}.")
            item = line.strip()
            if item == "[]":
                data[section] = []
                continue
            if not item.startswith("- "):
                raise ValueError(f"Only list items are supported in mutation YAML: {line!r}.")
            payload = json.loads(item[2:].strip())
            current = data.setdefault(section, [])
            if not isinstance(current, list):
                raise ValueError(f"Section {section!r} is not a list.")
            current.append(payload)

        add = tuple(
            MutationSpec(
                kind=str(row.get("kind", "add_alternative")),
                target_node=str(row["target_node"]),
                primitive=str(row.get("primitive", "source" if row.get("source") else "")),
                alternative_id=str(row["alternative_id"]),
                parents=tuple(str(parent) for parent in row.get("parents", [])),
                description=str(row.get("description", "")),
                source=str(row.get("source", "")),
                global_refs=tuple(str(name) for name in row.get("global_refs", [])),
            )
            for row in data.get("add", [])
        )
        nodes = tuple(
            NodeSpec(
                name=str(row["name"]),
                kind=str(row["kind"]),
                description=str(row.get("description", "")),
            )
            for row in data.get("nodes", [])
        )
        remove = tuple(
            RemoveSpec(
                target_node=str(row["target_node"]),
                alternative_id=str(row["alternative_id"]),
                reason=str(row.get("reason", "")),
            )
            for row in data.get("remove", [])
        )
        globals_ = tuple(
            GlobalSpec(
                name=str(row["name"]),
                value=[float(value) for value in row.get("value", [])],
                trainable=bool(row.get("trainable", True)),
                description=str(row.get("description", "")),
            )
            for row in data.get("globals", [])
        )
        return cls(
            hypotheses=tuple(str(item) for item in data.get("hypotheses", [])),
            rationale=str(data.get("rationale", "")),
            nodes=nodes,
            add=add,
            remove=remove,
            globals=globals_,
        )


@dataclass
class MutationApplication:
    graph: Graph
    maintenance: MaintenanceReport = field(default_factory=MaintenanceReport)


class MutationEngine:
    def __init__(
        self,
        registry: PrimitiveRegistry | None = None,
        maintenance: GraphMaintenance | None = None,
        allow_source: bool = False,
    ) -> None:
        self.registry = registry or PrimitiveRegistry.default()
        self.maintenance = maintenance or GraphMaintenance()
        self.allow_source = bool(allow_source)

    def apply(self, graph: Graph, spec: MutationSpec) -> Graph:
        return self.apply_document(graph, MutationDocument(add=(spec,))).graph

    def apply_document(self, graph: Graph, document: MutationDocument) -> MutationApplication:
        mutated = graph.clone()
        explicit_removed: list[str] = []
        for global_spec in document.globals:
            if global_spec.name in mutated.globals.names():
                raise ValueError(f"Global {global_spec.name!r} already exists; globals are append-only.")
            mutated.globals.add(
                global_spec.name,
                global_spec.value,
                trainable=global_spec.trainable,
                description=global_spec.description,
            )
        for node_spec in document.nodes:
            self._apply_node(mutated, node_spec)
        for remove_spec in document.remove:
            mutated.remove_alternative(remove_spec.target_node, remove_spec.alternative_id)
            explicit_removed.append(f"{remove_spec.target_node}.{remove_spec.alternative_id}")
        for spec in document.add:
            self._apply_add(mutated, spec)
        cleaned, report = self.maintenance.clean(mutated)
        report.removed_alternatives = explicit_removed + report.removed_alternatives
        return MutationApplication(graph=cleaned, maintenance=report)

    def _apply_add(self, graph: Graph, spec: MutationSpec) -> None:
        if spec.kind != "add_alternative":
            raise ValueError(f"Unsupported mutation kind {spec.kind!r}.")
        if spec.target_node not in graph.nodes:
            raise KeyError(f"Unknown target node {spec.target_node!r}.")
        if spec.source:
            if not self.allow_source:
                raise ValueError("Source-backed mutation alternatives require MutationEngine(allow_source=True).")
            alternative = build_source_alternative(
                spec.alternative_id,
                spec.parents,
                spec.source,
                description=spec.description,
                global_refs=spec.global_refs,
            )
        else:
            alternative = self.registry.build(spec.primitive, spec.alternative_id, spec.parents)
        if spec.description:
            alternative.description = spec.description
        graph.nodes[spec.target_node].add_alternative(alternative)

    @staticmethod
    def _apply_node(graph: Graph, spec: NodeSpec) -> None:
        allowed = {"intermediate", "callable", "output", "fitting"}
        if spec.kind not in allowed:
            raise ValueError(f"Unsupported node kind {spec.kind!r}; expected one of {sorted(allowed)}.")
        graph.add_node(spec.name, spec.kind, spec.description)


def extract_mutation_yaml(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    chunks = stripped.split("```")
    for chunk in chunks[1::2]:
        candidate = chunk.strip()
        if candidate.startswith(("yaml", "yml")):
            candidate = candidate.splitlines()[1:]
            candidate = "\n".join(candidate).strip()
        if "add:" in candidate or "remove:" in candidate or "rationale:" in candidate:
            return candidate
    return stripped


def built_in_mutations() -> list[MutationSpec]:
    return [
        MutationSpec(
            kind="add_alternative",
            target_node="shape_stats",
            primitive="spectral_basic",
            alternative_id="spectral_mutation",
            parents=("series",),
            description="Add a frequency-ratio shape alternative.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="segment_stats",
            primitive="segment_robust",
            alternative_id="robust_mutation",
            parents=("series",),
            description="Add a quantile-based robust segment alternative.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="segment_stats",
            primitive="segment_late_shift",
            alternative_id="late_shift_mutation",
            parents=("series",),
            description="Add period-boundary tail jump and late drift statistics.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="trend_stats",
            primitive="trend_late_window",
            alternative_id="late_trend_mutation",
            parents=("series",),
            description="Add late-window trend and post-period acceleration statistics.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="segment_stats",
            primitive="row_recent_change",
            alternative_id="row_recent_change_mutation",
            parents=("series",),
            description="Add row-local older-vs-recent level-change features for target-time windows.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="shape_stats",
            primitive="row_volatility_burst",
            alternative_id="row_volatility_burst_mutation",
            parents=("series",),
            description="Add row-local volatility burst features from causal lookback increments.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="shape_stats",
            primitive="row_cusum_local",
            alternative_id="row_cusum_local_mutation",
            parents=("series",),
            description="Add row-local CUSUM profile features over causal lookback windows.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="activation",
            primitive="sigmoid_gate_callable",
            alternative_id="sigmoid_gate_mutation",
            parents=(),
            description="Add a sigmoid gate callable alternative.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="structural_break_baseline_outputs",
            alternative_id="structural_break_baseline_output_mutation",
            parents=("series",),
            description="Add a deterministic structural-break baseline feature block as graph outputs.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="event_detection_outputs",
            alternative_id="event_detection_output_mutation",
            parents=("series",),
            description="Add always-evaluated event-detection features directly from the sequence.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="interaction_outputs",
            alternative_id="interaction_output_mutation",
            parents=("segment_stats", "trend_stats", "shape_stats"),
            description="Add standardized aggregate and pairwise interaction output features.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="row_local_outputs",
            alternative_id="row_local_output_mutation",
            parents=("series",),
            description="Add always-evaluated row-local target-time lookback features.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="row_time_basis_outputs",
            alternative_id="row_time_basis_output_mutation",
            parents=("series",),
            description="Add expanded target-time and observed-lookback basis features.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="row_multiscale_tail_outputs",
            alternative_id="row_multiscale_tail_output_mutation",
            parents=("series",),
            description="Add multiscale recent-tail drift, volatility, slope, and drawdown features.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="row_baseline_outputs",
            alternative_id="row_baseline_output_mutation",
            parents=("series",),
            description="Add the deterministic row-level baseline feature block as graph outputs.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="row_context_outputs",
            alternative_id="row_context_output_mutation",
            parents=("series",),
            description="Add optional row metadata features from target time, period, and observed lookback length.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="projection_outputs",
            alternative_id="projection_output_mutation",
            parents=("segment_stats", "trend_stats", "shape_stats"),
            description="Add a projection output feature without increasing configuration count.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="shape_stats",
            primitive="shape_drawdown",
            alternative_id="drawdown_mutation",
            parents=("series",),
            description="Add post-period drawdown and drawup shape statistics.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="shape_stats",
            primitive="shape_post_concentration",
            alternative_id="post_concentration_mutation",
            parents=("series",),
            description="Add late post-period energy concentration statistics.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="ridge_w",
            primitive="boundary_energy_weight",
            alternative_id="boundary_weight_mutation",
            parents=("series",),
            description="Add an alternative sample-weighting policy for Ridge.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="ridge_w",
            primitive="late_energy_weight",
            alternative_id="late_energy_weight_mutation",
            parents=("series",),
            description="Add a sample-weighting policy for late post-period movement.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="ridge_g",
            primitive="huber_residual_weight",
            alternative_id="huber_residual_mutation",
            parents=(),
            description="Add a robust residual reweighting policy for Ridge.",
        ),
    ]
