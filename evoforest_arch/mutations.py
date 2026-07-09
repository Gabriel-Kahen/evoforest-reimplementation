from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from evoforest_arch.graph import Graph, allowed_output_contract_types, default_output_contract
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
    node_kind: str = ""
    output_contract: dict[str, object] = field(default_factory=dict)
    torch_source: str = ""

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
            "node_kind": self.node_kind,
            "output_contract": self.output_contract,
            "torch_source": self.torch_source,
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
        if _looks_like_paper_style_yaml(text):
            return _from_paper_style_yaml(text)
        data = _parse_machine_style_yaml(text)

        add = tuple(
            MutationSpec(
                kind=str(row.get("kind", "add_alternative")),
                target_node=str(row["target_node"]),
                primitive=str(row.get("primitive", "source" if row.get("source") else "")),
                alternative_id=str(row["alternative_id"]),
                parents=_coerce_string_tuple(row.get("parents", [])),
                description=str(row.get("description", "")),
                source=str(row.get("source", "")),
                global_refs=_coerce_string_tuple(row.get("global_refs", [])),
                node_kind=str(row.get("node_kind", "")),
                output_contract=dict(row.get("output_contract", {})) if isinstance(row.get("output_contract", {}), dict) else {},
                torch_source=str(row.get("torch_source", "")),
            )
            for row in _mapping_rows(data.get("add", []), "add")
        )
        nodes = tuple(
            NodeSpec(
                name=str(row["name"]),
                kind=str(row["kind"]),
                description=str(row.get("description", "")),
            )
            for row in _mapping_rows(data.get("nodes", []), "nodes")
        )
        remove = tuple(
            RemoveSpec(
                target_node=str(row["target_node"]),
                alternative_id=str(row["alternative_id"]),
                reason=str(row.get("reason", "")),
            )
            for row in _mapping_rows(data.get("remove", []), "remove")
        )
        globals_ = tuple(
            GlobalSpec(
                name=str(row["name"]),
                value=_coerce_float_list(row.get("value", [])),
                trainable=_coerce_bool(row.get("trainable", True)),
                description=str(row.get("description", "")),
            )
            for row in _mapping_rows(data.get("globals", []), "globals")
        )
        return cls(
            hypotheses=_coerce_string_tuple(data.get("hypotheses", [])),
            rationale=str(data.get("rationale", "")),
            nodes=nodes,
            add=add,
            remove=remove,
            globals=globals_,
        )


def _parse_machine_style_yaml(text: str) -> dict[str, object]:
    data: dict[str, object] = {"hypotheses": [], "nodes": [], "remove": [], "globals": [], "add": [], "rationale": ""}
    section: str | None = None
    current_item: dict[str, object] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            key, separator, value = stripped.partition(":")
            if not separator:
                raise ValueError(f"Top-level mutation YAML line must be a key/value pair: {line!r}.")
            section = key.strip()
            current_item = None
            data.setdefault(section, [])
            if value.strip():
                data[section] = _parse_machine_yaml_value(value.strip())
            continue
        if section is None:
            raise ValueError(f"Indented YAML item without a section: {line!r}.")
        if stripped == "[]":
            data[section] = []
            current_item = None
            continue
        if indent == 2 and stripped.startswith("- "):
            payload_text = stripped[2:].strip()
            current = data.setdefault(section, [])
            if not isinstance(current, list):
                raise ValueError(f"Section {section!r} is not a list.")
            if not payload_text:
                payload: object = {}
            elif _looks_like_machine_yaml_mapping_item(payload_text):
                key, _separator, value = payload_text.partition(":")
                payload = {key.strip(): _parse_machine_yaml_value(value.strip())}
            else:
                payload = _parse_machine_yaml_value(payload_text)
            current.append(payload)
            current_item = payload if isinstance(payload, dict) else None
            continue
        if indent >= 4 and current_item is not None:
            key, separator, value = stripped.partition(":")
            if not separator:
                raise ValueError(f"Nested mutation YAML line must be a key/value pair: {line!r}.")
            current_item[key.strip()] = _parse_machine_yaml_value(value.strip())
            continue
        raise ValueError(f"Only list items and one-level mappings are supported in mutation YAML: {line!r}.")
    return data


def _looks_like_machine_yaml_mapping_item(text: str) -> bool:
    if not text or text[0] in {"{", "[", "'", '"'}:
        return False
    key, separator, _value = text.partition(":")
    return bool(separator) and bool(key.strip()) and all(character.isalnum() or character in {"_", "-"} for character in key.strip())


def _parse_machine_yaml_value(value: str) -> object:
    value = value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return _yaml_scalar(value)


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
        graph.validate_paper_architecture()
        validate_mutation_document_architecture(graph, document)
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
        cleaned.validate_paper_architecture()
        report.removed_alternatives = explicit_removed + report.removed_alternatives
        return MutationApplication(graph=cleaned, maintenance=report)

    def _apply_add(self, graph: Graph, spec: MutationSpec) -> None:
        if spec.kind != "add_alternative":
            raise ValueError(f"Unsupported mutation kind {spec.kind!r}.")
        if spec.target_node not in graph.nodes:
            raise KeyError(f"Unknown target node {spec.target_node!r}.")
        target_kind = graph.nodes[spec.target_node].kind
        if spec.node_kind and spec.node_kind != target_kind:
            raise ValueError(f"Mutation for {spec.target_node!r} declares node_kind {spec.node_kind!r}, but graph node kind is {target_kind!r}.")
        if spec.source:
            if not self.allow_source:
                raise ValueError("Source-backed mutation alternatives require MutationEngine(allow_source=True).")
            output_contract = default_output_contract(spec.target_node, target_kind)
            output_contract.update(spec.output_contract)
            alternative = build_source_alternative(
                spec.alternative_id,
                spec.parents,
                spec.source,
                description=spec.description,
                global_refs=spec.global_refs,
                node_kind=target_kind,
                output_contract=output_contract,
                torch_source=spec.torch_source,
            )
        else:
            alternative = self.registry.build(spec.primitive, spec.alternative_id, spec.parents)
            self._ensure_registry_globals(graph, alternative.global_refs)
        if spec.description:
            alternative.description = spec.description
        graph.nodes[spec.target_node].add_alternative(alternative)

    @staticmethod
    def _ensure_registry_globals(graph: Graph, global_refs: tuple[str, ...]) -> None:
        existing = set(graph.globals.names())
        for name in global_refs:
            if name in existing:
                continue
            if name == "gate_scale":
                graph.globals.add(name, [1.0], trainable=True, description="Shared scale for callable gate families.")
            elif name == "projection_vector":
                graph.globals.add(
                    name,
                    np.linspace(0.2, 1.0, 16),
                    trainable=True,
                    description="Low-dimensional global vector used by projection outputs.",
                )
            elif name == "residual_huber_scale":
                graph.globals.add(
                    name,
                    [1.0],
                    trainable=False,
                    description="Shared threshold for residual-based Ridge reweighting.",
                )
            existing.add(name)

    @staticmethod
    def _apply_node(graph: Graph, spec: NodeSpec) -> None:
        allowed = {"intermediate", "callable"}
        if spec.kind not in allowed:
            if spec.kind in {"output", "fitting"}:
                raise ValueError(
                    f"Mutation documents cannot add {spec.kind} nodes; add an alternative to 'output', 'ridge_w', or 'ridge_g'."
                )
            raise ValueError(f"Unsupported node kind {spec.kind!r}; expected one of {sorted(allowed)}.")
        graph.add_node(spec.name, spec.kind, spec.description)


def validate_mutation_document_architecture(graph: Graph, document: MutationDocument) -> None:
    """Reject topology and type violations before cloning or executing a mutation."""
    known_nodes = set(graph.nodes)
    new_nodes: set[str] = set()
    for node in document.nodes:
        if node.name in known_nodes or node.name in new_nodes:
            raise ValueError(f"Mutation declares duplicate node {node.name!r}.")
        if node.name in {"output", "ridge_w", "ridge_g"}:
            raise ValueError(f"Mutation cannot redeclare reserved paper architecture node {node.name!r}.")
        if node.kind in {"output", "fitting"}:
            raise ValueError(
                f"Mutation documents cannot add {node.kind} node {node.name!r}; "
                "add behavior as an alternative to 'output', 'ridge_w', or 'ridge_g'."
            )
        if node.kind not in {"intermediate", "callable"}:
            raise ValueError(f"Unsupported node kind {node.kind!r}; expected 'intermediate' or 'callable'.")
        new_nodes.add(node.name)
    all_nodes = known_nodes | new_nodes
    node_kinds = {name: node.kind for name, node in graph.nodes.items()}
    node_kinds.update({node.name: node.kind for node in document.nodes})
    additions_by_node: dict[str, int] = {}
    for addition in document.add:
        additions_by_node[addition.target_node] = additions_by_node.get(addition.target_node, 0) + 1
        if addition.target_node not in all_nodes:
            raise ValueError(f"Add target node {addition.target_node!r} does not exist.")
        for parent in addition.parents:
            if parent not in all_nodes:
                raise ValueError(f"Parent node {parent!r} does not exist.")
        target_kind = node_kinds[addition.target_node]
        if addition.node_kind and addition.node_kind != target_kind:
            raise ValueError(
                f"Mutation for {addition.target_node!r} declares node_kind {addition.node_kind!r}, "
                f"but the target is {target_kind!r}."
            )
        contract_type = str(addition.output_contract.get("type", ""))
        allowed_types = {"", *allowed_output_contract_types(addition.target_node, target_kind)}
        if contract_type not in allowed_types:
            raise ValueError(
                f"Alternative for {addition.target_node!r} has output contract type {contract_type!r}; "
                f"expected one of {sorted(allowed_types)!r}."
            )

    removals_by_node: dict[str, int] = {}
    for removal in document.remove:
        removals_by_node[removal.target_node] = removals_by_node.get(removal.target_node, 0) + 1
    for node_name, removal_count in removals_by_node.items():
        node = graph.nodes.get(node_name)
        if node is None or node.kind not in {"output", "fitting"}:
            continue
        remaining = len(node.alternatives) - removal_count + additions_by_node.get(node_name, 0)
        if remaining < 1:
            raise ValueError(f"Mutation would leave paper architecture node {node_name!r} without alternatives.")


def extract_mutation_yaml(text: str) -> str:
    stripped = text.strip()
    lines = stripped.splitlines()
    if lines and lines[0].strip().lower() in {"yaml", "yml"}:
        stripped = "\n".join(lines[1:]).strip()
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


def _looks_like_paper_style_yaml(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines()]
    in_add = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" "):
            key, _, _value = line.partition(":")
            in_add = key.strip() == "add"
            continue
        if in_add and not stripped.startswith("- ") and stripped.endswith(":"):
            return True
    return False


def _from_paper_style_yaml(text: str) -> MutationDocument:
    rationale = ""
    hypotheses: list[str] = []
    removals: list[RemoveSpec] = []
    additions: list[MutationSpec] = []
    section: str | None = None
    current_add_node: str | None = None
    add_counts: dict[str, int] = {}

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" "):
            key, _, value = line.partition(":")
            section = key.strip()
            current_add_node = None
            if section == "rationale" and value.strip():
                rationale = _yaml_scalar(value.strip())
            continue
        if section == "hypotheses" and stripped.startswith("- "):
            hypotheses.append(_yaml_scalar(stripped[2:].strip()))
            continue
        if section == "remove" and stripped.startswith("- "):
            target, alternative = _split_remove_target(_yaml_scalar(stripped[2:].strip()))
            removals.append(RemoveSpec(target_node=target, alternative_id=alternative))
            continue
        if section == "add":
            if stripped.endswith(":") and not stripped.startswith("- "):
                current_add_node = stripped[:-1].strip()
                continue
            if stripped.startswith("- ") and current_add_node:
                item = _yaml_scalar(stripped[2:].strip())
                count = add_counts.get(current_add_node, 0)
                add_counts[current_add_node] = count + 1
                payload = _paper_source_payload(item)
                source = str(payload.get("source", item))
                parents = _coerce_string_tuple(payload.get("parents", _infer_source_parents(source)))
                global_refs = _coerce_string_tuple(payload.get("global_refs", _infer_source_global_refs(source)))
                additions.append(
                    MutationSpec(
                        kind="add_alternative",
                        target_node=current_add_node,
                        primitive="source",
                        alternative_id=str(payload.get("alternative_id", _paper_source_alternative_id(current_add_node, source, count))),
                        parents=parents,
                        source=source,
                        description=str(payload.get("description", "Paper-style source-backed lambda alternative.")),
                        global_refs=global_refs,
                        node_kind=str(payload.get("node_kind", "")),
                        output_contract=dict(payload.get("output_contract", {})) if isinstance(payload.get("output_contract", {}), dict) else {},
                        torch_source=str(payload.get("torch_source", "")),
                    )
                )

    return MutationDocument(
        hypotheses=tuple(hypotheses),
        rationale=rationale,
        remove=tuple(removals),
        add=tuple(additions),
    )


def _yaml_scalar(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            if len(value) >= 2 and value[-1] == value[0]:
                return value[1:-1]
    return value


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if len(stripped) >= 2 and stripped[0] in {"[", "("} and stripped[-1] in {"]", ")"}:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                pass
            else:
                return _coerce_string_tuple(parsed)
        return (stripped,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _coerce_float_list(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if len(stripped) >= 2 and stripped[0] in {"[", "("} and stripped[-1] in {"]", ")"}:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                pass
            else:
                return _coerce_float_list(parsed)
        return [float(stripped)]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _coerce_bool(value: object) -> bool:
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"false", "no", "0", "off"}:
            return False
        if stripped in {"true", "yes", "1", "on"}:
            return True
    return bool(value)


def _mapping_rows(value: object, section: str) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return (value,)
    if not isinstance(value, list):
        raise ValueError(f"Mutation YAML section {section!r} must be a list of mappings.")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"Mutation YAML section {section!r} item {index} must be a mapping, got {type(row).__name__}.")
        rows.append(row)
    return tuple(rows)


def _paper_source_payload(item: str) -> dict[str, object]:
    stripped = item.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("Paper-style source mutation object must be a JSON mapping.")
        if "source" not in payload:
            raise ValueError("Paper-style source mutation object requires a source field.")
        return payload
    return {"source": item}


def _infer_source_parents(source: str) -> tuple[str, ...]:
    names: list[str] = []
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError:
        return ()
    for node in ast.walk(tree):
        value: str | None = None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "values":
            value = _literal_string_slice(node.slice)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "values"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            value = node.args[0].value
        if value is not None and value not in names:
            names.append(value)
    return tuple(names)


def _infer_source_global_refs(source: str) -> tuple[str, ...]:
    names: list[str] = []
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError:
        return ()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "globals"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "ctx"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value not in names
        ):
            names.append(node.args[0].value)
    return tuple(names)


def _literal_string_slice(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _split_remove_target(value: str) -> tuple[str, str]:
    if "." in value:
        return tuple(value.split(".", maxsplit=1))  # type: ignore[return-value]
    return "", value


def _paper_source_alternative_id(target_node: str, source: str, count: int) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    clean_target = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in target_node).strip("_") or "node"
    return f"{clean_target}_source_{digest}_{count}"


def built_in_mutations() -> list[MutationSpec]:
    return [
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="pass_outputs",
            alternative_id="generic_base_summary_output_mutation",
            parents=("base_features", "summary_features"),
            description="Add a compact generic output path over direct and summary features.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="activated_outputs",
            alternative_id="generic_activated_base_output_mutation",
            parents=("base_features", "nonlinear_features", "activation"),
            description="Add an activated generic output path over direct and nonlinear features.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="projection_outputs",
            alternative_id="generic_projection_output_mutation",
            parents=("base_features", "nonlinear_features"),
            description="Add a trainable-projection generic output path.",
        ),
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
