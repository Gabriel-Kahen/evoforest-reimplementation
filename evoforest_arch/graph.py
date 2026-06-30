from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import copy

import numpy as np

from evoforest_arch.globals import GlobalStore


ArrayLike = np.ndarray
AlternativeFn = Callable[["EvalContext", dict[str, Any]], Any]


@dataclass
class FeatureBlock:
    values: np.ndarray
    names: list[str]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError(f"FeatureBlock values must be 1-D or 2-D, got {values.shape}.")
        if len(self.names) != values.shape[1]:
            raise ValueError(f"Feature name count {len(self.names)} does not match {values.shape[1]} columns.")
        self.values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class CallableFamily:
    name: str
    apply: Callable[[np.ndarray], np.ndarray]
    description: str = ""


@dataclass(frozen=True)
class ResidualWeightRule:
    name: str
    apply: Callable[[np.ndarray], np.ndarray]
    description: str = ""


@dataclass
class NodeAlternative:
    id: str
    parents: tuple[str, ...]
    fn: AlternativeFn
    description: str = ""
    tags: tuple[str, ...] = ()
    primitive: str | None = None
    global_refs: tuple[str, ...] = ()
    torch_fn: AlternativeFn | None = None
    source: str = ""
    torch_source: str = ""
    output_contract: dict[str, object] = field(default_factory=dict)
    stats: dict[str, object] = field(default_factory=dict)
    age: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "parents": list(self.parents),
            "description": self.description,
            "tags": list(self.tags),
            "primitive": self.primitive,
            "global_refs": list(self.global_refs),
            "has_torch": self.torch_fn is not None,
            "source": self.source,
            "torch_source": self.torch_source,
            "output_contract": copy.deepcopy(self.output_contract),
            "age": int(self.age),
            "stats": copy.deepcopy(self.stats),
        }


@dataclass
class GraphNode:
    name: str
    kind: str
    alternatives: list[NodeAlternative] = field(default_factory=list)
    description: str = ""

    def add_alternative(self, alternative: NodeAlternative) -> None:
        if any(existing.id == alternative.id for existing in self.alternatives):
            raise ValueError(f"Node {self.name!r} already has alternative {alternative.id!r}.")
        self.alternatives.append(alternative)

    def default_alternative_id(self) -> str:
        if not self.alternatives:
            raise ValueError(f"Node {self.name!r} has no alternatives.")
        return self.alternatives[0].id

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
        }


@dataclass
class EvalContext:
    inputs: dict[str, Any]
    globals: GlobalStore
    cache: dict[object, Any] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    key_cache: dict[tuple[str, str], tuple[object, ...]] = field(default_factory=dict)
    alternative_lookup: dict[str, dict[str, NodeAlternative]] | None = None

    def read_input(self, name: str) -> Any:
        if name not in self.inputs:
            raise KeyError(f"Input {name!r} was not supplied.")
        return self.inputs[name]


class Graph:
    def __init__(self, name: str = "evoforest", task_schema: dict[str, object] | None = None) -> None:
        self.name = name
        self.task_schema = copy.deepcopy(task_schema) if task_schema is not None else None
        self.nodes: dict[str, GraphNode] = {}
        self.globals = GlobalStore()

    def add_input(self, name: str, description: str = "") -> None:
        self._add_node(GraphNode(name=name, kind="input", description=description))

    def add_node(self, name: str, kind: str, description: str = "") -> GraphNode:
        node = GraphNode(name=name, kind=kind, description=description)
        self._add_node(node)
        return node

    def add_alternative(
        self,
        node_name: str,
        alternative_id: str,
        parents: tuple[str, ...],
        fn: AlternativeFn,
        description: str = "",
        tags: tuple[str, ...] = (),
        primitive: str | None = None,
        global_refs: tuple[str, ...] = (),
        torch_fn: AlternativeFn | None = None,
        source: str = "",
        torch_source: str = "",
        output_contract: dict[str, object] | None = None,
    ) -> None:
        if node_name not in self.nodes:
            raise KeyError(f"Unknown node {node_name!r}.")
        for parent in parents:
            if parent not in self.nodes:
                raise KeyError(f"Alternative {alternative_id!r} references unknown parent {parent!r}.")
        self.nodes[node_name].add_alternative(
            NodeAlternative(
                id=alternative_id,
                parents=parents,
                fn=fn,
                description=description,
                tags=tags,
                primitive=primitive,
                global_refs=global_refs,
                torch_fn=torch_fn,
                source=source,
                torch_source=torch_source,
                output_contract=copy.deepcopy(output_contract or {}),
            )
        )

    def remove_alternative(self, node_name: str, alternative_id: str) -> NodeAlternative:
        if node_name not in self.nodes:
            raise KeyError(f"Unknown node {node_name!r}.")
        node = self.nodes[node_name]
        for index, alternative in enumerate(node.alternatives):
            if alternative.id == alternative_id:
                return node.alternatives.pop(index)
        raise KeyError(f"Node {node_name!r} has no alternative {alternative_id!r}.")

    def output_nodes(self) -> list[str]:
        return [name for name, node in self.nodes.items() if node.kind == "output"]

    def fitting_nodes(self) -> list[str]:
        return [name for name, node in self.nodes.items() if node.kind == "fitting"]

    def reachable_nodes(self, roots: list[str] | None = None) -> set[str]:
        """Return nodes that can influence output features or fitting rules."""
        if roots is None:
            roots = self.output_nodes() + self.fitting_nodes()
        seen: set[str] = set()

        def visit(name: str) -> None:
            if name in seen:
                return
            seen.add(name)
            node = self.nodes[name]
            for alternative in node.alternatives:
                for parent in alternative.parents:
                    visit(parent)

        for root in roots:
            visit(root)
        return seen

    def referenced_globals(self) -> set[str]:
        refs: set[str] = set()
        reachable = self.reachable_nodes()
        for name in reachable:
            for alternative in self.nodes[name].alternatives:
                refs.update(alternative.global_refs)
        return refs

    def validate_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str, path: tuple[str, ...]) -> None:
            if name in visiting:
                raise ValueError(f"Cycle detected in graph: {' -> '.join((*path, name))}.")
            if name in visited:
                return
            visiting.add(name)
            for alternative in self.nodes[name].alternatives:
                for parent in alternative.parents:
                    if parent not in self.nodes:
                        raise KeyError(f"Alternative {alternative.id!r} references unknown parent {parent!r}.")
                    visit(parent, (*path, name))
            visiting.remove(name)
            visited.add(name)

        for name in self.nodes:
            visit(name, ())

    def configuration_space(self) -> dict[str, list[str]]:
        reachable = self.reachable_nodes()
        return {
            name: [alt.id for alt in node.alternatives]
            for name, node in self.nodes.items()
            if name in reachable and node.kind in {"intermediate", "callable", "fitting"} and node.alternatives
        }

    def default_config(self) -> dict[str, str]:
        reachable = self.reachable_nodes()
        return {
            name: node.default_alternative_id()
            for name, node in self.nodes.items()
            if name in reachable and node.kind in {"intermediate", "callable", "fitting"} and node.alternatives
        }

    def selected_config(self, config: dict[str, str] | None = None) -> dict[str, str]:
        selected = self.default_config()
        if config:
            selected.update(config)
        return selected

    def selected_alternatives(self, config: dict[str, str] | None = None) -> dict[str, str]:
        selected = self.selected_config(config)
        return {
            name: selected[name]
            for name, node in self.nodes.items()
            if node.kind in {"intermediate", "callable", "fitting"} and name in selected
        }

    def update_alternative_statistics(self, rows: list[dict[str, object]]) -> None:
        """Persist quality history for alternatives after a stateful evaluation."""
        for node in self.nodes.values():
            for alternative in node.alternatives:
                alternative.age += 1

        for row in rows:
            node_name, alternative_id = alternative_name_parts(row)
            if not node_name or node_name not in self.nodes:
                continue
            node = self.nodes[node_name]
            try:
                alternative = self._alternative(node, alternative_id)
            except KeyError:
                continue

            stats = alternative.stats
            count = int(stats.get("participation_count", 0)) + 1
            stats["participation_count"] = count
            stats["selected_count"] = int(stats.get("selected_count", 0)) + (1 if bool(row.get("selected", False)) else 0)
            stats["output_count"] = int(stats.get("output_count", 0)) + (1 if node.kind == "output" else 0)
            stats["last_participation_age"] = int(alternative.age)
            stats["last_selected"] = bool(row.get("selected", False))
            stats["kind"] = node.kind

            feature_count = _as_float(row.get("feature_count", 0.0))
            importance = _as_float(row.get("importance", 0.0))
            shap_importance = _as_float(row.get("shap_importance", 0.0))
            mean_abs_shap = _as_float(row.get("mean_abs_shap", 0.0))
            target_alignment_value = _as_float(row.get("max_target_alignment", 0.0))
            residual_corr = _as_float(row.get("mean_abs_residual_corr", 0.0))
            redundancy = _as_float(row.get("mean_redundancy", 0.0))
            weight_stability = _as_float(row.get("mean_weight_stability", 0.0))
            config_score = _as_float(row.get("config_score", 0.0))

            stats["last_feature_count"] = int(feature_count)
            stats["mean_feature_count"] = _running_mean(stats.get("mean_feature_count", 0.0), count, feature_count)
            stats["last_importance"] = importance
            stats["mean_importance"] = _running_mean(stats.get("mean_importance", 0.0), count, importance)
            stats["last_shap_importance"] = shap_importance
            stats["mean_shap_importance"] = _running_mean(stats.get("mean_shap_importance", 0.0), count, shap_importance)
            stats["last_mean_abs_shap"] = mean_abs_shap
            stats["mean_abs_shap"] = _running_mean(stats.get("mean_abs_shap", 0.0), count, mean_abs_shap)
            stats["max_target_alignment"] = max(_as_float(stats.get("max_target_alignment", 0.0)), target_alignment_value)
            stats["last_abs_residual_corr"] = residual_corr
            stats["mean_abs_residual_corr"] = _running_mean(stats.get("mean_abs_residual_corr", 0.0), count, residual_corr)
            stats["last_redundancy"] = redundancy
            stats["mean_redundancy"] = _running_mean(stats.get("mean_redundancy", 0.0), count, redundancy)
            stats["last_weight_stability"] = weight_stability
            stats["mean_weight_stability"] = _running_mean(stats.get("mean_weight_stability", 0.0), count, weight_stability)
            stats["last_config_score"] = config_score
            stats["mean_config_score"] = _running_mean(stats.get("mean_config_score", 0.0), count, config_score)
            stats["best_config_score"] = max(_as_float(stats.get("best_config_score", 0.0)), config_score)

    def alternative_statistics_snapshot(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for node_name, node in self.nodes.items():
            for alternative in node.alternatives:
                row = {
                    "name": f"{node_name}.{alternative.id}",
                    "node": node_name,
                    "alternative": alternative.id,
                    "kind": node.kind,
                    "age": int(alternative.age),
                }
                row.update(copy.deepcopy(alternative.stats))
                rows.append(row)
        return sorted(
            rows,
            key=lambda row: (
                -_as_float(row.get("mean_importance", row.get("last_importance", 0.0))),
                -_as_float(row.get("max_target_alignment", 0.0)),
                str(row.get("name", "")),
            ),
        )

    def alternative_dependencies(self, node_name: str, alternative_id: str, config: dict[str, str] | None = None) -> list[str]:
        selected = self.selected_config(config)
        seen: set[tuple[str, str]] = set()
        ordered: list[str] = []

        def visit(name: str, alt_id: str) -> None:
            node = self.nodes[name]
            if node.kind == "input":
                return
            key = (name, alt_id)
            if key in seen:
                return
            seen.add(key)
            ordered.append(f"{name}.{alt_id}")
            alternative = self._alternative(node, alt_id)
            for parent in alternative.parents:
                parent_node = self.nodes[parent]
                if parent_node.kind == "input":
                    continue
                visit(parent, selected.get(parent, parent_node.default_alternative_id()))

        visit(node_name, alternative_id)
        return ordered

    def output_dependency_map(self, config: dict[str, str] | None = None) -> dict[str, list[str]]:
        selected = self.selected_config(config)
        dependencies: dict[str, list[str]] = {}
        for node_name in self.output_nodes():
            output = self.nodes[node_name]
            for alternative in output.alternatives:
                prefix = f"{node_name}.{alternative.id}"
                dependencies[prefix] = self.alternative_dependencies(node_name, alternative.id, selected)
        return dependencies

    def evaluate_node(self, name: str, config: dict[str, str], ctx: EvalContext) -> Any:
        node = self.nodes[name]
        if node.kind == "input":
            return ctx.read_input(name)
        alt_id = config.get(name, node.default_alternative_id())
        return self.evaluate_alternative(name, alt_id, config, ctx)

    def evaluate_alternative(self, name: str, alternative_id: str, config: dict[str, str], ctx: EvalContext) -> Any:
        node = self.nodes[name]
        key = self.alternative_cache_key(name, alternative_id, config, memo=ctx.key_cache, alternative_lookup=ctx.alternative_lookup)
        if key in ctx.cache:
            ctx.cache_hits += 1
            return ctx.cache[key]
        alternative = self._lookup_alternative(name, alternative_id, ctx.alternative_lookup, node=node)
        parent_values = {parent: self.evaluate_node(parent, config, ctx) for parent in alternative.parents}
        ctx.cache_misses += 1
        value = alternative.fn(ctx, parent_values)
        ctx.cache[key] = value
        return value

    def alternative_cache_key(
        self,
        node_name: str,
        alternative_id: str,
        config: dict[str, str],
        *,
        memo: dict[tuple[str, str], tuple[object, ...]] | None = None,
        alternative_lookup: dict[str, dict[str, NodeAlternative]] | None = None,
    ) -> tuple[object, ...]:
        memo_key = (node_name, alternative_id)
        if memo is not None and memo_key in memo:
            return memo[memo_key]
        node = self.nodes[node_name]
        alternative = self._lookup_alternative(node_name, alternative_id, alternative_lookup, node=node)
        parent_keys: list[object] = []
        for parent in alternative.parents:
            parent_node = self.nodes[parent]
            if parent_node.kind == "input":
                parent_keys.append(("input", parent))
                continue
            parent_alt_id = config.get(parent, parent_node.default_alternative_id())
            parent_keys.append(self.alternative_cache_key(parent, parent_alt_id, config, memo=memo, alternative_lookup=alternative_lookup))
        key = (node_name, alternative_id, tuple(parent_keys))
        if memo is not None:
            memo[memo_key] = key
        return key

    def evaluate_features(
        self,
        inputs: dict[str, Any],
        config: dict[str, str] | None = None,
        cache: dict[object, Any] | None = None,
    ) -> tuple[np.ndarray, list[str], EvalContext]:
        selected = self.selected_config(config)
        ctx = EvalContext(
            inputs=inputs,
            globals=self.globals.clone(),
            cache=cache if cache is not None else {},
            alternative_lookup=self._alternative_lookup(),
        )
        blocks: list[FeatureBlock] = []
        for node_name in self.output_nodes():
            output = self.nodes[node_name]
            for alternative in output.alternatives:
                value = self.evaluate_alternative(node_name, alternative.id, selected, ctx)
                blocks.append(as_feature_block(value, prefix=f"{node_name}.{alternative.id}"))
        if not blocks:
            raise ValueError("Graph has no output nodes.")
        values = blocks[0].values if len(blocks) == 1 else np.column_stack([block.values for block in blocks])
        names = [name for block in blocks for name in block.names]
        return values, names, ctx

    def clone(self) -> "Graph":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "task_schema": copy.deepcopy(self.task_schema),
            "globals": self.globals.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes.values()],
        }

    def _add_node(self, node: GraphNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"Graph already has a node named {node.name!r}.")
        self.nodes[node.name] = node

    @staticmethod
    def _alternative(node: GraphNode, alternative_id: str) -> NodeAlternative:
        for alternative in node.alternatives:
            if alternative.id == alternative_id:
                return alternative
        raise KeyError(f"Node {node.name!r} has no alternative {alternative_id!r}.")

    def _alternative_lookup(self) -> dict[str, dict[str, NodeAlternative]]:
        return {name: {alternative.id: alternative for alternative in node.alternatives} for name, node in self.nodes.items()}

    def _lookup_alternative(
        self,
        node_name: str,
        alternative_id: str,
        alternative_lookup: dict[str, dict[str, NodeAlternative]] | None,
        *,
        node: GraphNode | None = None,
    ) -> NodeAlternative:
        if alternative_lookup is not None:
            try:
                return alternative_lookup[node_name][alternative_id]
            except KeyError:
                pass
        return self._alternative(node or self.nodes[node_name], alternative_id)


def as_feature_block(value: Any, prefix: str) -> FeatureBlock:
    if isinstance(value, FeatureBlock):
        return FeatureBlock(
            values=value.values,
            names=[name if name.startswith(f"{prefix}.") else f"{prefix}.{name}" for name in value.names],
        )
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        names = [prefix]
    elif array.ndim == 2:
        names = [f"{prefix}.{idx}" for idx in range(array.shape[1])]
    else:
        raise ValueError(f"Cannot convert value with shape {array.shape} to FeatureBlock.")
    return FeatureBlock(values=array, names=names)


def alternative_name_parts(row: dict[str, object]) -> tuple[str, str]:
    node = str(row.get("node", ""))
    alternative = str(row.get("alternative", ""))
    if node and alternative:
        return node, alternative
    name = str(row.get("name", ""))
    if "." not in name:
        return "", ""
    return name.split(".", maxsplit=1)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _running_mean(previous: object, count: int, value: float) -> float:
    if count <= 1:
        return float(value)
    return (_as_float(previous) * float(count - 1) + float(value)) / float(count)
