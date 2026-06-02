from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evoforest_arch.evaluator import EvaluationResult
from evoforest_arch.feedback import feedback_summary
from evoforest_arch.graph import Graph
from evoforest_arch.mutations import MutationDocument, MutationSpec, RemoveSpec, built_in_mutations


@dataclass(frozen=True)
class Hypothesis:
    kind: str
    target_node: str
    rationale: str
    expected_improvement: str
    risk: str = "Balanced"

    def to_text(self) -> str:
        return (
            f"{self.kind} on {self.target_node}: {self.rationale} "
            f"Expected: {self.expected_improvement}. Risk: {self.risk}."
        )


class ScientistAgent:
    """Deterministic analogue of the paper's diagnostic-grounded scientist role."""

    def generate(self, graph: Graph, result: EvaluationResult, max_hypotheses: int = 4) -> tuple[Hypothesis, ...]:
        feedback = feedback_summary(result)
        hypotheses: list[Hypothesis] = []
        subnodes = feedback.get("top_subnodes", [])
        if subnodes:
            top = subnodes[0]
            target = str(top.get("name", "")).split(".", maxsplit=1)[0]
            if target in graph.nodes and graph.nodes[target].kind in {"intermediate", "callable"}:
                hypotheses.append(
                    Hypothesis(
                        kind="expand_productive_bottleneck",
                        target_node=target,
                        rationale=f"{top.get('name')} carries high aggregate importance {float(top.get('importance', 0.0)):.4f}.",
                        expected_improvement="more competing implementations around a productive subpath",
                        risk="Conservative",
                    )
                )

        strongest_residual = self._strongest_residual_feature(result)
        if strongest_residual:
            hypotheses.append(
                Hypothesis(
                    kind="residual_feature_search",
                    target_node="output",
                    rationale=f"{strongest_residual['name']} has residual correlation {float(strongest_residual.get('residual_corr', 0.0)):.4f}.",
                    expected_improvement="a complementary output feature that explains remaining residual structure",
                    risk="Balanced",
                )
            )

        fitting = result.diagnostics.get("fitting", {})
        ridge_g = fitting.get("ridge_g", {}) if isinstance(fitting, dict) else {}
        if isinstance(ridge_g, dict) and ridge_g.get("alternative") == "identity":
            hypotheses.append(
                Hypothesis(
                    kind="robust_fitting_rule",
                    target_node="ridge_g",
                    rationale="The selected residual rule is identity, leaving robust fitting unexplored.",
                    expected_improvement="better handling of heavy-tailed residuals",
                    risk="Conservative",
                )
            )

        underexplored = self._underexplored_node(graph)
        if underexplored:
            hypotheses.append(
                Hypothesis(
                    kind="diversify_underexplored_node",
                    target_node=underexplored,
                    rationale=f"{underexplored} has fewer alternatives than neighboring search axes.",
                    expected_improvement="broader configuration diversity",
                    risk="Balanced",
                )
            )

        if not hypotheses:
            hypotheses.append(
                Hypothesis(
                    kind="output_diversity",
                    target_node="output",
                    rationale="No single diagnostic dominates, so add a low-cost output alternative.",
                    expected_improvement="feature ensemble diversity without multiplying configurations",
                    risk="Conservative",
                )
            )
        return tuple(hypotheses[:max_hypotheses])

    @staticmethod
    def _strongest_residual_feature(result: EvaluationResult) -> dict[str, object] | None:
        features = result.diagnostics.get("features", [])
        if not features:
            return None
        return max(features, key=lambda row: abs(float(row.get("residual_corr", 0.0))))

    @staticmethod
    def _underexplored_node(graph: Graph) -> str | None:
        candidates = [
            (name, len(node.alternatives))
            for name, node in graph.nodes.items()
            if node.kind in {"intermediate", "callable", "fitting"} and node.alternatives
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[1], item[0]))[0]


class EngineerAgent:
    """Deterministic analogue of the paper's YAML-producing engineer role."""

    def __init__(self, templates: tuple[MutationSpec, ...] | None = None) -> None:
        self.templates = templates or tuple(built_in_mutations())

    def synthesize(
        self,
        graph: Graph,
        result: EvaluationResult,
        hypotheses: tuple[Hypothesis, ...],
        step: int,
        island: int | None,
        rng: np.random.Generator,
    ) -> MutationDocument:
        del result
        selected = self._select_template(graph, hypotheses, step, rng)
        suffix = f"{step}" if island is None else f"i{island}_{step}"
        spec = MutationSpec(
            kind=selected.kind,
            target_node=selected.target_node,
            primitive=selected.primitive,
            alternative_id=f"{selected.alternative_id}_{suffix}",
            parents=selected.parents,
            description=selected.description,
            source=selected.source,
            global_refs=selected.global_refs,
        )
        removals = self._safe_redundancy_removals(graph)
        return MutationDocument(
            hypotheses=tuple(hypothesis.to_text() for hypothesis in hypotheses),
            rationale=self._rationale(selected, hypotheses),
            remove=tuple(removals),
            add=(spec,),
        )

    def _select_template(
        self,
        graph: Graph,
        hypotheses: tuple[Hypothesis, ...],
        step: int,
        rng: np.random.Generator,
    ) -> MutationSpec:
        target_order = [hypothesis.target_node for hypothesis in hypotheses]
        target_rank = {target: index for index, target in enumerate(target_order)}
        candidates = [
            template
            for template in self.templates
            if template.target_node in graph.nodes and not self._template_already_present(graph, template)
        ]
        if not candidates:
            candidates = [template for template in self.templates if template.target_node in graph.nodes]
        if not candidates:
            return self.templates[int(rng.integers(0, len(self.templates)))]
        candidates = sorted(
            candidates,
            key=lambda template: (
                target_rank.get(template.target_node, len(target_rank) + 1),
                0 if template.source else 1,
                template.target_node,
                template.primitive,
                template.alternative_id,
            ),
        )
        return candidates[(max(1, int(step)) - 1) % len(candidates)]

    @staticmethod
    def _template_already_present(graph: Graph, template: MutationSpec) -> bool:
        node = graph.nodes.get(template.target_node)
        if node is None:
            return False
        for alternative in node.alternatives:
            if template.source:
                if alternative.source.strip() == template.source.strip():
                    return True
                continue
            if alternative.primitive == template.primitive and alternative.parents == template.parents:
                return True
        return False

    @staticmethod
    def _safe_redundancy_removals(graph: Graph) -> list[RemoveSpec]:
        removals: list[RemoveSpec] = []
        output_nodes = graph.output_nodes()
        for node_name in output_nodes:
            node = graph.nodes[node_name]
            if len(node.alternatives) > 8:
                removals.append(
                    RemoveSpec(
                        target_node=node_name,
                        alternative_id=node.alternatives[-1].id,
                        reason="Conservative cap on output ensemble growth.",
                    )
                )
        return removals

    @staticmethod
    def _rationale(template: MutationSpec, hypotheses: tuple[Hypothesis, ...]) -> str:
        if hypotheses:
            return f"Engineer selected {template.target_node}.{template.primitive} from {len(hypotheses)} diagnostic hypotheses."
        return template.description
