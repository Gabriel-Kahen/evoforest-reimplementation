from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Protocol, Sequence
import urllib.error
import urllib.request

import numpy as np

from evoforest_arch.agents import EngineerAgent, Hypothesis, ScientistAgent
from evoforest_arch.evaluator import EvaluationResult
from evoforest_arch.feedback import feedback_summary, toon_report
from evoforest_arch.graph import Graph
from evoforest_arch.mutations import MutationDocument
from evoforest_arch.primitives import PrimitiveRegistry


DEFAULT_ISLAND_TEMPERATURES = (0.35, 0.5, 0.6, 0.75)


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        """Return one text completion for a system/user prompt pair."""


@dataclass(frozen=True)
class PromptRecord:
    stage: str
    system_prompt: str
    user_prompt: str
    response: str = ""
    error: str = ""

    def to_text(self) -> str:
        parts = [
            f"# {self.stage} prompt",
            "",
            "## System",
            "```text",
            self.system_prompt,
            "```",
            "",
            "## User",
            "```text",
            self.user_prompt,
            "```",
        ]
        if self.response:
            parts.extend(["", "## Response", "```text", self.response, "```"])
        if self.error:
            parts.extend(["", "## Error", "```text", self.error, "```"])
        return "\n".join(parts) + "\n"


class StaticLLMClient:
    """Deterministic client for tests and prompt-pipeline dry runs."""

    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.requests.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": float(temperature),
            }
        )
        if not self.responses:
            raise RuntimeError("StaticLLMClient has no remaining responses.")
        return self.responses.pop(0)


@dataclass(frozen=True)
class HTTPJSONLLMClient:
    """Small generic JSON-over-HTTP client for OpenAI-compatible chat servers."""

    url: str
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "HTTPJSONLLMClient":
        url = os.getenv("EVOFOREST_LLM_URL")
        if not url:
            raise ValueError("Set EVOFOREST_LLM_URL before using --llm-provider http-json.")
        return cls(
            url=url,
            api_key=os.getenv("EVOFOREST_LLM_API_KEY"),
            model=os.getenv("EVOFOREST_LLM_MODEL"),
        )

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        payload: dict[str, object] = {
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.model:
            payload["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP request failed with status {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM HTTP request failed: {exc}") from exc
        return _extract_llm_content(json.loads(body))


@dataclass
class PromptBuilder:
    task_context: str = (
        "Clean-room EvoForest reimplementation for supervised time-series "
        "experiments. The executable mutation layer accepts registry-backed primitives, "
        "optional trusted source-backed alternatives, and structured mutation documents."
    )
    max_graph_chars: int = 18000
    registry: PrimitiveRegistry | None = None
    allow_source: bool = False

    def scientist_prompts(
        self,
        graph: Graph,
        result: EvaluationResult,
        *,
        memorandum: str = "",
        step: int = 0,
        island: int | None = None,
        max_hypotheses: int = 8,
    ) -> tuple[str, str]:
        system = self.global_rules()
        user = "\n".join(
            [
                f"Generate {max_hypotheses}-12 structured actionable hypotheses for the next step.",
                "Do not output code or YAML.",
                "Each hypothesis should include Hypothesis, Rationale, Expected Improvement, Risk Mode, and a brief self-evaluation.",
                f"Step: {step}",
                f"Island: {island if island is not None else 'single'}",
                "",
                "==== TASK CONTEXT ====",
                self.task_context,
                "",
                "==== CURRENT EVOFOREST (JSON WITH STATS) ====",
                self.annotated_graph(graph, result),
                "",
                "==== FEATURE DIAGNOSTICS (TOON - PRIMARY EVIDENCE) ====",
                toon_report(result),
                "",
                "==== EXPERIMENT LOG (MEMORANDUM) ====",
                memorandum.strip() or "(no memorandum yet)",
                "",
                "Use the diagnostics as evidence. Prefer structural, callable, fitting-rule, global-parameter, or pruning hypotheses.",
            ]
        )
        return system, user

    def engineer_prompts(
        self,
        graph: Graph,
        result: EvaluationResult,
        hypotheses: tuple[Hypothesis, ...],
        *,
        memorandum: str = "",
        execution_errors: str = "",
        step: int = 0,
        island: int | None = None,
    ) -> tuple[str, str]:
        hypothesis_text = "\n".join(f"- {item.to_text()}" for item in hypotheses) or "- No hypotheses supplied."
        system = "\n".join(
            [
                self.global_rules(),
                "",
                "==== SCIENTIST HYPOTHESES ====",
                hypothesis_text,
            ]
        )
        schema_lines = [
            "Act as the EvoForest engineer. Select, combine, or reject the scientist hypotheses and emit one structured mutation document.",
            "Output YAML only. Do not wrap the YAML in explanations.",
            "Use only the schema below; list items are JSON objects so the document remains machine-parseable.",
            "",
            "rationale: \"why this mutation is coherent\"",
            "hypotheses:",
            "  - \"short hypothesis text\"",
            "nodes:",
            "  - {\"name\": \"optional_new_node\", \"kind\": \"intermediate\", \"description\": \"what it computes\"}",
            "remove:",
            "  - {\"target_node\": \"node\", \"alternative_id\": \"alt\", \"reason\": \"why it is safe to remove\"}",
            "globals:",
            "  - {\"name\": \"new_global\", \"value\": [0.1], \"trainable\": true, \"description\": \"append-only parameter\"}",
            "add:",
            "  - {\"kind\": \"add_alternative\", \"target_node\": \"node\", \"primitive\": \"primitive_name\", \"alternative_id\": \"unique_alt_id\", \"parents\": [\"parent_node\"], \"description\": \"what changes\"}",
        ]
        if self.allow_source:
            schema_lines.extend(
                [
                    "",
                    "For trusted source-backed alternatives, use primitive \"source\" and include:",
                    "  \"source\": \"lambda ctx, values: <expression>\"",
                    "  \"global_refs\": [\"optional_existing_or_new_global\"]",
                    "The lambda receives ctx and values, where values maps parent node names to evaluated parent outputs.",
                ]
            )
        user = "\n".join(
            schema_lines
            + [
                "",
                "Available primitive names:",
                ", ".join(sorted((self.registry or PrimitiveRegistry.default()).factories)),
                "",
                "Allowed new node kinds: intermediate, callable, output, fitting.",
                "Existing node names and newly declared nodes may be used as mutation targets or parents.",
                "Keep the DAG valid, keep globals append-only, and avoid adding broad branching unless diagnostics justify it.",
                f"Step: {step}",
                f"Island: {island if island is not None else 'single'}",
                "",
                "==== CURRENT EVOFOREST (JSON WITH STATS) ====",
                self.annotated_graph(graph, result),
                "",
                "==== EXECUTION ERRORS FROM PREVIOUS ATTEMPTS ====",
                execution_errors.strip() or "(none)",
                "",
                "==== EXPERIMENT LOG (MEMORANDUM, BACKGROUND ONLY) ====",
                memorandum.strip() or "(no memorandum yet)",
            ]
        )
        return system, user

    def global_rules(self) -> str:
        return "\n".join(
            [
                "GLOBAL EVOFOREST RULES",
                "You are evolving a directed acyclic graph of reusable computations.",
                "Intermediate and callable nodes contain competing alternatives; a configuration selects one alternative per reachable such node.",
                "The output node is different: all output alternatives are evaluated as features for each configuration.",
                "Fitting nodes are selected by configuration and may alter Ridge sample weights or iterative residual reweighting.",
                "Score is the best configuration's stratified Ridge cross-validation ROC-AUC.",
                "Evaluation caches deterministic alternatives by selected ancestor subpath, so shared intermediates can be reused across configurations.",
                "The @globals store contains persistent trainable parameters. New globals are append-only at mutation time; unused globals may be pruned by maintenance.",
                "TOON diagnostics include exact additive Ridge contribution fields: shap is the normalized global linear contribution, and cv_shap is the out-of-fold contribution magnitude.",
                "Prefer complementary features, productive bottleneck expansion, useful callable reuse, and removal of redundant or dead structure.",
                "Respect DAG validity and computational cost. Adding intermediate alternatives multiplies configurations; adding output alternatives increases feature count.",
            ]
        )

    def annotated_graph(self, graph: Graph, result: EvaluationResult) -> str:
        payload = {
            "graph": graph.to_dict(),
            "configuration_space": graph.configuration_space(),
            "best_config": result.config,
            "feedback": feedback_summary(result),
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        if len(text) <= self.max_graph_chars:
            return text
        return text[: self.max_graph_chars] + "\n... [truncated]"


class LLMScientistAgent(ScientistAgent):
    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_builder: PromptBuilder | None = None,
        fallback: ScientistAgent | None = None,
        temperature: float = 0.35,
        island_temperatures: Sequence[float] | None = DEFAULT_ISLAND_TEMPERATURES,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.fallback = fallback or ScientistAgent()
        self.temperature = float(temperature)
        self.island_temperatures = (
            tuple(float(item) for item in island_temperatures)
            if island_temperatures is not None
            else ()
        )
        self._prompt_records: list[PromptRecord] = []

    def generate(
        self,
        graph: Graph,
        result: EvaluationResult,
        max_hypotheses: int = 8,
        *,
        step: int = 0,
        island: int | None = None,
        memorandum: str = "",
    ) -> tuple[Hypothesis, ...]:
        system, user = self.prompt_builder.scientist_prompts(
            graph,
            result,
            memorandum=memorandum,
            step=step,
            island=island,
            max_hypotheses=max_hypotheses,
        )
        response = ""
        error = ""
        try:
            response = self.client.complete(system, user, temperature=self._temperature_for(island))
            hypotheses = parse_hypotheses(response, graph, max_hypotheses)
            if not hypotheses:
                raise ValueError("LLM scientist returned no parseable hypotheses.")
            return hypotheses
        except Exception as exc:
            error = str(exc)
            return self.fallback.generate(graph, result, max_hypotheses=max_hypotheses)
        finally:
            self._prompt_records.append(PromptRecord("scientist", system, user, response, error))

    def _temperature_for(self, island: int | None) -> float:
        if island is None or not self.island_temperatures:
            return self.temperature
        return self.island_temperatures[island % len(self.island_temperatures)]

    def pop_prompt_records(self) -> tuple[PromptRecord, ...]:
        records = tuple(self._prompt_records)
        self._prompt_records.clear()
        return records


class LLMEngineerAgent(EngineerAgent):
    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_builder: PromptBuilder | None = None,
        fallback: EngineerAgent | None = None,
        registry: PrimitiveRegistry | None = None,
        temperature: float = 0.0,
        allow_source: bool = False,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder(allow_source=allow_source)
        self.fallback = fallback or EngineerAgent()
        self.registry = registry or PrimitiveRegistry.default()
        self.temperature = float(temperature)
        self.allow_source = bool(allow_source)
        self.templates = self.fallback.templates
        self._prompt_records: list[PromptRecord] = []

    def synthesize(
        self,
        graph: Graph,
        result: EvaluationResult,
        hypotheses: tuple[Hypothesis, ...],
        step: int,
        island: int | None,
        rng: np.random.Generator,
        *,
        memorandum: str = "",
        execution_errors: str = "",
    ) -> MutationDocument:
        system, user = self.prompt_builder.engineer_prompts(
            graph,
            result,
            hypotheses,
            memorandum=memorandum,
            execution_errors=execution_errors,
            step=step,
            island=island,
        )
        response = ""
        error = ""
        try:
            response = self.client.complete(system, user, temperature=self.temperature)
            document = MutationDocument.from_yaml(response)
            document = self._fill_missing_fields(document, hypotheses)
            self._validate_supported_document(graph, document)
            if not (document.nodes or document.add or document.remove or document.globals):
                raise ValueError("LLM engineer returned an empty mutation document.")
            return document
        except Exception as exc:
            error = str(exc)
            return self.fallback.synthesize(graph, result, hypotheses, step, island, rng)
        finally:
            self._prompt_records.append(PromptRecord("engineer", system, user, response, error))

    def pop_prompt_records(self) -> tuple[PromptRecord, ...]:
        records = tuple(self._prompt_records)
        self._prompt_records.clear()
        return records

    @staticmethod
    def _fill_missing_fields(document: MutationDocument, hypotheses: tuple[Hypothesis, ...]) -> MutationDocument:
        return MutationDocument(
            hypotheses=document.hypotheses or tuple(item.to_text() for item in hypotheses),
            rationale=document.rationale or "LLM engineer selected a structured mutation from scientist hypotheses.",
            nodes=document.nodes,
            add=document.add,
            remove=document.remove,
            globals=document.globals,
        )

    def _validate_supported_document(self, graph: Graph, document: MutationDocument) -> None:
        known_nodes = set(graph.nodes)
        new_nodes = set()
        allowed_kinds = {"intermediate", "callable", "output", "fitting"}
        for node in document.nodes:
            if node.name in known_nodes or node.name in new_nodes:
                raise ValueError(f"Mutation declares duplicate node {node.name!r}.")
            if node.kind not in allowed_kinds:
                raise ValueError(f"Unsupported node kind {node.kind!r}.")
            new_nodes.add(node.name)
        all_nodes = known_nodes | new_nodes
        for remove in document.remove:
            if remove.target_node not in known_nodes:
                raise ValueError(f"Removal target node {remove.target_node!r} does not exist.")
            alternatives = graph.nodes[remove.target_node].alternatives
            if not any(alternative.id == remove.alternative_id for alternative in alternatives):
                raise ValueError(f"Removal target alternative {remove.target_node}.{remove.alternative_id} does not exist.")
        for global_spec in document.globals:
            if global_spec.name in graph.globals.names():
                raise ValueError(f"Global {global_spec.name!r} already exists.")
        for add in document.add:
            if add.kind != "add_alternative":
                raise ValueError(f"Unsupported mutation kind {add.kind!r}.")
            if add.target_node not in all_nodes:
                raise ValueError(f"Add target node {add.target_node!r} does not exist.")
            if add.source:
                if not self.allow_source:
                    raise ValueError("Source-backed LLM mutations are disabled for this engineer agent.")
            elif add.primitive not in self.registry.factories:
                raise ValueError(f"Unknown primitive {add.primitive!r}.")
            for parent in add.parents:
                if parent not in all_nodes:
                    raise ValueError(f"Parent node {parent!r} does not exist.")


def parse_hypotheses(text: str, graph: Graph, max_hypotheses: int) -> tuple[Hypothesis, ...]:
    blocks = _hypothesis_blocks(text)
    hypotheses: list[Hypothesis] = []
    for block in blocks:
        clean = _clean_hypothesis_text(block)
        if not clean:
            continue
        target = _infer_target_node(clean, graph)
        hypotheses.append(
            Hypothesis(
                kind=_field(clean, "Hypothesis") or "llm_structural_hypothesis",
                target_node=target,
                rationale=_field(clean, "Rationale") or clean,
                expected_improvement=_field(clean, "Expected Improvement") or "improved complementary structure or pruning",
                risk=_field(clean, "Risk Mode") or _field(clean, "Risk") or "Balanced",
            )
        )
        if len(hypotheses) >= max_hypotheses:
            break
    return tuple(hypotheses)


def _hypothesis_blocks(text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n|(?=\n\s*(?:[-*]|\d+[.)])\s+Hypothesis\s*:)", text.strip())
    if len(chunks) > 1:
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _clean_hypothesis_text(text: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text.strip())


def _field(text: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}\s*:\s*(.+?)(?=\n[A-Z][A-Za-z ]+\s*:|\Z)", text, flags=re.S)
    return match.group(1).strip(" \n-") if match else ""


def _infer_target_node(text: str, graph: Graph) -> str:
    lowered = text.lower()
    for name in sorted(graph.nodes, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            return name
    return "output" if "output" in graph.nodes else next(iter(graph.nodes))


def _extract_llm_content(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object.")
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and "content" in message:
                return _stringify_content(message["content"])
            if "text" in first:
                return _stringify_content(first["text"])
    if "content" in data:
        return _stringify_content(data["content"])
    output = data.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))
            elif content is not None:
                parts.append(str(content))
        if parts:
            return "\n".join(parts)
    raise ValueError("Could not find text content in LLM response JSON.")


def _stringify_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)
