from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
import pathlib
import re
import sys
import time
from typing import Protocol, Sequence
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from evoforest_arch.agents import EngineerAgent, Hypothesis, ScientistAgent
from evoforest_arch.evaluator import EvaluationResult
from evoforest_arch.feedback import feedback_summary, toon_report
from evoforest_arch.graph import Graph
from evoforest_arch.mutations import MutationDocument, RemoveSpec
from evoforest_arch.primitives import PrimitiveRegistry


DEFAULT_ISLAND_TEMPERATURES = (0.35, 0.5, 0.6, 0.75)
DEFAULT_ENV_FILE = pathlib.Path(".env")
SUPPORTED_LLM_PROVIDERS = ("openai", "claude", "gemini")
RETRYABLE_LLM_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class LLMRetryConfig:
    max_retries: int = 5
    initial_delay_seconds: float = 5.0
    max_delay_seconds: float = 120.0


@dataclass(frozen=True)
class LLMStructuredOutputRetryConfig:
    max_retries: int = 3


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


def load_env_file(path: str | pathlib.Path | None = DEFAULT_ENV_FILE, *, override: bool = False) -> bool:
    """Load simple KEY=VALUE entries from a dotenv file into os.environ."""
    if path is None:
        return False
    env_path = pathlib.Path(path)
    if not env_path.exists():
        return False
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, raw_value = line.partition("=")
        if not separator:
            raise ValueError(f"Invalid dotenv line {line_number} in {env_path}: expected KEY=VALUE.")
        key = key.strip()
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ValueError(f"Invalid dotenv key {key!r} on line {line_number} in {env_path}.")
        if not override and key in os.environ:
            continue
        os.environ[key] = _parse_env_value(raw_value)
    return True


def llm_provider_from_env(path: str | pathlib.Path | None = DEFAULT_ENV_FILE, *, required: bool = False) -> str | None:
    load_env_file(path)
    provider = os.getenv("EVOFOREST_LLM_PROVIDER", "").strip()
    if not provider:
        if required:
            raise ValueError(
                "Set EVOFOREST_LLM_PROVIDER in the environment or .env file when --llm-provider env is used."
            )
        return None
    normalized = _normalize_provider(provider)
    if normalized not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported EVOFOREST_LLM_PROVIDER={provider!r}; expected one of {', '.join(SUPPORTED_LLM_PROVIDERS)}."
        )
    return normalized


def llm_client_from_env(env_file: str | pathlib.Path | None = DEFAULT_ENV_FILE) -> LLMClient:
    provider = llm_provider_from_env(env_file, required=True)
    if provider == "openai":
        return OpenAILLMClient.from_env(env_file)
    if provider == "claude":
        return ClaudeLLMClient.from_env(env_file)
    if provider == "gemini":
        return GeminiLLMClient.from_env(env_file)
    raise ValueError(f"Unsupported LLM provider {provider!r}.")


@dataclass(frozen=True)
class OpenAILLMClient:
    """OpenAI chat-completions client."""

    api_key: str
    model: str
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, env_file: str | pathlib.Path | None = DEFAULT_ENV_FILE) -> "OpenAILLMClient":
        load_env_file(env_file)
        return cls(
            api_key=_required_env(("OPENAI_API_KEY", "EVOFOREST_OPENAI_API_KEY"), "openai"),
            model=_required_env(("EVOFOREST_LLM_MODEL", "EVOFOREST_OPENAI_MODEL"), "openai"),
            timeout_seconds=_env_float("EVOFOREST_LLM_TIMEOUT_SECONDS", 120.0),
        )

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        max_tokens = _optional_env_int("EVOFOREST_LLM_MAX_TOKENS")
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        data = _post_json(
            "https://api.openai.com/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds=self.timeout_seconds,
        )
        return _extract_openai_content(data)


@dataclass(frozen=True)
class ClaudeLLMClient:
    """Anthropic Claude Messages API client."""

    api_key: str
    model: str
    timeout_seconds: float = 120.0
    max_tokens: int = 4096

    @classmethod
    def from_env(cls, env_file: str | pathlib.Path | None = DEFAULT_ENV_FILE) -> "ClaudeLLMClient":
        load_env_file(env_file)
        return cls(
            api_key=_required_env(("ANTHROPIC_API_KEY", "EVOFOREST_CLAUDE_API_KEY"), "claude"),
            model=_required_env(("EVOFOREST_LLM_MODEL", "EVOFOREST_CLAUDE_MODEL"), "claude"),
            timeout_seconds=_env_float("EVOFOREST_LLM_TIMEOUT_SECONDS", 120.0),
            max_tokens=_env_int("EVOFOREST_LLM_MAX_TOKENS", 4096),
        )

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": float(temperature),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout_seconds=self.timeout_seconds,
        )
        return _extract_claude_content(data)


@dataclass(frozen=True)
class GeminiLLMClient:
    """Google Gemini generateContent client."""

    api_key: str
    model: str
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, env_file: str | pathlib.Path | None = DEFAULT_ENV_FILE) -> "GeminiLLMClient":
        load_env_file(env_file)
        return cls(
            api_key=_required_env(("GEMINI_API_KEY", "GOOGLE_API_KEY", "EVOFOREST_GEMINI_API_KEY"), "gemini"),
            model=_required_env(("EVOFOREST_LLM_MODEL", "EVOFOREST_GEMINI_MODEL"), "gemini"),
            timeout_seconds=_env_float("EVOFOREST_LLM_TIMEOUT_SECONDS", 120.0),
        )

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        payload: dict[str, object] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": float(temperature)},
        }
        max_tokens = _optional_env_int("EVOFOREST_LLM_MAX_TOKENS")
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens  # type: ignore[index]
        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"
        quoted_model = urllib.parse.quote(model_path, safe="/")
        query = urllib.parse.urlencode({"key": self.api_key})
        data = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/{quoted_model}:generateContent?{query}",
            payload,
            timeout_seconds=self.timeout_seconds,
        )
        return _extract_gemini_content(data)


@dataclass
class PromptBuilder:
    task_context: str = (
        "Clean-room EvoForest reimplementation for supervised tasks. "
        "The executable mutation layer accepts registry-backed primitives, "
        "sandboxed source-backed alternatives, and structured mutation documents."
    )
    max_graph_chars: int = 18000
    registry: PrimitiveRegistry | None = None
    allow_source: bool = True

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
        system = "\n".join(
            [
                "You are an elite computational scientist and algorithm designer specializing in algorithm evolution, representation learning, and structural optimization.",
                "Analyze EvoForest graphs, diagnose bottlenecks, and propose high-value structural directions rather than incremental tweaks.",
                "",
                "===================== TASK CONTEXT =====================",
                self.task_context,
                "========================================================",
                "",
                self.global_rules(),
            ]
        )
        user = "\n".join(
            [
                f"Generate {max_hypotheses}-12 structured actionable hypotheses for the next step.",
                "Do not output code or YAML.",
                "Each hypothesis must include Hypothesis, Rationale, Expected Improvement, Risk Mode, and Self-Evaluation scores for Improvement, Creativity, Implementability, and Risk.",
                f"Step: {step}",
                f"Island: {island if island is not None else 'single'}",
                "",
                "==== CURRENT EVOFOREST (YAML WITH STATS) ====",
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
            "Act as the EvoForest engineer. Select one compact graph edit from the scientist hypotheses and emit one structured mutation document.",
            "Output YAML only. Do not wrap the YAML in explanations, analysis, markdown fences, or scratch work.",
            "The first non-whitespace text in your response must be 'rationale:'.",
            "The document must include at least one non-empty add, remove, nodes, or globals entry.",
            "Prefer registry-backed primitives from the available primitive list; use source-backed lambdas only when no listed primitive can express the edit.",
            "Keep every source, torch_source, and inline expression on one physical line. Never emit multiline code.",
        ]
        if self.allow_source:
            schema_lines.extend(
                [
                    "Prefer the paper-style lambda schema below for executable graph edits.",
                    "",
                    "remove:",
                    "  - node.alternative_id",
                    "add:",
                    "  output:",
                    "    - \"lambda ctx, values: <expression>\"",
                    "    - {\"source\": \"lambda ctx, values: <expression>\", \"parents\": [\"parent_node\"], \"node_kind\": \"output\", \"output_contract\": {\"type\": \"feature_block\", \"min_columns\": 1, \"differentiable\": true}, \"torch_source\": \"lambda ctx, values: <torch expression>\"}",
                    "",
                    "You may also use the extended machine schema below when a mutation needs explicit parents, nodes, or globals.",
                ]
            )
        else:
            schema_lines.extend(
                [
                    "Use the extended machine schema below. Source-backed lambda edits are disabled in this run.",
                ]
            )
        schema_lines.extend(
            [
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
        )
        if self.allow_source:
            schema_lines.extend(
                [
                    "",
                    "For sandboxed source-backed alternatives, use primitive \"source\" and include:",
                    "  \"source\": \"lambda ctx, values: <expression>\"",
                    "  \"parents\": [\"explicit_parent_node\"]",
                    "  \"global_refs\": [\"optional_existing_or_new_global\"]",
                    "  \"node_kind\": \"intermediate|callable|output|fitting\"",
                    "  \"output_contract\": {\"type\": \"feature_block\", \"n_columns\": 1, \"differentiable\": true}",
                    "  \"torch_source\": \"lambda ctx, values: <optional differentiable torch expression>\"",
                    "The lambda receives ctx and values, where values maps parent node names to evaluated parent outputs.",
                    "Available lambda symbols: np, math, FeatureBlock, CallableFamily, ResidualWeightRule.",
                    "Use task inputs from ctx.read_input(...) only when they are part of the current task schema.",
                    "Paper-style shorthand lambdas infer parents from values['parent'] and global_refs from ctx.globals.get('name'); use the object form when shape contracts or torch_source matter.",
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
                "Keep the mutation small enough to evaluate quickly: usually one add/remove, or one new node/global plus its required add.",
                f"Step: {step}",
                f"Island: {island if island is not None else 'single'}",
                "",
                "==== CURRENT EVOFOREST (YAML WITH STATS) ====",
                self.annotated_graph(graph, result),
                "",
                "==== EXECUTION ERRORS FROM PREVIOUS ATTEMPTS ====",
                execution_errors.strip() or "(none)",
                "",
                "Do not rely on the raw memorandum here. TOON and memorandum evidence should reach this stage through the scientist hypotheses.",
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
                "Score is the best configuration's cross-validated task score from the configured evaluator scorer.",
                "Evaluation caches deterministic alternatives by selected ancestor subpath, so shared intermediates can be reused across configurations.",
                "The @globals store contains persistent trainable parameters. New globals are append-only at mutation time; unused globals may be pruned by maintenance.",
                "TOON diagnostics include exact additive Ridge contribution fields: shap is the normalized global linear contribution, and cv_shap is the out-of-fold contribution magnitude.",
                "Prefer complementary features, productive bottleneck expansion, useful callable reuse, and removal of redundant or dead structure.",
                "Respect DAG validity and computational cost. Adding intermediate alternatives multiplies configurations; adding output alternatives increases feature count.",
            ]
        )

    def memorandum_prompts(
        self,
        result: EvaluationResult,
        *,
        previous_memorandum: str = "",
        history: Sequence[str] = (),
        error_log: Sequence[str] = (),
        step: int = 0,
        island: int | None = None,
    ) -> tuple[str, str]:
        system = "\n".join(
            [
                "You maintain an experiment log for one island of an evolutionary EvoForest search.",
                "Record observations: what happened, what changed, and what is noteworthy.",
                "Do not add hypotheses or recommendations.",
                "Use only values present in TOON diagnostics or mutation outcomes. Never fabricate numbers.",
                "Keep the memorandum under 500 words and include these sections exactly:",
                "[OUTCOME HISTORY], [STATE], [WHAT WORKS], [WHAT FAILED], [ERROR LOG].",
            ]
        )
        user = "\n".join(
            [
                f"Step: {step}",
                f"Island: {island if island is not None else 'single'}",
                "",
                "==== PREVIOUS MEMORANDUM ====",
                previous_memorandum.strip() or "(none)",
                "",
                "==== RECENT OUTCOMES ====",
                "\n".join(history[-8:]) if history else "- No mutation outcomes recorded yet.",
                "",
                "==== RECENT EXECUTION ERRORS ====",
                "\n".join(error_log[-8:]) if error_log else "- No runtime errors recorded.",
                "",
                "==== FEATURE DIAGNOSTICS (TOON) ====",
                toon_report(result),
            ]
        )
        return system, user

    def annotated_graph(self, graph: Graph, result: EvaluationResult) -> str:
        payload = {
            "graph": graph.to_dict(),
            "configuration_space": graph.configuration_space(),
            "best_config": result.config,
            "feedback": feedback_summary(result),
        }
        text = _to_yaml_like(payload)
        if len(text) <= self.max_graph_chars:
            return text
        return text[: self.max_graph_chars] + "\n... [truncated]"


class LLMScientistAgent(ScientistAgent):
    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_builder: PromptBuilder | None = None,
        temperature: float = 0.35,
        island_temperatures: Sequence[float] | None = DEFAULT_ISLAND_TEMPERATURES,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder()
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
        retry_config = _structured_output_retry_config()
        retry_error = ""
        retry_response = ""
        for attempt_index in range(retry_config.max_retries + 1):
            attempt_user = user
            if attempt_index:
                attempt_user = _structured_output_retry_prompt(
                    user,
                    stage="scientist",
                    previous_response=retry_response,
                    error=retry_error,
                    attempt_index=attempt_index,
                    max_retries=retry_config.max_retries,
                )
            response = ""
            error = ""
            try:
                response = self.client.complete(system, attempt_user, temperature=self._temperature_for(island))
            except Exception as exc:
                error = str(exc)
                self._prompt_records.append(
                    PromptRecord(_prompt_stage("scientist", attempt_index), system, attempt_user, response, error)
                )
                raise
            try:
                hypotheses = parse_hypotheses(response, graph, max_hypotheses)
                if not hypotheses:
                    raise ValueError("LLM scientist returned no parseable hypotheses.")
                self._prompt_records.append(
                    PromptRecord(_prompt_stage("scientist", attempt_index), system, attempt_user, response, "")
                )
                return hypotheses
            except Exception as exc:
                error = str(exc)
                self._prompt_records.append(
                    PromptRecord(_prompt_stage("scientist", attempt_index), system, attempt_user, response, error)
                )
                if attempt_index >= retry_config.max_retries:
                    raise
                retry_error = error
                retry_response = response
                _log_structured_output_retry("scientist", error, attempt_index, retry_config.max_retries)
        raise RuntimeError("LLM scientist structured-output retry loop exhausted.")

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
        registry: PrimitiveRegistry | None = None,
        temperature: float = 0.0,
        allow_source: bool = True,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder(allow_source=allow_source)
        self.registry = registry or PrimitiveRegistry.default()
        self.temperature = float(temperature)
        self.allow_source = bool(allow_source)
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
        retry_config = _structured_output_retry_config()
        retry_error = ""
        retry_response = ""
        for attempt_index in range(retry_config.max_retries + 1):
            attempt_user = user
            if attempt_index:
                attempt_user = _structured_output_retry_prompt(
                    user,
                    stage="engineer",
                    previous_response=retry_response,
                    error=retry_error,
                    attempt_index=attempt_index,
                    max_retries=retry_config.max_retries,
                )
            response = ""
            error = ""
            try:
                response = self.client.complete(system, attempt_user, temperature=self.temperature)
            except Exception as exc:
                error = str(exc)
                self._prompt_records.append(
                    PromptRecord(_prompt_stage("engineer", attempt_index), system, attempt_user, response, error)
                )
                raise
            try:
                document = MutationDocument.from_yaml(response)
                document = self._fill_missing_fields(graph, document, hypotheses)
                self._validate_supported_document(graph, document)
                if not (document.nodes or document.add or document.remove or document.globals):
                    raise ValueError("LLM engineer returned an empty mutation document.")
                self._prompt_records.append(
                    PromptRecord(_prompt_stage("engineer", attempt_index), system, attempt_user, response, "")
                )
                return document
            except Exception as exc:
                error = str(exc)
                self._prompt_records.append(
                    PromptRecord(_prompt_stage("engineer", attempt_index), system, attempt_user, response, error)
                )
                if attempt_index >= retry_config.max_retries:
                    raise
                retry_error = error
                retry_response = response
                _log_structured_output_retry("engineer", error, attempt_index, retry_config.max_retries)
        raise RuntimeError("LLM engineer structured-output retry loop exhausted.")

    def pop_prompt_records(self) -> tuple[PromptRecord, ...]:
        records = tuple(self._prompt_records)
        self._prompt_records.clear()
        return records

    @staticmethod
    def _fill_missing_fields(graph: Graph, document: MutationDocument, hypotheses: tuple[Hypothesis, ...]) -> MutationDocument:
        return MutationDocument(
            hypotheses=document.hypotheses or tuple(item.to_text() for item in hypotheses),
            rationale=document.rationale or "LLM engineer selected a structured mutation from scientist hypotheses.",
            nodes=document.nodes,
            add=document.add,
            remove=tuple(_resolve_remove_spec(graph, spec) for spec in document.remove),
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


class LLMMemorandumAgent:
    def __init__(
        self,
        client: LLMClient,
        *,
        prompt_builder: PromptBuilder | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.temperature = float(temperature)
        self._prompt_records: list[PromptRecord] = []

    def update(
        self,
        result: EvaluationResult,
        *,
        previous_memorandum: str = "",
        history: Sequence[str] = (),
        error_log: Sequence[str] = (),
        step: int = 0,
        island: int | None = None,
    ) -> str:
        system, user = self.prompt_builder.memorandum_prompts(
            result,
            previous_memorandum=previous_memorandum,
            history=history,
            error_log=error_log,
            step=step,
            island=island,
        )
        retry_config = _structured_output_retry_config()
        retry_error = ""
        retry_response = ""
        for attempt_index in range(retry_config.max_retries + 1):
            attempt_user = user
            if attempt_index:
                attempt_user = _structured_output_retry_prompt(
                    user,
                    stage="memorandum",
                    previous_response=retry_response,
                    error=retry_error,
                    attempt_index=attempt_index,
                    max_retries=retry_config.max_retries,
                )
            response = ""
            error = ""
            try:
                response = self.client.complete(system, attempt_user, temperature=self.temperature)
            except Exception as exc:
                error = str(exc)
                self._prompt_records.append(
                    PromptRecord(_prompt_stage("memorandum", attempt_index), system, attempt_user, response, error)
                )
                raise
            try:
                memorandum = response.strip()
                missing = [
                    section
                    for section in ("[OUTCOME HISTORY]", "[STATE]", "[WHAT WORKS]", "[WHAT FAILED]", "[ERROR LOG]")
                    if section not in memorandum
                ]
                if missing:
                    raise ValueError(f"LLM memorandum response is missing required sections: {', '.join(missing)}.")
                self._prompt_records.append(
                    PromptRecord(_prompt_stage("memorandum", attempt_index), system, attempt_user, response, "")
                )
                return memorandum + "\n"
            except Exception as exc:
                error = str(exc)
                self._prompt_records.append(
                    PromptRecord(_prompt_stage("memorandum", attempt_index), system, attempt_user, response, error)
                )
                if attempt_index >= retry_config.max_retries:
                    raise
                retry_error = error
                retry_response = response
                _log_structured_output_retry("memorandum", error, attempt_index, retry_config.max_retries)
        raise RuntimeError("LLM memorandum structured-output retry loop exhausted.")

    def pop_prompt_records(self) -> tuple[PromptRecord, ...]:
        records = tuple(self._prompt_records)
        self._prompt_records.clear()
        return records


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
                improvement_score=_score_field(clean, "Improvement", 5),
                creativity_score=_score_field(clean, "Creativity", 5),
                implementability_score=_score_field(clean, "Implementability", 5),
                risk_score=_score_field(clean, "Risk", 5),
            )
        )
        if len(hypotheses) >= max_hypotheses:
            break
    return tuple(hypotheses)


def _resolve_remove_spec(graph: Graph, spec: RemoveSpec) -> RemoveSpec:
    if spec.target_node:
        return spec
    matches = [
        node_name
        for node_name, node in graph.nodes.items()
        if any(alternative.id == spec.alternative_id for alternative in node.alternatives)
    ]
    if len(matches) != 1:
        return spec
    return RemoveSpec(target_node=matches[0], alternative_id=spec.alternative_id, reason=spec.reason)


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


def _score_field(text: str, name: str, default: int) -> int:
    matches = re.findall(rf"{re.escape(name)}\s*(?:\(|:)?\s*(\d{{1,2}})(?:\s*/\s*10)?", text, flags=re.I)
    if not matches:
        return int(default)
    value = max(1, min(10, int(matches[-1])))
    return value


def _infer_target_node(text: str, graph: Graph) -> str:
    lowered = text.lower()
    for name in sorted(graph.nodes, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            return name
    return "output" if "output" in graph.nodes else next(iter(graph.nodes))


def _to_yaml_like(value: object, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(value, dict):
        rows: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple)):
                rows.append(f"{prefix}{key}:")
                rows.append(_to_yaml_like(item, indent + 1))
            else:
                rows.append(f"{prefix}{key}: {json.dumps(item)}")
        return "\n".join(rows)
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{prefix}[]"
        rows = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                rows.append(f"{prefix}-")
                rows.append(_to_yaml_like(item, indent + 1))
            else:
                rows.append(f"{prefix}- {json.dumps(item)}")
        return "\n".join(rows)
    return f"{prefix}{json.dumps(value)}"


def _structured_output_retry_config() -> LLMStructuredOutputRetryConfig:
    max_retries = _env_int("EVOFOREST_LLM_PARSE_MAX_RETRIES", 3)
    if max_retries < 0:
        raise ValueError("EVOFOREST_LLM_PARSE_MAX_RETRIES must be non-negative.")
    return LLMStructuredOutputRetryConfig(max_retries=max_retries)


def _prompt_stage(stage: str, attempt_index: int) -> str:
    if attempt_index <= 0:
        return stage
    return f"{stage}_retry_{attempt_index:02d}"


def _structured_output_retry_prompt(
    original_user_prompt: str,
    *,
    stage: str,
    previous_response: str,
    error: str,
    attempt_index: int,
    max_retries: int,
) -> str:
    response_excerpt = _truncate_for_retry_prompt(previous_response.strip() or "(empty response)")
    return "\n".join(
        [
            original_user_prompt,
            "",
            "==== RETRY REQUIRED: INVALID STRUCTURED OUTPUT ====",
            f"The previous {stage} response failed parsing or validation.",
            f"Validation error: {error}",
            f"This is repair attempt {attempt_index} of {max_retries}.",
            "Return a complete replacement response that satisfies the original output schema.",
            "Do not explain the error. Do not include prose outside the requested structured output.",
            _structured_output_retry_stage_instruction(stage),
            "",
            "Previous invalid response:",
            "```text",
            response_excerpt,
            "```",
        ]
    )


def _structured_output_retry_stage_instruction(stage: str) -> str:
    if stage == "engineer":
        return (
            "For engineer repairs, emit only the extended mutation YAML beginning with rationale:. "
            "Include at least one add/remove/global/node entry. Prefer a registry-backed primitive and avoid source lambdas in repair output unless strictly necessary."
        )
    if stage == "scientist":
        return "For scientist repairs, emit concise Hypothesis/Rationale/Expected Improvement/Risk Mode blocks only."
    if stage == "memorandum":
        return "For memorandum repairs, emit all required memorandum sections exactly once."
    return "Repair only the structured output requested by the original prompt."


def _truncate_for_retry_prompt(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _log_structured_output_retry(stage: str, error: str, attempt_index: int, max_retries: int) -> None:
    retry_number = attempt_index + 1
    one_line_error = " ".join(error.split())
    print(
        f"LLM {stage} returned invalid structured output ({one_line_error}); retrying {retry_number}/{max_retries}.",
        file=sys.stderr,
        flush=True,
    )


def _post_json(
    url: str,
    payload: dict[str, object],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float,
) -> dict[str, object]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request_body = json.dumps(payload).encode("utf-8")
    retry_config = _llm_retry_config()
    for attempt_index in range(retry_config.max_retries + 1):
        request = urllib.request.Request(
            url,
            data=request_body,
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in RETRYABLE_LLM_HTTP_STATUSES or attempt_index >= retry_config.max_retries:
                raise RuntimeError(f"LLM HTTP request failed with status {exc.code}: {detail[:500]}") from exc
            _sleep_before_llm_retry(f"HTTP {exc.code}", attempt_index, retry_config, exc)
        except urllib.error.URLError as exc:
            if attempt_index >= retry_config.max_retries:
                raise RuntimeError(f"LLM HTTP request failed: {exc}") from exc
            _sleep_before_llm_retry(str(exc), attempt_index, retry_config, None)
    else:  # pragma: no cover - loop always breaks or raises.
        raise RuntimeError("LLM HTTP request failed after retry loop exhausted.")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object.")
    return data


def _llm_retry_config() -> LLMRetryConfig:
    max_retries = _env_int("EVOFOREST_LLM_MAX_RETRIES", 5)
    initial_delay_seconds = _env_float("EVOFOREST_LLM_RETRY_INITIAL_SECONDS", 5.0)
    max_delay_seconds = _env_float("EVOFOREST_LLM_RETRY_MAX_SECONDS", 120.0)
    if max_retries < 0:
        raise ValueError("EVOFOREST_LLM_MAX_RETRIES must be non-negative.")
    if initial_delay_seconds < 0.0:
        raise ValueError("EVOFOREST_LLM_RETRY_INITIAL_SECONDS must be non-negative.")
    if max_delay_seconds < 0.0:
        raise ValueError("EVOFOREST_LLM_RETRY_MAX_SECONDS must be non-negative.")
    return LLMRetryConfig(
        max_retries=max_retries,
        initial_delay_seconds=initial_delay_seconds,
        max_delay_seconds=max_delay_seconds,
    )


def _sleep_before_llm_retry(
    reason: str,
    attempt_index: int,
    retry_config: LLMRetryConfig,
    exc: urllib.error.HTTPError | None,
) -> None:
    retry_after = _retry_after_seconds(exc)
    exponential_delay = retry_config.initial_delay_seconds * (2**attempt_index)
    delay = retry_after if retry_after is not None else exponential_delay
    delay = min(delay, retry_config.max_delay_seconds) if retry_config.max_delay_seconds > 0.0 else 0.0
    retry_number = attempt_index + 1
    print(
        f"LLM request failed transiently ({reason}); retrying {retry_number}/{retry_config.max_retries} in {delay:.1f}s.",
        file=sys.stderr,
        flush=True,
    )
    if delay > 0.0:
        time.sleep(delay)


def _retry_after_seconds(exc: urllib.error.HTTPError | None) -> float | None:
    if exc is None:
        return None
    value = exc.headers.get("Retry-After") if exc.headers is not None else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _extract_openai_content(data: object) -> str:
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


def _extract_claude_content(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("Claude response JSON must be an object.")
    content = data.get("content")
    if not isinstance(content, list):
        raise ValueError("Claude response JSON did not contain a content list.")
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
            parts.append(str(item["text"]))
    if not parts:
        raise ValueError("Claude response JSON did not contain text content.")
    return "\n".join(parts)


def _extract_gemini_content(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("Gemini response JSON must be an object.")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response JSON did not contain candidates.")
    first = candidates[0]
    if not isinstance(first, dict):
        raise ValueError("Gemini candidate must be an object.")
    content = first.get("content")
    if not isinstance(content, dict):
        raise ValueError("Gemini candidate did not contain content.")
    blocks = content.get("parts")
    if not isinstance(blocks, list):
        raise ValueError("Gemini content did not contain parts.")
    parts = []
    for block in blocks:
        if isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    if not parts:
        raise ValueError("Gemini response JSON did not contain text content.")
    return "\n".join(parts)


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


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower().replace("_", "-")


def _required_env(names: tuple[str, ...], provider: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = " or ".join(names)
    raise ValueError(f"Set {joined} in the environment or .env file before using the {provider} LLM provider.")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return float(default)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}.") from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid quoted dotenv value: {value!r}.") from exc
        return str(parsed)
    return _strip_inline_comment(value).strip()


def _strip_inline_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if character == "#" and not in_single_quote and not in_double_quote:
            if index == 0 or value[index - 1].isspace():
                return value[:index]
    return value
