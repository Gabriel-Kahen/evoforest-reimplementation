from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evoforest_arch.llm import (
    DEFAULT_ISLAND_TEMPERATURES,
    GeminiLLMClient,
    LLMClient,
    LLMEngineerAgent,
    LLMMemorandumAgent,
    LLMScientistAgent,
    PromptBuilder,
    llm_client_from_env,
)
from evoforest_arch.primitives import PrimitiveRegistry


@dataclass(frozen=True)
class PaperAgentBundle:
    scientist: LLMScientistAgent
    engineer: LLMEngineerAgent
    memorandum: LLMMemorandumAgent


def build_paper_agents(
    client: LLMClient,
    *,
    task_context: str = "",
    registry: PrimitiveRegistry | None = None,
    allow_source: bool = True,
) -> PaperAgentBundle:
    prompt_builder = PromptBuilder(
        task_context=task_context or PromptBuilder().task_context,
        registry=registry,
        allow_source=allow_source,
    )
    return PaperAgentBundle(
        scientist=LLMScientistAgent(
            client,
            prompt_builder=prompt_builder,
            temperature=0.35,
            island_temperatures=DEFAULT_ISLAND_TEMPERATURES,
        ),
        engineer=LLMEngineerAgent(
            client,
            prompt_builder=prompt_builder,
            registry=registry,
            temperature=0.0,
            allow_source=allow_source,
        ),
        memorandum=LLMMemorandumAgent(client, prompt_builder=prompt_builder, temperature=0.0),
    )


def build_gemini_paper_agents(
    env_file: str | Path = ".env",
    *,
    task_context: str = "",
    registry: PrimitiveRegistry | None = None,
    allow_source: bool = True,
) -> PaperAgentBundle:
    client = llm_client_from_env(env_file)
    if not isinstance(client, GeminiLLMClient):
        raise ValueError("Paper-style Gemini setup requires EVOFOREST_LLM_PROVIDER=gemini.")
    return build_paper_agents(
        client,
        task_context=task_context,
        registry=registry,
        allow_source=allow_source,
    )
