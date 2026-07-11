from __future__ import annotations

from evoforest_arch.llm import LLMEngineerAgent, LLMMemorandumAgent, LLMScientistAgent
from evoforest_arch.mutations import MutationDocument, MutationSpec
from evoforest_arch.paper_agents import PaperAgentBundle
from evoforest_arch.primitives import PrimitiveRegistry


def paper_memorandum(label: str = "test") -> str:
    return "\n".join(
        [
            "[OUTCOME HISTORY]",
            f"- {label} outcome.",
            "[STATE]",
            f"- {label} state.",
            "[WHAT WORKS]",
            "- Valid graph candidates are retained.",
            "[WHAT FAILED]",
            "- Invalid candidates are recorded.",
            "[ERROR LOG]",
            "- No runtime errors recorded.",
        ]
    )


class PaperTestClient:
    """Prompt-aware fake client that exercises only the real paper LLM pipeline."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.mutation_index = 0

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.requests.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt, "temperature": float(temperature)}
        )
        if "You maintain an experiment log" in system_prompt:
            memorandum = paper_memorandum(f"step-{len(self.requests)}")
            if "KeyError" in user_prompt:
                memorandum = memorandum.replace(
                    "- No runtime errors recorded.",
                    "- KeyError recorded from the failed candidate.",
                )
            return memorandum
        if "elite computational scientist" in system_prompt:
            return "\n".join(
                [
                    "Hypothesis: Add one complementary representation alternative.",
                    "Rationale: The current diagnostic report shows residual structure.",
                    "Expected Improvement: improve complementary residual coverage.",
                    "Risk Mode: Conservative.",
                ]
            )
        self.mutation_index += 1
        if "shape_stats" in user_prompt:
            spec = MutationSpec(
                kind="add_alternative",
                target_node="shape_stats",
                primitive="spectral_basic",
                alternative_id=f"paper_test_spectral_{self.mutation_index}",
                parents=("series",),
                description="Paper-pipeline test spectral alternative.",
            )
        else:
            spec = MutationSpec(
                kind="add_alternative",
                target_node="base_features",
                primitive="tabular_signed_log",
                alternative_id=f"paper_test_tabular_{self.mutation_index}",
                parents=("x",),
                description="Paper-pipeline test tabular alternative.",
            )
        return MutationDocument(
            hypotheses=("LLM-generated test hypothesis. Expected: complementary coverage.",),
            rationale="Exercise the required paper-style LLM pipeline.",
            add=(spec,),
        ).to_yaml()


def paper_test_agents(registry: PrimitiveRegistry | None = None) -> PaperAgentBundle:
    client = PaperTestClient()
    return PaperAgentBundle(
        scientist=LLMScientistAgent(client),
        engineer=LLMEngineerAgent(client, registry=registry),
        memorandum=LLMMemorandumAgent(client),
    )
