from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hypothesis:
    """Structured hypothesis emitted by the paper-style LLM scientist."""

    kind: str
    target_node: str
    rationale: str
    expected_improvement: str
    risk: str = "Balanced"
    improvement_score: int = 5
    creativity_score: int = 5
    implementability_score: int = 5
    risk_score: int = 5

    def to_text(self) -> str:
        return (
            f"{self.kind} on {self.target_node}: {self.rationale} "
            f"Expected: {self.expected_improvement}. Risk: {self.risk}. "
            "Self-Evaluation: "
            f"Improvement {self.improvement_score}/10, "
            f"Creativity {self.creativity_score}/10, "
            f"Implementability {self.implementability_score}/10, "
            f"Risk {self.risk_score}/10."
        )
