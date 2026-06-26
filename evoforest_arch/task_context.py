from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TensorSummary:
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    finite_fraction: float | None = None
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    value: object | None = None

    def to_line(self) -> str:
        if self.kind == "scalar":
            return f"- {self.name}: scalar {self.dtype}, value={self.value!r}."
        stats = ""
        if self.finite_fraction is not None:
            stats = (
                f", finite={self.finite_fraction:.3f}, mean={self.mean:.6f}, std={self.std:.6f}, "
                f"min={self.minimum:.6f}, max={self.maximum:.6f}"
            )
        return f"- {self.name}: {self.kind} {self.dtype}, shape={list(self.shape)}{stats}."


@dataclass(frozen=True)
class TaskContextSummary:
    title: str
    source_brief: tuple[str, ...]
    tensors: tuple[TensorSummary, ...]
    target_rows: int
    positive_count: int
    negative_count: int
    positive_rate: float
    scorer: tuple[str, ...]
    constraints: tuple[str, ...]

    def to_text(self) -> str:
        lines = [
            "# Task Context Summary",
            "",
            self.title,
        ]
        if self.source_brief:
            lines.extend(["", "## Task Source Brief"])
            lines.extend(f"- {item}" for item in self.source_brief)
        lines.extend(["", "## Tensor Inventory"])
        lines.extend(tensor.to_line() for tensor in self.tensors)
        lines.extend(
            [
                "",
                "## Target",
                f"- rows={self.target_rows}, positives={self.positive_count}, negatives={self.negative_count}, positive_rate={self.positive_rate:.6f}.",
                "",
                "## Scorer Mechanics",
            ]
        )
        lines.extend(f"- {item}" for item in self.scorer)
        lines.extend(["", "## Implementation Constraints"])
        lines.extend(f"- {item}" for item in self.constraints)
        return "\n".join(lines) + "\n"


def build_task_context(
    inputs: dict[str, Any],
    y: np.ndarray,
    evaluator: Any,
    *,
    source: str = "runtime inputs",
    task_sources: tuple[tuple[str, str], ...] = (),
) -> TaskContextSummary:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    positives = int(np.sum(y > 0.5))
    negatives = int(y.shape[0] - positives)
    scorer = (
        f"Fitness is best configuration mean ROC-AUC across stratified {int(getattr(evaluator, 'n_splits', 3))}-fold Ridge CV folds.",
        f"Configuration enumeration is capped at {int(getattr(evaluator, 'max_configurations', 64))} candidates per evaluation.",
        "Features are standardized inside each fold; Ridge is solved by closed-form SVD.",
        f"Alpha is selected from {len(getattr(evaluator, 'alphas', []))} log-scale values using leave-one-out leverage MSE.",
        f"ridge_g residual rules run IRLS for up to {int(getattr(evaluator, 'irls_steps', 0))} residual-weighted refits.",
        f"Global refinement enabled={bool(getattr(evaluator, 'refine_globals', False))}, backend={getattr(evaluator, 'refine_backend', 'auto')}, steps={int(getattr(evaluator, 'refine_steps', 0))}.",
    )
    constraints = (
        "Intermediate, callable, and fitting nodes are selected by configuration.",
        "All output alternatives are evaluated and stacked as Ridge features for each configuration.",
        "Graph alternatives should be deterministic over parents, inputs, and fixed globals during one evaluator pass because subpaths are cached.",
        "Globals are persistent trainable parameters; new globals are append-only at mutation time and unused globals may be pruned.",
        "Mutation documents must preserve DAG validity; paper-style source-backed lambda alternatives are first-class when LLM mutation synthesis is enabled.",
    )
    return TaskContextSummary(
        title=f"Clean-room EvoForest task context generated from {source}.",
        source_brief=_source_brief(task_sources),
        tensors=tuple(_summarize_input(name, value) for name, value in sorted(inputs.items())),
        target_rows=int(y.shape[0]),
        positive_count=positives,
        negative_count=negatives,
        positive_rate=float(positives / max(y.shape[0], 1)),
        scorer=scorer,
        constraints=constraints,
    )


def _source_brief(task_sources: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    rows: list[str] = []
    for label, text in task_sources:
        clean = " ".join(text.strip().split())
        if not clean:
            continue
        rows.append(f"{label}: {clean[:900]}{'...' if len(clean) > 900 else ''}")
    return tuple(rows)


def _summarize_input(name: str, value: Any) -> TensorSummary:
    array = np.asarray(value)
    if array.ndim == 0:
        return TensorSummary(
            name=name,
            kind="scalar",
            shape=(),
            dtype=str(array.dtype),
            value=array.item(),
        )
    kind = "numeric_tensor" if np.issubdtype(array.dtype, np.number) else "tensor"
    if not np.issubdtype(array.dtype, np.number):
        return TensorSummary(name=name, kind=kind, shape=tuple(int(dim) for dim in array.shape), dtype=str(array.dtype))
    numeric = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(numeric)
    if not np.any(finite):
        return TensorSummary(
            name=name,
            kind=kind,
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            finite_fraction=0.0,
            mean=0.0,
            std=0.0,
            minimum=0.0,
            maximum=0.0,
        )
    values = numeric[finite]
    return TensorSummary(
        name=name,
        kind=kind,
        shape=tuple(int(dim) for dim in array.shape),
        dtype=str(array.dtype),
        finite_fraction=float(np.mean(finite)),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )
