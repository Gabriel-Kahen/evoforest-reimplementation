"""Evaluation boundaries, accounting, and result schemas for research runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import math
import secrets
from threading import Lock
from typing import Any

import numpy as np

from .metrics import nrmse, r2_score, rmse


class ProtocolViolation(RuntimeError):
    """Raised when sealed-test or finalization rules are violated."""


class BudgetExceeded(RuntimeError):
    """Raised before an operation would exceed a declared budget."""


@dataclass(frozen=True)
class DatasetPartition:
    """A named, immutable matrix/target partition with stable sample IDs."""

    name: str
    X: np.ndarray
    y: np.ndarray
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        X = np.array(self.X, dtype=float, copy=True)
        y = np.array(self.y, dtype=float, copy=True)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be 2-D and y must be 1-D with matching rows.")
        if not X.shape[0] or len(self.sample_ids) != X.shape[0]:
            raise ValueError("sample_ids must provide one ID for every non-empty row.")
        ids = tuple(str(value) for value in self.sample_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("sample_ids must be unique within a partition.")
        if np.any(np.isinf(X)) or not np.all(np.isfinite(y)):
            raise ValueError("Targets must be finite and features may contain NaN but not infinity.")
        X.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "sample_ids", ids)


@dataclass(frozen=True)
class _FinalizedModelToken:
    model_id: str
    nonce: str = field(repr=False)


@dataclass(frozen=True)
class TestEvaluation:
    model_id: str
    n_samples: int
    metrics: Mapping[str, float]


class EvaluationProtocol:
    """Owns search/selection data and permits one terminal test evaluation.

    Test labels are never returned. ``finalize`` permanently closes model
    selection, and ``evaluate_test`` accepts only the token minted at that
    transition. A protocol instance permits exactly one test evaluation.
    """

    def __init__(
        self,
        search_train: DatasetPartition,
        selection_validation: DatasetPartition,
        sealed_test: DatasetPartition,
    ) -> None:
        if len({search_train.name, selection_validation.name, sealed_test.name}) != 3:
            raise ValueError("Partition names must be distinct.")
        id_sets = [set(part.sample_ids) for part in (search_train, selection_validation, sealed_test)]
        if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
            raise ValueError("Sample IDs must be disjoint across all partitions.")
        feature_counts = {part.X.shape[1] for part in (search_train, selection_validation, sealed_test)}
        if len(feature_counts) != 1:
            raise ValueError("All partitions must have the same feature count.")
        self.search_train = search_train
        self.selection_validation = selection_validation
        self._test_X = sealed_test.X
        self._test_y = sealed_test.y
        self._test_name = sealed_test.name
        self._nonce = secrets.token_hex(32)
        self._finalized = False
        self._evaluated = False
        self._model_id: str | None = None
        self._lock = Lock()

    @property
    def test_manifest(self) -> Mapping[str, int | str]:
        return {
            "name": self._test_name,
            "n_samples": int(self._test_X.shape[0]),
            "n_features": int(self._test_X.shape[1]),
        }

    def finalize(self, model_id: str) -> _FinalizedModelToken:
        clean_id = str(model_id).strip()
        if not clean_id:
            raise ValueError("model_id must be non-empty.")
        with self._lock:
            if self._finalized:
                raise ProtocolViolation("Model selection has already been finalized.")
            self._finalized = True
            self._model_id = clean_id
            return _FinalizedModelToken(clean_id, self._nonce)

    def evaluate_test(
        self,
        token: _FinalizedModelToken,
        predict: Callable[[np.ndarray], np.ndarray],
        metric_names: tuple[str, ...] = ("rmse", "nrmse", "r2"),
    ) -> TestEvaluation:
        if not metric_names:
            raise ValueError("At least one test metric is required.")
        metric_registry = {"rmse": rmse, "nrmse": nrmse, "r2": r2_score}
        unknown = sorted(set(metric_names) - set(metric_registry))
        if unknown:
            raise ValueError(f"Unknown sealed-test metrics: {', '.join(unknown)}")
        with self._lock:
            if not self._finalized or token.nonce != self._nonce or token.model_id != self._model_id:
                raise ProtocolViolation("A valid finalized-model token is required.")
            if self._evaluated:
                raise ProtocolViolation("The sealed test set may be evaluated only once.")
            # Consume access before user code runs, including when prediction fails.
            self._evaluated = True

        X = np.array(self._test_X, copy=True)
        X.setflags(write=False)
        predictions = np.asarray(predict(X), dtype=float)
        if predictions.shape != self._test_y.shape or not np.all(np.isfinite(predictions)):
            raise ValueError("predict must return one finite prediction per test row.")
        values: dict[str, float] = {}
        for name in metric_names:
            metric = metric_registry[name]
            value = float(metric(self._test_y, predictions))
            if not math.isfinite(value):
                raise ValueError(f"Metric {name!r} returned a non-finite value.")
            values[str(name)] = value
        return TestEvaluation(token.model_id, int(self._test_y.size), values)


@dataclass(frozen=True)
class BudgetLimits:
    exact_evaluations: int
    screening_evaluations: int = 0
    wall_time_seconds: float = math.inf
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(math.isnan(float(value)) or float(value) < 0 for value in values.values()):
            raise ValueError("Budget limits cannot be negative or NaN.")


@dataclass(frozen=True)
class BudgetUsage:
    exact_evaluations: int = 0
    screening_evaluations: int = 0
    wall_time_seconds: float = 0.0
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(float(value) < 0 or not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Budget usage must be finite and non-negative.")


class BudgetAccountant:
    """Thread-safe, fail-before-consume accounting across all budget axes."""

    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self._usage = BudgetUsage()
        self._lock = Lock()

    @property
    def usage(self) -> BudgetUsage:
        with self._lock:
            return self._usage

    def consume(self, amount: BudgetUsage) -> BudgetUsage:
        with self._lock:
            proposed = BudgetUsage(
                **{
                    key: getattr(self._usage, key) + getattr(amount, key)
                    for key in asdict(self._usage)
                }
            )
            exceeded = [
                key
                for key in asdict(proposed)
                if getattr(proposed, key) > getattr(self.limits, key)
            ]
            if exceeded:
                raise BudgetExceeded(f"Budget exceeded for: {', '.join(exceeded)}")
            self._usage = proposed
            return proposed


@dataclass(frozen=True)
class ExperimentResultRow:
    """One analysis-ready observation from a method/task/seed run."""

    task_id: str
    task_family: str
    method: str
    seed: int
    split_id: str
    status: str
    metrics: Mapping[str, float]
    usage: BudgetUsage
    graph_nodes: int | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "budget_exhausted"}:
            raise ValueError("status must be completed, failed, or budget_exhausted.")
        if self.status == "completed" and not self.metrics:
            raise ValueError("Completed runs require at least one metric.")
        if self.graph_nodes is not None and self.graph_nodes < 0:
            raise ValueError("graph_nodes cannot be negative.")
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("Result metrics must be finite.")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["metrics"] = dict(self.metrics)
        row["metadata"] = dict(self.metadata)
        return row
