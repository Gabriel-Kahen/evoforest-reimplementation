from __future__ import annotations

import numpy as np
import pytest

from benchmarks.research_suite.metrics import area_under_learning_curve, nrmse, r2_score, rmse
from benchmarks.research_suite.protocol import (
    BudgetAccountant,
    BudgetExceeded,
    BudgetLimits,
    BudgetUsage,
    DatasetPartition,
    EvaluationProtocol,
    ExperimentResultRow,
    ProtocolViolation,
)


def _partition(name: str, offset: int) -> DatasetPartition:
    X = np.arange(12, dtype=float).reshape(4, 3) + offset
    y = np.arange(4, dtype=float) + offset
    return DatasetPartition(name, X, y, tuple(f"{name}-{i}" for i in range(4)))


def test_protocol_seals_test_labels_and_allows_one_terminal_evaluation() -> None:
    protocol = EvaluationProtocol(_partition("train", 0), _partition("validation", 10), _partition("test", 20))
    assert protocol.test_manifest == {"name": "test", "n_samples": 4, "n_features": 3}
    assert not hasattr(protocol, "sealed_test")

    token = protocol.finalize("graph-v7")
    result = protocol.evaluate_test(token, lambda X: np.arange(4) + 20, ("rmse",))
    assert result.model_id == "graph-v7"
    assert result.metrics["rmse"] == pytest.approx(0.0)

    with pytest.raises(ProtocolViolation, match="only once"):
        protocol.evaluate_test(token, lambda X: np.zeros(4), ("rmse",))
    with pytest.raises(ProtocolViolation, match="already"):
        protocol.finalize("different-model")


def test_protocol_rejects_overlapping_partitions() -> None:
    train = _partition("train", 0)
    validation = DatasetPartition("validation", train.X, train.y, train.sample_ids)
    with pytest.raises(ValueError, match="disjoint"):
        EvaluationProtocol(train, validation, _partition("test", 20))


def test_partition_copies_and_freezes_arrays() -> None:
    X = np.ones((3, 2))
    y = np.ones(3)
    partition = DatasetPartition("train", X, y, ("a", "b", "c"))
    X[0, 0] = 99
    assert partition.X[0, 0] == 1
    with pytest.raises(ValueError):
        partition.y[0] = 2


def test_partition_allows_missing_features_for_train_fitted_imputation() -> None:
    partition = _partition("missing", 0)
    values = np.array(partition.X, copy=True)
    values[0, 0] = np.nan

    accepted = DatasetPartition(partition.name, values, partition.y, partition.sample_ids)

    assert np.isnan(accepted.X[0, 0])


def test_budget_accounting_is_atomic_across_axes() -> None:
    accountant = BudgetAccountant(BudgetLimits(10, 5, 60.0, 2, 1000, 1.0))
    used = accountant.consume(BudgetUsage(3, 2, 5.0, 1, 400, 0.2))
    assert used.exact_evaluations == 3

    with pytest.raises(BudgetExceeded, match="llm_calls"):
        accountant.consume(BudgetUsage(exact_evaluations=1, llm_calls=2))
    assert accountant.usage == used


def test_learning_curve_uses_best_incumbent_and_full_budget() -> None:
    # Minimization incumbent is 10 on [0, 2), 6 on [2, 5), and 4 on [5, 10].
    value = area_under_learning_curve([0, 2, 5], [10.0, 6.0, 4.0], budget=10, maximize=False)
    assert value == pytest.approx((20 + 18 + 20) / 10)
    assert area_under_learning_curve([0, 3], [1.0, 0.0], budget=4, maximize=True) == pytest.approx(1.0)


def test_regression_metrics_and_result_schema() -> None:
    truth = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.0, 2.0, 3.0])
    assert rmse(truth, prediction) == 0.0
    assert nrmse(truth, prediction) == 0.0
    assert r2_score(truth, prediction) == 1.0

    row = ExperimentResultRow(
        task_id="dag-001",
        task_family="gated_reuse",
        method="evoforest-full",
        seed=4,
        split_id="outer-0",
        status="completed",
        metrics={"nrmse": 0.2},
        usage=BudgetUsage(exact_evaluations=20),
        graph_nodes=8,
    )
    assert row.to_dict()["usage"]["exact_evaluations"] == 20
