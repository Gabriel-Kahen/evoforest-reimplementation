"""Small, dependency-free metrics used by the research suite."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _paired_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape:
        raise ValueError("y_true and y_pred must be one-dimensional arrays with equal shape.")
    if true.size == 0 or not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("Metric inputs must be non-empty and finite.")
    return true, pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _paired_vectors(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(true - pred))))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalized by target standard deviation."""
    true, pred = _paired_vectors(y_true, y_pred)
    scale = float(np.std(true))
    if scale <= np.finfo(float).eps:
        raise ValueError("NRMSE is undefined for a constant target.")
    return rmse(true, pred) / scale


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _paired_vectors(y_true, y_pred)
    denominator = float(np.sum(np.square(true - np.mean(true))))
    if denominator <= np.finfo(float).eps:
        raise ValueError("R2 is undefined for a constant target.")
    return 1.0 - float(np.sum(np.square(true - pred))) / denominator


def best_so_far(scores: Sequence[float], *, maximize: bool) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty, finite one-dimensional sequence.")
    return np.maximum.accumulate(values) if maximize else np.minimum.accumulate(values)


def area_under_learning_curve(
    evaluations: Sequence[int],
    scores: Sequence[float],
    *,
    budget: int,
    maximize: bool,
) -> float:
    """Return normalized area under a best-so-far, right-continuous step curve.

    A score at evaluation ``e`` becomes the incumbent from ``e`` onward. The
    trace must include the seed model at evaluation zero. Repeated evaluation
    counts are allowed; the last incumbent at that count is used.
    """
    x = np.asarray(evaluations, dtype=int)
    y = np.asarray(scores, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("evaluations and scores must be non-empty vectors of equal length.")
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if x[0] != 0 or np.any(x < 0) or np.any(np.diff(x) < 0):
        raise ValueError("evaluations must be sorted, non-negative, and begin at zero.")
    if x[-1] > budget or not np.all(np.isfinite(y)):
        raise ValueError("The trace must be finite and cannot exceed budget.")

    incumbent = best_so_far(y, maximize=maximize)
    area = 0.0
    for index in range(x.size - 1):
        area += float(x[index + 1] - x[index]) * float(incumbent[index])
    area += float(budget - x[-1]) * float(incumbent[-1])
    return area / float(budget)
