from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ScoreFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class TaskScorer:
    name: str
    description: str
    score_fn: ScoreFunction

    def score(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        return float(self.score_fn(np.asarray(y_true, dtype=np.float64), np.asarray(predictions, dtype=np.float64)))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "higher_is_better": True,
        }


def variance_explained_score(y_true: np.ndarray, predictions: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if y.shape != pred.shape:
        raise ValueError(f"Score arrays must have identical shape, got {y.shape} and {pred.shape}.")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    mse = float(np.mean((y - pred) ** 2))
    variance = float(np.var(y))
    if variance < 1e-12:
        return 1.0 if mse < 1e-12 else -mse
    return 1.0 - mse / variance


DEFAULT_SCORER = TaskScorer(
    name="variance_explained",
    description="Variance explained by predictions against the supplied target; callers may provide any higher-is-better task scorer.",
    score_fn=variance_explained_score,
)


def coerce_scorer(scorer: TaskScorer | ScoreFunction | None) -> TaskScorer:
    if scorer is None:
        return DEFAULT_SCORER
    if isinstance(scorer, TaskScorer):
        return scorer
    return TaskScorer(
        name=getattr(scorer, "__name__", "custom_score"),
        description="User-supplied higher-is-better task score.",
        score_fn=scorer,
    )


def target_alignment(feature: np.ndarray, y: np.ndarray) -> float:
    return abs(safe_corr(feature, y))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Correlation arrays must have identical shape, got {a.shape} and {b.shape}.")
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def random_folds(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D array.")
    effective_splits = max(2, min(int(n_splits), int(y.shape[0])))
    shuffled = np.arange(y.shape[0])
    rng.shuffle(shuffled)
    fold_indices = np.array_split(shuffled, effective_splits)
    all_idx = np.arange(y.shape[0])
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in fold_indices:
        val = np.asarray(sorted(fold.tolist()), dtype=np.int64)
        mask = np.ones(y.shape[0], dtype=bool)
        mask[val] = False
        out.append((all_idx[mask], val))
    return out


def group_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    group_array = np.asarray(groups)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D array.")
    if group_array.ndim != 1 or group_array.shape[0] != y.shape[0]:
        raise ValueError("groups must be a 1-D array with the same length as y.")
    unique_groups = np.unique(group_array)
    effective_splits = max(2, min(int(n_splits), int(unique_groups.shape[0])))
    shuffled = np.arange(unique_groups.shape[0])
    rng.shuffle(shuffled)
    fold_group_indices = np.array_split(shuffled, effective_splits)

    all_idx = np.arange(y.shape[0])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for group_idx in fold_group_indices:
        if group_idx.size == 0:
            continue
        val_groups = unique_groups[np.asarray(group_idx, dtype=np.int64)]
        val_mask = np.isin(group_array, val_groups)
        val = np.flatnonzero(val_mask)
        train = all_idx[~val_mask]
        if train.size and val.size:
            folds.append((train, val))
    if len(folds) < 2:
        return random_folds(y, min(max(2, int(n_splits)), y.shape[0]), seed)
    return folds
