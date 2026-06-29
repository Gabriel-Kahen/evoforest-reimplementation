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
    higher_is_better: bool = True
    raw_name: str = ""
    raw_description: str = ""

    def score(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        raw = self.raw_score(y_true, predictions)
        return float(raw if self.higher_is_better else -raw)

    def raw_score(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        return float(self.score_fn(np.asarray(y_true, dtype=np.float64), np.asarray(predictions, dtype=np.float64)))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "raw_metric": self.raw_name or self.name,
            "raw_description": self.raw_description or self.description,
            "raw_higher_is_better": bool(self.higher_is_better),
            "optimization_score": "raw" if self.higher_is_better else "negative_raw",
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
    higher_is_better=True,
)


def rmse_score(y_true: np.ndarray, predictions: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if y.shape != pred.shape:
        raise ValueError(f"Score arrays must have identical shape, got {y.shape} and {pred.shape}.")
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae_score(y_true: np.ndarray, predictions: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if y.shape != pred.shape:
        raise ValueError(f"Score arrays must have identical shape, got {y.shape} and {pred.shape}.")
    return float(np.mean(np.abs(y - pred)))


RMSE_SCORER = TaskScorer(
    name="negative_rmse",
    description="Negative RMSE optimization score; raw RMSE is preserved as a lower-is-better metric.",
    score_fn=rmse_score,
    higher_is_better=False,
    raw_name="rmse",
    raw_description="Root mean squared error.",
)


MAE_SCORER = TaskScorer(
    name="negative_mae",
    description="Negative MAE optimization score; raw MAE is preserved as a lower-is-better metric.",
    score_fn=mae_score,
    higher_is_better=False,
    raw_name="mae",
    raw_description="Mean absolute error.",
)


SCORER_REGISTRY = {
    "variance_explained": DEFAULT_SCORER,
    "default": DEFAULT_SCORER,
    "rmse": RMSE_SCORER,
    "negative_rmse": RMSE_SCORER,
    "mae": MAE_SCORER,
    "negative_mae": MAE_SCORER,
}


def scorer_from_name(name: str) -> TaskScorer:
    key = str(name).strip().lower()
    if key not in SCORER_REGISTRY:
        raise ValueError(f"Unsupported scorer {name!r}; available scorers: {sorted(SCORER_REGISTRY)}.")
    return SCORER_REGISTRY[key]


def coerce_scorer(scorer: TaskScorer | ScoreFunction | str | None) -> TaskScorer:
    if scorer is None:
        return DEFAULT_SCORER
    if isinstance(scorer, TaskScorer):
        return scorer
    if isinstance(scorer, str):
        return scorer_from_name(scorer)
    return TaskScorer(
        name=getattr(scorer, "__name__", "custom_score"),
        description="User-supplied higher-is-better task score.",
        score_fn=scorer,
        higher_is_better=True,
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


def leave_group_out_folds(y: np.ndarray, groups: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    group_array = np.asarray(groups)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D array.")
    if group_array.ndim != 1 or group_array.shape[0] != y.shape[0]:
        raise ValueError("groups must be a 1-D array with the same length as y.")
    unique_groups = np.unique(group_array)
    if unique_groups.shape[0] < 2:
        return random_folds(y, min(max(2, int(y.shape[0])), y.shape[0]), seed)
    order = np.arange(unique_groups.shape[0])
    rng.shuffle(order)
    all_idx = np.arange(y.shape[0])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for group_idx in order:
        val_mask = group_array == unique_groups[int(group_idx)]
        val = np.flatnonzero(val_mask)
        train = all_idx[~val_mask]
        if train.size and val.size:
            folds.append((train, val))
    return folds


@dataclass(frozen=True)
class FoldStrategy:
    name: str = "random"
    group_key: str | None = None
    time_key: str | None = None
    stratify_bins: int = 5

    def split(self, inputs: dict[str, object], y: np.ndarray, n_splits: int, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
        y_array = np.asarray(y)
        if self.name == "random":
            return random_folds(y_array, n_splits, seed), self._diagnostics("random")
        if self.name in {"group_random", "leave_group_out"}:
            return self._group_split(inputs, y_array, n_splits, seed)
        if self.name == "stratified":
            folds = stratified_folds(y_array, n_splits, seed, bins=self.stratify_bins)
            return folds, {**self._diagnostics("stratified"), "stratify_bins": int(self.stratify_bins)}
        if self.name == "time_blocked":
            return self._time_blocked_split(inputs, y_array, n_splits)
        raise ValueError(f"Unsupported fold strategy {self.name!r}.")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "group_key": self.group_key,
            "time_key": self.time_key,
            "stratify_bins": int(self.stratify_bins),
        }

    def _diagnostics(self, method: str, **extra: object) -> dict[str, object]:
        return {"method": method, "group_key": self.group_key, "time_key": self.time_key, **extra}

    def _group_split(
        self,
        inputs: dict[str, object],
        y: np.ndarray,
        n_splits: int,
        seed: int,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
        if self.group_key is None or self.group_key not in inputs:
            return random_folds(y, n_splits, seed), self._diagnostics("random", grouped=False, reason="group key was not supplied")
        groups = np.asarray(inputs[self.group_key])
        if groups.ndim != 1 or groups.shape[0] != y.shape[0]:
            return random_folds(y, n_splits, seed), self._diagnostics("random", grouped=False, reason="group array shape did not match y")
        folds = leave_group_out_folds(y, groups, seed) if self.name == "leave_group_out" else group_folds(y, groups, n_splits, seed)
        overlap_count = 0
        validation_group_counts: list[int] = []
        for train_idx, val_idx in folds:
            train_groups = set(groups[train_idx].tolist())
            val_groups = set(groups[val_idx].tolist())
            overlap_count += len(train_groups & val_groups)
            validation_group_counts.append(len(val_groups))
        return folds, self._diagnostics(
            self.name,
            grouped=True,
            fold_group_overlap_count=int(overlap_count),
            validation_group_counts=validation_group_counts,
            n_groups=int(np.unique(groups).shape[0]),
            actual_folds=int(len(folds)),
            requested_folds=int(n_splits),
        )

    def _time_blocked_split(self, inputs: dict[str, object], y: np.ndarray, n_splits: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
        if self.time_key is not None and self.time_key in inputs:
            time_values = np.asarray(inputs[self.time_key])
            if time_values.ndim != 1 or time_values.shape[0] != y.shape[0]:
                order = np.arange(y.shape[0])
                reason = "time array shape did not match y; used sample order"
            else:
                order = np.argsort(time_values, kind="mergesort")
                reason = ""
        else:
            order = np.arange(y.shape[0])
            reason = "time key was not supplied; used sample order"
        effective_splits = max(2, min(int(n_splits), int(y.shape[0])))
        blocks = np.array_split(order, effective_splits)
        all_idx = np.arange(y.shape[0])
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for block in blocks:
            val = np.asarray(sorted(block.tolist()), dtype=np.int64)
            mask = np.ones(y.shape[0], dtype=bool)
            mask[val] = False
            folds.append((all_idx[mask], val))
        diagnostics = self._diagnostics("time_blocked", blocked=True, validation_block_sizes=[int(block.size) for block in blocks])
        if reason:
            diagnostics["reason"] = reason
        return folds, diagnostics


def coerce_fold_strategy(
    strategy: FoldStrategy | str | None = None,
    *,
    group_key: str | None = None,
    time_key: str | None = None,
    stratify_bins: int = 5,
) -> FoldStrategy:
    if isinstance(strategy, FoldStrategy):
        return strategy
    name = str(strategy or ("group_random" if group_key else "random"))
    return FoldStrategy(name=name, group_key=group_key, time_key=time_key, stratify_bins=int(stratify_bins))


def stratified_folds(y: np.ndarray, n_splits: int, seed: int, *, bins: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError("y must be a 1-D array.")
    effective_splits = max(2, min(int(n_splits), int(y_array.shape[0])))
    labels = _stratification_labels(y_array, bins=max(2, int(bins)))
    validation_parts: list[list[int]] = [[] for _ in range(effective_splits)]
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for fold_idx, part in enumerate(np.array_split(indices, effective_splits)):
            validation_parts[fold_idx].extend(int(index) for index in part.tolist())
    all_idx = np.arange(y_array.shape[0])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for part in validation_parts:
        val = np.asarray(sorted(part), dtype=np.int64)
        if val.size == 0:
            continue
        mask = np.ones(y_array.shape[0], dtype=bool)
        mask[val] = False
        folds.append((all_idx[mask], val))
    if len(folds) < 2:
        return random_folds(y_array, n_splits, seed)
    return folds


def _stratification_labels(y: np.ndarray, *, bins: int) -> np.ndarray:
    unique = np.unique(y)
    if unique.shape[0] <= bins:
        return y
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    edges = np.unique(np.quantile(y, quantiles))
    if edges.size == 0:
        return np.zeros(y.shape[0], dtype=np.int64)
    return np.searchsorted(edges, y, side="right")
