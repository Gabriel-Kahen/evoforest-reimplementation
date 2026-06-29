from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evoforest_arch.metrics import ScoreFunction, TaskScorer, coerce_scorer, group_folds, safe_corr, target_alignment
from evoforest_arch.readout import Standardizer


@dataclass(frozen=True)
class RankFeatureExpansion:
    standardizer: Standardizer
    selected_for_interactions: tuple[int, ...]
    names: tuple[str, ...]

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = np.clip(self.standardizer.transform(np.asarray(x, dtype=np.float64)), -8.0, 8.0)
        columns: list[np.ndarray] = [z]
        if self.selected_for_interactions:
            selected = z[:, np.asarray(self.selected_for_interactions, dtype=np.int64)]
            columns.append(np.abs(selected))
            pairs = [
                selected[:, left] * selected[:, right]
                for left in range(selected.shape[1])
                for right in range(left + 1, selected.shape[1])
            ]
            if pairs:
                columns.append(np.column_stack(pairs))
        return np.nan_to_num(np.column_stack(columns), nan=0.0, posinf=0.0, neginf=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_for_interactions": list(self.selected_for_interactions),
            "n_features": len(self.names),
            "names": list(self.names),
        }


@dataclass(frozen=True)
class RankEnsembleModel:
    feature_indices: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    max_features: int
    power: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.feature_indices.size == 0:
            return np.zeros(x.shape[0], dtype=np.float64)
        selected = x[:, self.feature_indices]
        ranks = rank_columns(selected)
        oriented = np.where(self.directions.reshape(1, -1) > 0, ranks, 1.0 - ranks)
        return oriented @ self.weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_indices": [int(index) for index in self.feature_indices.tolist()],
            "directions": [float(value) for value in self.directions.tolist()],
            "weights": [float(value) for value in self.weights.tolist()],
            "max_features": int(self.max_features),
            "power": float(self.power),
        }


@dataclass(frozen=True)
class RankSelectionResult:
    model: RankEnsembleModel
    oof_predictions: np.ndarray
    oof_score: float
    config: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "oof_score": float(self.oof_score),
            "config": dict(self.config),
            "model": self.model.to_dict(),
            "candidates": [dict(row) for row in self.candidates],
        }


def fit_rank_feature_expansion(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    max_interaction_base: int = 12,
) -> RankFeatureExpansion:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    standardizer = Standardizer.fit(x_train)
    z = np.clip(standardizer.transform(x_train), -8.0, 8.0)
    edges = np.asarray([target_alignment(z[:, idx], y_train) for idx in range(z.shape[1])], dtype=np.float64)
    selected = tuple(int(index) for index in np.argsort(edges)[::-1][: max(0, min(int(max_interaction_base), z.shape[1]))])
    names = list(feature_names)
    names.extend(f"abs::{feature_names[idx]}" for idx in selected)
    names.extend(f"interaction::{feature_names[selected[left]]}*{feature_names[selected[right]]}" for left in range(len(selected)) for right in range(left + 1, len(selected)))
    return RankFeatureExpansion(standardizer=standardizer, selected_for_interactions=selected, names=tuple(names))


def select_rank_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 3,
    seed: int = 0,
    max_features_options: tuple[int, ...] = (8, 16, 32, 64, 128),
    power_options: tuple[float, ...] = (1.0, 1.5, 2.0),
    scorer: TaskScorer | ScoreFunction | None = None,
) -> RankSelectionResult:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    groups = np.asarray(groups)
    task_scorer = coerce_scorer(scorer)
    folds = group_folds(y_train, groups, n_splits, seed)
    best: dict[str, Any] | None = None
    candidate_rows: list[dict[str, Any]] = []
    for max_features in max_features_options:
        for power in power_options:
            oof = np.zeros(y_train.shape[0], dtype=np.float64)
            for train_idx, validation_idx in folds:
                model = fit_rank_ensemble(x_train[train_idx], y_train[train_idx], max_features=max_features, power=power)
                oof[validation_idx] = model.predict(x_train[validation_idx])
            score = task_scorer.score(y_train, oof)
            row = {"max_features": int(max_features), "power": float(power), "oof_score": float(score)}
            candidate_rows.append(row)
            if best is None or score > float(best["oof_score"]):
                best = {**row, "oof_predictions": oof}
    if best is None:
        model = fit_rank_ensemble(x_train, y_train, max_features=1, power=1.0)
        oof = model.predict(x_train)
        best = {"max_features": 1, "power": 1.0, "oof_score": task_scorer.score(y_train, oof), "oof_predictions": oof}
    model = fit_rank_ensemble(x_train, y_train, max_features=int(best["max_features"]), power=float(best["power"]))
    return RankSelectionResult(
        model=model,
        oof_predictions=np.asarray(best["oof_predictions"], dtype=np.float64),
        oof_score=float(best["oof_score"]),
        config={"max_features": int(best["max_features"]), "power": float(best["power"])},
        candidates=tuple(candidate_rows),
    )


def fit_rank_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_features: int,
    power: float,
) -> RankEnsembleModel:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if x_train.ndim != 2 or x_train.shape[1] == 0:
        return RankEnsembleModel(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), int(max_features), float(power))
    correlations = np.asarray([safe_corr(x_train[:, idx], y_train) for idx in range(x_train.shape[1])], dtype=np.float64)
    directions = np.where(correlations >= 0.0, 1.0, -1.0)
    edges = np.abs(correlations)
    order = np.argsort(edges)[::-1]
    selected = order[: max(1, min(int(max_features), x_train.shape[1]))]
    raw_weights = np.maximum(edges[selected], 1e-8) ** float(power)
    weights = raw_weights / max(float(np.sum(raw_weights)), 1e-8)
    return RankEnsembleModel(
        feature_indices=np.asarray(selected, dtype=np.int64),
        directions=directions[selected].astype(np.float64),
        weights=weights.astype(np.float64),
        max_features=int(max_features),
        power=float(power),
    )


def rank_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    out = np.zeros_like(x, dtype=np.float64)
    if x.shape[0] <= 1:
        out.fill(0.5)
        return out
    for col in range(x.shape[1]):
        out[:, col] = normalized_ranks(x[:, col])
    return out


def normalized_ranks(x: np.ndarray) -> np.ndarray:
    score = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks / max(float(score.shape[0] - 1), 1.0)
