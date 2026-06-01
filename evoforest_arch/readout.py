from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_ALPHAS = np.logspace(-4, 4, 17)
EPS = 1e-8


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale < EPS, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - self.mean) / self.scale, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class RidgeModel:
    coef: np.ndarray
    intercept: float
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coef + self.intercept


def normalize_sample_weight(sample_weight: np.ndarray | None, n_rows: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if weights.shape[0] != n_rows:
        raise ValueError(f"Expected {n_rows} sample weights, got {weights.shape[0]}.")
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, None)
    mean = float(np.mean(weights))
    if mean < EPS:
        return np.ones(n_rows, dtype=np.float64)
    return weights / mean


def combine_sample_weights(*weights: np.ndarray | None) -> np.ndarray | None:
    present = [np.asarray(weight, dtype=np.float64).reshape(-1) for weight in weights if weight is not None]
    if not present:
        return None
    combined = np.ones_like(present[0], dtype=np.float64)
    for weight in present:
        if weight.shape != combined.shape:
            raise ValueError("Sample-weight arrays must have identical shapes.")
        combined *= normalize_sample_weight(weight, weight.shape[0])
    return normalize_sample_weight(combined, combined.shape[0])


def _weighted_center(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    weights = normalize_sample_weight(sample_weight, x.shape[0])
    x_mean = np.average(x, axis=0, weights=weights)
    y_mean = float(np.average(y, weights=weights))
    return x - x_mean, y - y_mean, x_mean, y_mean


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, sample_weight: np.ndarray | None = None) -> RidgeModel:
    xc, yc, x_mean, y_mean = _weighted_center(x, y, sample_weight)
    weights = normalize_sample_weight(sample_weight, x.shape[0])
    sqrt_w = np.sqrt(weights)
    xw = xc * sqrt_w[:, None]
    yw = yc * sqrt_w
    u, s, vt = np.linalg.svd(xw, full_matrices=False)
    coef = vt.T @ ((s / (s * s + alpha)) * (u.T @ yw))
    intercept = y_mean - float(x_mean @ coef)
    return RidgeModel(coef=coef, intercept=float(intercept), alpha=float(alpha))


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray = DEFAULT_ALPHAS,
    sample_weight: np.ndarray | None = None,
) -> float:
    if x.size == 0:
        return float(alphas[0])
    xc, yc, x_mean, y_mean = _weighted_center(x, y, sample_weight)
    weights = normalize_sample_weight(sample_weight, x.shape[0])
    sqrt_w = np.sqrt(weights)
    xw = xc * sqrt_w[:, None]
    yw = yc * sqrt_w
    u, s, vt = np.linalg.svd(xw, full_matrices=False)
    best_alpha = float(alphas[0])
    best_mse = float("inf")
    for alpha in alphas:
        coef = vt.T @ ((s / (s * s + alpha)) * (u.T @ yw))
        intercept = y_mean - float(x_mean @ coef)
        pred = x @ coef + intercept
        leverage = np.sum((u**2) * ((s * s) / (s * s + alpha)), axis=1)
        residual = (y - pred) / np.maximum(1.0 - leverage, EPS)
        mse = float(np.average(residual**2, weights=weights))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha
