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


@dataclass(frozen=True)
class _WeightedSVD:
    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray
    uy: np.ndarray
    x_mean: np.ndarray
    y_mean: float
    weights: np.ndarray
    sqrt_w: np.ndarray
    n_features: int


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


def _weighted_center(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    weights = normalize_sample_weight(sample_weight, x.shape[0])
    x_mean = np.average(x, axis=0, weights=weights)
    y_mean = float(np.average(y, weights=weights))
    return x - x_mean, y - y_mean, x_mean, y_mean, weights


def _empty_ridge_model(x: np.ndarray, y: np.ndarray, alpha: float, sample_weight: np.ndarray | None = None) -> RidgeModel:
    weights = normalize_sample_weight(sample_weight, y.shape[0])
    y_mean = float(np.average(y, weights=weights)) if y.size else 0.0
    return RidgeModel(coef=np.zeros(x.shape[1], dtype=np.float64), intercept=y_mean, alpha=float(alpha))


def _weighted_svd(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> _WeightedSVD:
    xc, yc, x_mean, y_mean, weights = _weighted_center(x, y, sample_weight)
    sqrt_w = np.sqrt(weights)
    xw = xc * sqrt_w[:, None]
    yw = yc * sqrt_w
    u, s, vt = np.linalg.svd(xw, full_matrices=False)
    return _WeightedSVD(
        u=u,
        s=s,
        vt=vt,
        uy=u.T @ yw,
        x_mean=x_mean,
        y_mean=y_mean,
        weights=weights,
        sqrt_w=sqrt_w,
        n_features=x.shape[1],
    )


def _model_from_svd(decomp: _WeightedSVD, alpha: float) -> RidgeModel:
    coef = decomp.vt.T @ ((decomp.s / (decomp.s * decomp.s + alpha)) * decomp.uy)
    intercept = decomp.y_mean - float(decomp.x_mean @ coef)
    return RidgeModel(coef=coef, intercept=float(intercept), alpha=float(alpha))


def _select_alpha_from_svd(
    decomp: _WeightedSVD,
    y: np.ndarray,
    alphas: np.ndarray,
) -> float:
    best_alpha = float(alphas[0])
    best_mse = float("inf")
    spectral_power = decomp.s * decomp.s
    u_squared = decomp.u * decomp.u
    for alpha in alphas:
        shrinkage = spectral_power / (spectral_power + alpha)
        centered_pred = decomp.u @ (shrinkage * decomp.uy)
        pred = decomp.y_mean + centered_pred / np.maximum(decomp.sqrt_w, EPS)
        leverage = u_squared @ shrinkage
        residual = (y - pred) / np.maximum(1.0 - leverage, EPS)
        mse = float(np.average(residual**2, weights=decomp.weights))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, sample_weight: np.ndarray | None = None) -> RidgeModel:
    if x.size == 0:
        return _empty_ridge_model(x, y, alpha, sample_weight=sample_weight)
    return _model_from_svd(_weighted_svd(x, y, sample_weight), float(alpha))


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray = DEFAULT_ALPHAS,
    sample_weight: np.ndarray | None = None,
) -> float:
    if x.size == 0:
        return float(alphas[0])
    return _select_alpha_from_svd(_weighted_svd(x, y, sample_weight), y, alphas)


def select_alpha_and_fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray = DEFAULT_ALPHAS,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, RidgeModel]:
    if x.size == 0:
        alpha = float(alphas[0])
        return alpha, _empty_ridge_model(x, y, alpha, sample_weight=sample_weight)
    decomp = _weighted_svd(x, y, sample_weight)
    alpha = _select_alpha_from_svd(decomp, y, alphas)
    return alpha, _model_from_svd(decomp, alpha)
