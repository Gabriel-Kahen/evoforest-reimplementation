from __future__ import annotations

import math
import os
import pickle
from typing import Iterable, Optional, Sequence

import numpy as np


MODEL_FILE = "evoforest_realtime_model.pkl"
DEFAULT_MAX_ROWS_PER_SERIES = 32
DEFAULT_SERIES_LENGTH = 480
DEFAULT_MAX_ONLINE_LENGTH = 1000
ALPHAS = np.logspace(-4, 4, 17)
TAIL_WINDOWS = (4, 8, 16, 32, 64, 128, 240)
EPS = 1e-8


def train(
    datasets: list[tuple[int, Sequence[float], Sequence[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    """Fit the streaming EvoForest-style ridge readout for CrunchDAO.

    The training rows mirror the full-data row benchmark: each sampled online
    time step becomes one row, labelled 1 iff a break has already occurred.
    """

    os.makedirs(model_directory_path, exist_ok=True)
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for dataset_id, x_historical, x_online, tau_index in datasets:
        del dataset_id
        historical = _as_1d(x_historical)
        online = _as_1d(x_online)
        if online.size == 0:
            continue
        for idx in _training_indices(online.size, tau_index, DEFAULT_MAX_ROWS_PER_SERIES):
            features, names = extract_streaming_features(
                historical,
                online[: idx + 1],
                online_index=idx,
                online_length=online.size,
            )
            x_rows.append(features)
            y_rows.append(float(tau_index is not None and idx >= int(tau_index)))

    if not x_rows:
        model = _fallback_model()
    else:
        x = np.vstack(x_rows)
        y = np.asarray(y_rows, dtype=np.float64)
        model = fit_streaming_ridge(x, y, names)

    model["metadata"] = {
        "model": "evoforest_realtime_streaming_ridge",
        "series_length": DEFAULT_SERIES_LENGTH,
        "max_rows_per_series": DEFAULT_MAX_ROWS_PER_SERIES,
        "tail_windows": list(TAIL_WINDOWS),
        "score_transform": "sigmoid(raw_ridge_prediction)",
    }
    with open(os.path.join(model_directory_path, MODEL_FILE), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def infer(
    datasets: Iterable[tuple[Sequence[float], Iterable[float]]],
    model_directory_path: str,
):
    """Yield one break score for each online observation, in order."""

    model = load_model(model_directory_path)
    yield

    for item in datasets:
        historical, x_online = _unpack_infer_item(item)
        historical = _as_1d(historical)
        online_length = _safe_len(x_online)
        capacity = max(int(online_length) if online_length else DEFAULT_MAX_ONLINE_LENGTH, 1)
        prefix = np.empty(capacity, dtype=np.float64)
        for online_index, point in enumerate(x_online):
            if online_index >= prefix.shape[0]:
                grown = np.empty(prefix.shape[0] * 2, dtype=np.float64)
                grown[: prefix.shape[0]] = prefix
                prefix = grown
            prefix[online_index] = float(point)
            features, _names = extract_streaming_features(
                historical,
                prefix[: online_index + 1],
                online_index=online_index,
                online_length=online_length,
            )
            raw = predict_raw(model, features)
            yield float(_sigmoid(raw))


def load_model(model_directory_path: str) -> dict[str, object]:
    path = os.path.join(model_directory_path, MODEL_FILE)
    if not os.path.exists(path):
        return _fallback_model()
    with open(path, "rb") as handle:
        return pickle.load(handle)


def fit_streaming_ridge(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, object]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < EPS, 1.0, scale)
    z = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = select_alpha(z, y, ALPHAS)
    coef, intercept = fit_ridge(z, y, alpha)
    return {
        "feature_names": list(names),
        "mean": mean,
        "scale": scale,
        "coef": coef,
        "intercept": float(intercept),
        "alpha": float(alpha),
    }


def predict_raw(model: dict[str, object], features: np.ndarray) -> float:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    z = np.nan_to_num((np.asarray(features, dtype=np.float64) - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    return float(z @ coef + float(model["intercept"]))


def extract_streaming_features(
    historical: np.ndarray,
    online_prefix: np.ndarray,
    *,
    online_index: int,
    online_length: int | None,
) -> tuple[np.ndarray, list[str]]:
    historical = _as_1d(historical)
    online = _as_1d(online_prefix)
    if online.size == 0:
        online = np.zeros(1, dtype=np.float64)
    hist_mean = float(np.mean(historical)) if historical.size else 0.0
    hist_std = _safe_std_1d(historical)
    hist_diff = np.diff(historical) if historical.size > 1 else np.zeros(1, dtype=np.float64)
    hist_absdiff = float(np.mean(np.abs(hist_diff))) if hist_diff.size else 0.0
    hist_diff_std = _safe_std_1d(hist_diff)
    hist_slope = _slope_1d(historical)
    hist_autocorr = _autocorr1_1d(historical)
    hist_drawdown, hist_drawup = _drawdown_drawup_1d(historical)
    hist_q = _quantiles(historical)

    online_mean = float(np.mean(online))
    online_std = _safe_std_1d(online)
    online_diff = np.diff(online) if online.size > 1 else np.zeros(1, dtype=np.float64)
    online_absdiff = float(np.mean(np.abs(online_diff))) if online_diff.size else 0.0
    online_diff_std = _safe_std_1d(online_diff)
    online_slope = _slope_1d(online)
    online_autocorr = _autocorr1_1d(online)
    online_drawdown, online_drawup = _drawdown_drawup_1d(online)
    online_q = _quantiles(online)
    centered = online - hist_mean
    cumulative = np.cumsum(centered)
    peak_abs = float(np.max(np.abs(cumulative))) if cumulative.size else 0.0
    last = float(online[-1])
    n_online = int(online.size)
    denom = max(int(online_length) - 1, 1) if online_length else DEFAULT_MAX_ONLINE_LENGTH
    time_norm = min(max(float(online_index) / float(denom), 0.0), 1.0)
    observed_fraction = min(float(n_online) / float(DEFAULT_MAX_ONLINE_LENGTH), 1.0)

    values = [
        last,
        (last - hist_mean) / hist_std,
        (last - online_mean) / online_std,
        (online_mean - hist_mean) / hist_std,
        math.log((online_std + EPS) / (hist_std + EPS)),
        online_absdiff - hist_absdiff,
        math.log((online_absdiff + EPS) / (hist_absdiff + EPS)),
        math.log((online_diff_std + EPS) / (hist_diff_std + EPS)),
        online_slope,
        online_slope - hist_slope,
        online_autocorr - hist_autocorr,
        online_drawdown,
        online_drawup,
        online_drawdown - hist_drawdown,
        online_drawup - hist_drawup,
        peak_abs / hist_std,
        float(np.argmax(np.abs(cumulative)) / max(cumulative.size - 1, 1)) if cumulative.size else 0.0,
        float(cumulative[-1] / max(peak_abs, EPS)) if cumulative.size else 0.0,
        float(np.mean(np.abs(online_q - hist_q))),
        float(np.max(np.abs(online_q - hist_q))),
        time_norm,
        math.log1p(max(float(online_index), 0.0)) / math.log1p(DEFAULT_MAX_ONLINE_LENGTH),
        time_norm * time_norm,
        time_norm * time_norm * time_norm,
        math.sqrt(max(time_norm, 0.0)),
        math.sin(math.pi * time_norm),
        math.cos(math.pi * time_norm),
        math.sin(2.0 * math.pi * time_norm),
        math.cos(2.0 * math.pi * time_norm),
        observed_fraction,
        observed_fraction * observed_fraction,
        time_norm * observed_fraction,
    ]
    names = [
        "last_value",
        "last_vs_hist_mean_z",
        "last_vs_online_mean_z",
        "online_vs_hist_mean_z",
        "online_hist_std_log_ratio",
        "absdiff_delta",
        "absdiff_log_ratio",
        "diff_std_log_ratio",
        "online_slope",
        "slope_delta",
        "autocorr_delta",
        "online_drawdown",
        "online_drawup",
        "drawdown_delta",
        "drawup_delta",
        "hist_centered_cusum_peak",
        "hist_centered_cusum_peak_location",
        "terminal_cusum_balance",
        "quantile_l1_distance",
        "quantile_max_distance",
        "time_norm",
        "time_log_norm",
        "time_sq",
        "time_cube",
        "time_sqrt",
        "time_sin1",
        "time_cos1",
        "time_sin2",
        "time_cos2",
        "observed_fraction",
        "observed_sq",
        "time_observed_interaction",
    ]

    for width in TAIL_WINDOWS:
        tail_values, tail_names = _tail_feature_values(historical, online, width, hist_mean, hist_std)
        values.extend(tail_values)
        names.extend(tail_names)
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), names


def _tail_feature_values(
    historical: np.ndarray,
    online: np.ndarray,
    width: int,
    hist_mean: float,
    hist_std: float,
) -> tuple[list[float], list[str]]:
    if online.size < 2:
        values = [0.0] * 13
    else:
        usable = min(int(width), int(online.size))
        tail = online[-usable:]
        previous = online[max(0, online.size - 2 * usable) : online.size - usable]
        if previous.size < 2:
            previous = historical[-min(usable, historical.size) :] if historical.size else np.asarray([hist_mean])
        old = historical[-min(usable, historical.size) :] if historical.size else np.asarray([hist_mean])
        tail_mean = float(np.mean(tail))
        previous_mean = float(np.mean(previous))
        old_mean = float(np.mean(old))
        tail_std = _safe_std_1d(tail)
        previous_std = _safe_std_1d(previous)
        old_std = _safe_std_1d(old)
        tail_diff = np.diff(tail) if tail.size > 1 else np.zeros(1, dtype=np.float64)
        previous_diff = np.diff(previous) if previous.size > 1 else np.zeros(1, dtype=np.float64)
        old_diff = np.diff(old) if old.size > 1 else np.zeros(1, dtype=np.float64)
        tail_absdiff = float(np.mean(np.abs(tail_diff))) if tail_diff.size else 0.0
        previous_absdiff = float(np.mean(np.abs(previous_diff))) if previous_diff.size else 0.0
        old_absdiff = float(np.mean(np.abs(old_diff))) if old_diff.size else 0.0
        drawdown, drawup = _drawdown_drawup_1d(tail)
        tail_slope = _slope_1d(tail)
        previous_slope = _slope_1d(previous)
        values = [
            (float(tail[-1]) - tail_mean) / max(tail_std, EPS),
            (tail_mean - previous_mean) / max(hist_std, EPS),
            (tail_mean - old_mean) / max(hist_std, EPS),
            math.log((tail_std + EPS) / (previous_std + EPS)),
            math.log((tail_std + EPS) / (old_std + EPS)),
            tail_absdiff - previous_absdiff,
            math.log((tail_absdiff + EPS) / (previous_absdiff + EPS)),
            math.log((tail_absdiff + EPS) / (old_absdiff + EPS)),
            tail_slope,
            tail_slope - previous_slope,
            drawdown,
            drawup,
            float(np.max(tail) - np.min(tail)),
        ]
    names = [
        f"tail_last_vs_mean_{width}",
        f"tail_prev_mean_delta_{width}",
        f"tail_old_mean_delta_{width}",
        f"tail_prev_std_log_ratio_{width}",
        f"tail_old_std_log_ratio_{width}",
        f"tail_prev_absdiff_delta_{width}",
        f"tail_prev_absdiff_log_ratio_{width}",
        f"tail_old_absdiff_log_ratio_{width}",
        f"tail_slope_{width}",
        f"tail_prev_slope_delta_{width}",
        f"tail_drawdown_{width}",
        f"tail_drawup_{width}",
        f"tail_range_{width}",
    ]
    return values, names


def select_alpha(x: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> float:
    if x.size == 0:
        return float(alphas[0])
    y_mean = float(np.mean(y))
    yc = y - y_mean
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    best_alpha = float(alphas[0])
    best_mse = float("inf")
    for alpha in alphas:
        coef = vt.T @ ((s / (s * s + alpha)) * (u.T @ yc))
        pred = x @ coef + y_mean
        leverage = np.sum((u**2) * ((s * s) / (s * s + alpha)), axis=1)
        residual = (y - pred) / np.maximum(1.0 - leverage, EPS)
        mse = float(np.mean(residual**2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x_mean = np.mean(x, axis=0)
    y_mean = float(np.mean(y))
    xc = x - x_mean
    yc = y - y_mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    coef = vt.T @ ((s / (s * s + float(alpha))) * (u.T @ yc))
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept


def _training_indices(n_online: int, tau_index: int | None, max_rows: int) -> np.ndarray:
    base_count = min(max(int(max_rows), 1), int(n_online))
    indices = set(np.linspace(0, n_online - 1, base_count, dtype=np.int64).tolist())
    if tau_index is not None:
        tau = int(tau_index)
        for offset in range(-4, 9):
            idx = tau + offset
            if 0 <= idx < n_online:
                indices.add(idx)
    return np.asarray(sorted(indices), dtype=np.int64)


def _as_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_std_1d(values: np.ndarray) -> float:
    if values.size < 2:
        return 1.0
    return max(float(np.std(values)), EPS)


def _slope_1d(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    t = np.linspace(-1.0, 1.0, values.size)
    centered_t = t - float(np.mean(t))
    centered_x = values - float(np.mean(values))
    denom = max(float(np.sum(centered_t**2)), EPS)
    return float(centered_x @ centered_t / denom)


def _drawdown_drawup_1d(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    cumulative_max = np.maximum.accumulate(values)
    cumulative_min = np.minimum.accumulate(values)
    scale = _safe_std_1d(values)
    return float(np.max(cumulative_max - values) / scale), float(np.max(values - cumulative_min) / scale)


def _autocorr1_1d(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    centered = values - float(np.mean(values))
    denom = max(float(np.sum(centered[:-1] ** 2)), EPS)
    return float(np.sum(centered[1:] * centered[:-1]) / denom)


def _quantiles(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(9, dtype=np.float64)
    return np.quantile(values, np.linspace(0.1, 0.9, 9))


def _sigmoid(value: float) -> float:
    clipped = max(min(float(value), 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _safe_len(values: object) -> int | None:
    try:
        return int(len(values))  # type: ignore[arg-type]
    except TypeError:
        return None


def _unpack_infer_item(item: object) -> tuple[Sequence[float], Iterable[float]]:
    row = tuple(item)  # type: ignore[arg-type]
    if len(row) == 2:
        return row[0], row[1]  # type: ignore[return-value]
    if len(row) == 3:
        return row[1], row[2]  # type: ignore[return-value]
    raise ValueError(f"Expected infer item with 2 or 3 fields, got {len(row)}.")


def _fallback_model() -> dict[str, object]:
    features, names = extract_streaming_features(
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        online_index=0,
        online_length=1,
    )
    return {
        "feature_names": names,
        "mean": np.zeros_like(features),
        "scale": np.ones_like(features),
        "coef": np.zeros_like(features),
        "intercept": 0.0,
        "alpha": 0.0,
        "metadata": {"model": "fallback_zero_evidence"},
    }
