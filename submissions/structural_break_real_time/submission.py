from __future__ import annotations

import bisect
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
TAIL_WINDOWS = (8, 32, 128)
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
        selected_indices = set(int(idx) for idx in _training_indices(online.size, tau_index, 32))
        state = _StreamingFeatureState(historical, online.size)
        for idx, point in enumerate(online):
            if idx in selected_indices:
                features, names = state.append(float(point), emit=True)
                x_rows.append(features)
                y_rows.append(float(tau_index is not None and idx >= int(tau_index)))
            else:
                state.append(float(point), emit=False)

    if not x_rows:
        model = _fallback_model()
    else:
        x = np.vstack(x_rows)
        y = np.asarray(y_rows, dtype=np.float64)
        model = fit_streaming_ridge(x, y, names)

    model["metadata"] = {
        "model": "evoforest_realtime_lite_streaming_ridge",
        "series_length": 480,
        "max_rows_per_series": 32,
        "tail_windows": list(_tail_windows()),
        "feature_runtime": "incremental_prefix_stats_with_short_medium_long_tail_windows",
        "score_transform": "sigmoid(raw_ridge_prediction)",
    }
    with open(os.path.join(model_directory_path, "evoforest_realtime_model.pkl"), "wb") as handle:
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
        state = _StreamingFeatureState(historical, online_length)
        for online_index, point in enumerate(x_online):
            del online_index
            features, _names = state.append(float(point))
            raw = predict_raw(model, features)
            yield float(_sigmoid(raw))


def load_model(model_directory_path: str) -> dict[str, object]:
    path = os.path.join(model_directory_path, "evoforest_realtime_model.pkl")
    if not os.path.exists(path):
        return _fallback_model()
    with open(path, "rb") as handle:
        return pickle.load(handle)


def fit_streaming_ridge(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, object]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    z = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = select_alpha(z, y, np.logspace(-4, 4, 17))
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
    denom = max(int(online_length) - 1, 1) if online_length else 1000
    time_norm = min(max(float(online_index) / float(denom), 0.0), 1.0)
    observed_fraction = min(float(n_online) / 1000.0, 1.0)

    values = [
        last,
        (last - hist_mean) / hist_std,
        (last - online_mean) / online_std,
        (online_mean - hist_mean) / hist_std,
        math.log((online_std + 1e-8) / (hist_std + 1e-8)),
        online_absdiff - hist_absdiff,
        math.log((online_absdiff + 1e-8) / (hist_absdiff + 1e-8)),
        math.log((online_diff_std + 1e-8) / (hist_diff_std + 1e-8)),
        online_slope,
        online_slope - hist_slope,
        online_autocorr - hist_autocorr,
        online_drawdown,
        online_drawup,
        online_drawdown - hist_drawdown,
        online_drawup - hist_drawup,
        peak_abs / hist_std,
        float(np.argmax(np.abs(cumulative)) / max(cumulative.size - 1, 1)) if cumulative.size else 0.0,
        float(cumulative[-1] / max(peak_abs, 1e-8)) if cumulative.size else 0.0,
        float(np.mean(np.abs(online_q - hist_q))),
        float(np.max(np.abs(online_q - hist_q))),
        time_norm,
        math.log1p(max(float(online_index), 0.0)) / math.log1p(1000),
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

    for width in _tail_windows():
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
            (float(tail[-1]) - tail_mean) / max(tail_std, 1e-8),
            (tail_mean - previous_mean) / max(hist_std, 1e-8),
            (tail_mean - old_mean) / max(hist_std, 1e-8),
            math.log((tail_std + 1e-8) / (previous_std + 1e-8)),
            math.log((tail_std + 1e-8) / (old_std + 1e-8)),
            tail_absdiff - previous_absdiff,
            math.log((tail_absdiff + 1e-8) / (previous_absdiff + 1e-8)),
            math.log((tail_absdiff + 1e-8) / (old_absdiff + 1e-8)),
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


class _StreamingFeatureState:
    def __init__(self, historical: np.ndarray, online_length: int | None) -> None:
        self.historical = _as_1d(historical)
        self.online_length = online_length
        capacity = max(int(online_length) if online_length else 1000, 1)
        self.online = np.empty(capacity, dtype=np.float64)
        self.sorted_online: list[float] = []

        self.hist_mean = float(np.mean(self.historical)) if self.historical.size else 0.0
        self.hist_std = _safe_std_1d(self.historical)
        hist_diff = np.diff(self.historical) if self.historical.size > 1 else np.zeros(1, dtype=np.float64)
        self.hist_absdiff = float(np.mean(np.abs(hist_diff))) if hist_diff.size else 0.0
        self.hist_diff_std = _safe_std_1d(hist_diff)
        self.hist_slope = _slope_1d(self.historical)
        self.hist_autocorr = _autocorr1_1d(self.historical)
        self.hist_drawdown, self.hist_drawup = _drawdown_drawup_1d(self.historical)
        self.hist_q = _quantiles(self.historical)

        self.n = 0
        self.sum_x = 0.0
        self.sumsq_x = 0.0
        self.sum_i_x = 0.0
        self.first = 0.0
        self.last = 0.0
        self.diff_count = 0
        self.diff_sum = 0.0
        self.diff_sumsq = 0.0
        self.diff_abs_sum = 0.0
        self.adjacent_product_sum = 0.0
        self.running_max = 0.0
        self.running_min = 0.0
        self.max_drawdown_raw = 0.0
        self.max_drawup_raw = 0.0
        self.centered_cumsum = 0.0
        self.peak_abs_cumsum = 0.0
        self.peak_abs_index = 0

    def append(self, point: float, *, emit: bool = True) -> tuple[np.ndarray, list[str]] | None:
        value = float(point)
        if self.n >= self.online.shape[0]:
            grown = np.empty(max(self.online.shape[0] * 2, 1), dtype=np.float64)
            grown[: self.online.shape[0]] = self.online
            self.online = grown

        idx = int(self.n)
        self.online[idx] = value
        if idx == 0:
            self.first = value
            self.running_max = value
            self.running_min = value
        else:
            diff = value - self.last
            self.diff_count += 1
            self.diff_sum += diff
            self.diff_sumsq += diff * diff
            self.diff_abs_sum += abs(diff)
            self.adjacent_product_sum += value * self.last
            self.max_drawdown_raw = max(self.max_drawdown_raw, self.running_max - value)
            self.max_drawup_raw = max(self.max_drawup_raw, value - self.running_min)
            self.running_max = max(self.running_max, value)
            self.running_min = min(self.running_min, value)

        self.last = value
        self.sum_x += value
        self.sumsq_x += value * value
        self.sum_i_x += float(idx) * value
        self.centered_cumsum += value - self.hist_mean
        abs_cumsum = abs(self.centered_cumsum)
        if abs_cumsum > self.peak_abs_cumsum:
            self.peak_abs_cumsum = abs_cumsum
            self.peak_abs_index = idx
        bisect.insort(self.sorted_online, value)
        self.n += 1
        if not emit:
            return None
        return self.features()

    def features(self) -> tuple[np.ndarray, list[str]]:
        n_online = int(self.n)
        online = self.online[:n_online]
        last = float(self.last)
        online_mean = float(self.sum_x / max(n_online, 1))
        online_std = _safe_std_from_moments(n_online, self.sum_x, self.sumsq_x)
        online_absdiff = float(self.diff_abs_sum / self.diff_count) if self.diff_count else 0.0
        online_diff_std = _safe_std_from_moments(self.diff_count, self.diff_sum, self.diff_sumsq)
        online_slope = self._slope()
        online_autocorr = self._autocorr1()
        online_drawdown = float(self.max_drawdown_raw / max(online_std, 1e-8))
        online_drawup = float(self.max_drawup_raw / max(online_std, 1e-8))
        online_q = _quantiles_sorted(self.sorted_online)
        denom = max(int(self.online_length) - 1, 1) if self.online_length else 1000
        online_index = n_online - 1
        time_norm = min(max(float(online_index) / float(denom), 0.0), 1.0)
        observed_fraction = min(float(n_online) / 1000.0, 1.0)

        values = [
            last,
            (last - self.hist_mean) / self.hist_std,
            (last - online_mean) / online_std,
            (online_mean - self.hist_mean) / self.hist_std,
            math.log((online_std + 1e-8) / (self.hist_std + 1e-8)),
            online_absdiff - self.hist_absdiff,
            math.log((online_absdiff + 1e-8) / (self.hist_absdiff + 1e-8)),
            math.log((online_diff_std + 1e-8) / (self.hist_diff_std + 1e-8)),
            online_slope,
            online_slope - self.hist_slope,
            online_autocorr - self.hist_autocorr,
            online_drawdown,
            online_drawup,
            online_drawdown - self.hist_drawdown,
            online_drawup - self.hist_drawup,
            self.peak_abs_cumsum / self.hist_std,
            float(self.peak_abs_index / max(n_online - 1, 1)),
            float(self.centered_cumsum / max(self.peak_abs_cumsum, 1e-8)),
            float(np.mean(np.abs(online_q - self.hist_q))),
            float(np.max(np.abs(online_q - self.hist_q))),
            time_norm,
            math.log1p(max(float(online_index), 0.0)) / math.log1p(1000),
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
        for width in _tail_windows():
            tail_values, tail_names = _tail_feature_values(self.historical, online, width, self.hist_mean, self.hist_std)
            values.extend(tail_values)
            names.extend(tail_names)
        return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), names

    def _slope(self) -> float:
        n = int(self.n)
        if n < 2:
            return 0.0
        numerator = (2.0 * self.sum_i_x / float(n - 1)) - self.sum_x
        denom = float(n * (n + 1)) / (3.0 * float(n - 1))
        return float(numerator / max(denom, 1e-8))

    def _autocorr1(self) -> float:
        n = int(self.n)
        if n < 3:
            return 0.0
        mean = self.sum_x / float(n)
        sum_except_first = self.sum_x - self.first
        sum_except_last = self.sum_x - self.last
        sumsq_except_last = self.sumsq_x - self.last * self.last
        numerator = self.adjacent_product_sum - mean * (sum_except_first + sum_except_last) + float(n - 1) * mean * mean
        denom = sumsq_except_last - 2.0 * mean * sum_except_last + float(n - 1) * mean * mean
        return float(numerator / max(denom, 1e-8))


def _safe_std_from_moments(count: int, total: float, total_sq: float) -> float:
    if int(count) < 2:
        return 1.0
    mean = float(total) / float(count)
    variance = max(float(total_sq) / float(count) - mean * mean, 0.0)
    return max(math.sqrt(variance), 1e-8)


def _quantiles_sorted(sorted_values: list[float]) -> np.ndarray:
    n = len(sorted_values)
    if n == 0:
        return np.zeros(9, dtype=np.float64)
    values: list[float] = []
    for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        position = float(n - 1) * q
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            values.append(float(sorted_values[low]))
        else:
            fraction = position - float(low)
            values.append(float(sorted_values[low]) * (1.0 - fraction) + float(sorted_values[high]) * fraction)
    return np.asarray(values, dtype=np.float64)


def _tail_windows() -> tuple[int, ...]:
    return (8, 32, 128)


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
        residual = (y - pred) / np.maximum(1.0 - leverage, 1e-8)
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
    return max(float(np.std(values)), 1e-8)


def _slope_1d(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    t = np.linspace(-1.0, 1.0, values.size)
    centered_t = t - float(np.mean(t))
    centered_x = values - float(np.mean(values))
    denom = max(float(np.sum(centered_t**2)), 1e-8)
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
    denom = max(float(np.sum(centered[:-1] ** 2)), 1e-8)
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
