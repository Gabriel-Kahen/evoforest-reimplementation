from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from evoforest_arch.graph import CallableFamily, EvalContext, FeatureBlock, NodeAlternative, ResidualWeightRule


PrimitiveFactory = Callable[[str, tuple[str, ...]], NodeAlternative]


@dataclass
class PrimitiveRegistry:
    factories: dict[str, PrimitiveFactory]

    @classmethod
    def default(cls) -> "PrimitiveRegistry":
        registry = cls(factories={})
        registry.register("segment_basic", segment_basic_factory)
        registry.register("segment_robust", segment_robust_factory)
        registry.register("trend_basic", trend_basic_factory)
        registry.register("trend_late_window", trend_late_window_factory)
        registry.register("cusum_basic", cusum_basic_factory)
        registry.register("spectral_basic", spectral_basic_factory)
        registry.register("segment_late_shift", segment_late_shift_factory)
        registry.register("shape_drawdown", shape_drawdown_factory)
        registry.register("shape_post_concentration", shape_post_concentration_factory)
        registry.register("row_recent_change", row_recent_change_factory)
        registry.register("row_volatility_burst", row_volatility_burst_factory)
        registry.register("row_cusum_local", row_cusum_local_factory)
        registry.register("row_context_outputs", row_context_outputs_factory)
        registry.register("row_local_outputs", row_local_outputs_factory)
        registry.register("row_time_basis_outputs", row_time_basis_outputs_factory)
        registry.register("row_multiscale_tail_outputs", row_multiscale_tail_outputs_factory)
        registry.register("adia_row_baseline_outputs", adia_row_baseline_outputs_factory)
        registry.register("identity_callable", identity_callable_factory)
        registry.register("sigmoid_gate_callable", sigmoid_gate_callable_factory)
        registry.register("clipped_linear_callable", clipped_linear_callable_factory)
        registry.register("pass_outputs", pass_outputs_factory)
        registry.register("activated_outputs", activated_outputs_factory)
        registry.register("projection_outputs", projection_outputs_factory)
        registry.register("competition_event_outputs", competition_event_outputs_factory)
        registry.register("adia_structural_break_baseline_outputs", adia_structural_break_baseline_outputs_factory)
        registry.register("interaction_outputs", interaction_outputs_factory)
        registry.register("uniform_sample_weight", uniform_sample_weight_factory)
        registry.register("boundary_energy_weight", boundary_energy_weight_factory)
        registry.register("late_energy_weight", late_energy_weight_factory)
        registry.register("identity_residual_weight", identity_residual_weight_factory)
        registry.register("huber_residual_weight", huber_residual_weight_factory)
        return registry

    def register(self, name: str, factory: PrimitiveFactory) -> None:
        self.factories[name] = factory

    def build(self, primitive: str, alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
        if primitive not in self.factories:
            raise KeyError(f"Unknown primitive {primitive!r}.")
        alternative = self.factories[primitive](alternative_id, parents)
        alternative.primitive = primitive
        return alternative


def split_segments(values: np.ndarray, boundary: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    return x[:, :boundary], x[:, boundary:]


def split_segments_torch(values: object, boundary: int) -> tuple[object, object]:
    return values[:, :boundary], values[:, boundary:]


def safe_std(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.maximum(np.std(x, axis=axis), 1e-8)


def safe_std_torch(x: object, axis: int = 1) -> object:
    import torch

    return torch.clamp(torch.std(x, dim=axis, unbiased=False), min=1e-8)


def slope(x: np.ndarray) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, x.shape[1])
    centered_t = t - np.mean(t)
    centered_x = x - np.mean(x, axis=1, keepdims=True)
    denom = np.sum(centered_t**2)
    return (centered_x @ centered_t) / max(float(denom), 1e-8)


def slope_torch(x: object) -> object:
    import torch

    t = torch.linspace(-1.0, 1.0, x.shape[1], dtype=x.dtype, device=x.device)
    centered_t = t - torch.mean(t)
    centered_x = x - torch.mean(x, dim=1, keepdim=True)
    denom = torch.clamp(torch.sum(centered_t**2), min=1e-8)
    return (centered_x @ centered_t) / denom


def segment_basic_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        pre_mean = np.mean(pre, axis=1)
        post_mean = np.mean(post, axis=1)
        pre_std = safe_std(pre)
        post_std = safe_std(post)
        block = np.column_stack(
            [
                post_mean - pre_mean,
                np.abs(post_mean - pre_mean),
                np.log(post_std / pre_std),
                np.abs(np.log(post_std / pre_std)),
            ]
        )
        return FeatureBlock(block, ["mean_delta", "mean_delta_abs", "std_log_ratio", "std_log_ratio_abs"])

    def tfn(ctx: EvalContext, values: dict[str, object]) -> object:
        import torch

        series = values[parents[0]]
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments_torch(series, boundary)
        pre_mean = torch.mean(pre, dim=1)
        post_mean = torch.mean(post, dim=1)
        pre_std = safe_std_torch(pre)
        post_std = safe_std_torch(post)
        delta = post_mean - pre_mean
        log_ratio = torch.log(post_std / pre_std)
        return torch.column_stack([delta, torch.abs(delta), log_ratio, torch.abs(log_ratio)])

    return NodeAlternative(alternative_id, parents, fn, "Mean and variance segment statistics.", torch_fn=tfn)


def segment_robust_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        pre_q = np.quantile(pre, [0.25, 0.5, 0.75], axis=1).T
        post_q = np.quantile(post, [0.25, 0.5, 0.75], axis=1).T
        iqr_ratio = np.log(np.maximum(post_q[:, 2] - post_q[:, 0], 1e-8) / np.maximum(pre_q[:, 2] - pre_q[:, 0], 1e-8))
        return FeatureBlock(
            np.column_stack([post_q[:, 1] - pre_q[:, 1], np.abs(post_q[:, 1] - pre_q[:, 1]), iqr_ratio, np.abs(iqr_ratio)]),
            ["median_delta", "median_delta_abs", "iqr_log_ratio", "iqr_log_ratio_abs"],
        )

    def tfn(ctx: EvalContext, values: dict[str, object]) -> object:
        import torch

        series = values[parents[0]]
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments_torch(series, boundary)
        qs = torch.tensor([0.25, 0.5, 0.75], dtype=series.dtype, device=series.device)
        pre_q = torch.quantile(pre, qs, dim=1).T
        post_q = torch.quantile(post, qs, dim=1).T
        iqr_ratio = torch.log(torch.clamp(post_q[:, 2] - post_q[:, 0], min=1e-8) / torch.clamp(pre_q[:, 2] - pre_q[:, 0], min=1e-8))
        median_delta = post_q[:, 1] - pre_q[:, 1]
        return torch.column_stack([median_delta, torch.abs(median_delta), iqr_ratio, torch.abs(iqr_ratio)])

    return NodeAlternative(alternative_id, parents, fn, "Robust quantile segment statistics.", torch_fn=tfn)


def trend_basic_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        pre_slope = slope(pre)
        post_slope = slope(post)
        delta = post_slope - pre_slope
        return FeatureBlock(np.column_stack([pre_slope, post_slope, delta, np.abs(delta)]), ["pre_slope", "post_slope", "slope_delta", "slope_delta_abs"])

    def tfn(ctx: EvalContext, values: dict[str, object]) -> object:
        import torch

        series = values[parents[0]]
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments_torch(series, boundary)
        pre_slope = slope_torch(pre)
        post_slope = slope_torch(post)
        delta = post_slope - pre_slope
        return torch.column_stack([pre_slope, post_slope, delta, torch.abs(delta)])

    return NodeAlternative(alternative_id, parents, fn, "Linear trend summaries before and after boundary.", torch_fn=tfn)


def trend_late_window_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        pre_tail = _tail_window(pre)
        post_head = _head_window(post)
        post_tail = _tail_window(post)
        pre_tail_slope = slope(pre_tail)
        post_head_slope = slope(post_head)
        post_tail_slope = slope(post_tail)
        block = np.column_stack(
            [
                post_tail_slope,
                post_tail_slope - pre_tail_slope,
                post_tail_slope - post_head_slope,
                np.abs(post_tail_slope - post_head_slope),
            ]
        )
        return FeatureBlock(block, ["post_tail_slope", "tail_pre_slope_delta", "post_slope_accel", "post_slope_accel_abs"])

    return NodeAlternative(alternative_id, parents, fn, "Late-window trend and acceleration summaries.")


def cusum_basic_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        centered = series - np.mean(series, axis=1, keepdims=True)
        cumulative = np.cumsum(centered, axis=1)
        peak = np.max(np.abs(cumulative), axis=1) / np.maximum(safe_std(series), 1e-8)
        boundary = int(ctx.read_input("boundary"))
        boundary_strength = np.abs(cumulative[:, boundary - 1]) / np.maximum(np.max(np.abs(cumulative), axis=1), 1e-8)
        return FeatureBlock(np.column_stack([peak, boundary_strength]), ["cusum_peak", "boundary_alignment"])

    def tfn(ctx: EvalContext, values: dict[str, object]) -> object:
        import torch

        series = values[parents[0]]
        centered = series - torch.mean(series, dim=1, keepdim=True)
        cumulative = torch.cumsum(centered, dim=1)
        peak = torch.max(torch.abs(cumulative), dim=1).values / torch.clamp(safe_std_torch(series), min=1e-8)
        boundary = int(ctx.read_input("boundary"))
        boundary_strength = torch.abs(cumulative[:, boundary - 1]) / torch.clamp(torch.max(torch.abs(cumulative), dim=1).values, min=1e-8)
        return torch.column_stack([peak, boundary_strength])

    return NodeAlternative(alternative_id, parents, fn, "CUSUM profile statistics.", torch_fn=tfn)


def segment_late_shift_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        pre_tail = _tail_window(pre)
        post_head = _head_window(post)
        post_tail = _tail_window(post)
        pre_tail_mean = np.mean(pre_tail, axis=1)
        post_head_mean = np.mean(post_head, axis=1)
        post_tail_mean = np.mean(post_tail, axis=1)
        pre_tail_std = safe_std(pre_tail)
        post_tail_std = safe_std(post_tail)
        tail_jump = post_head_mean - pre_tail_mean
        late_drift = post_tail_mean - post_head_mean
        block = np.column_stack(
            [
                tail_jump,
                np.abs(tail_jump),
                late_drift,
                np.abs(late_drift),
                np.log(post_tail_std / pre_tail_std),
            ]
        )
        return FeatureBlock(block, ["tail_jump", "tail_jump_abs", "post_late_drift", "post_late_drift_abs", "tail_std_log_ratio"])

    return NodeAlternative(alternative_id, parents, fn, "Competition-style late shift statistics around the period boundary.")


def shape_drawdown_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        post_drawdown, post_drawup = _drawdown_drawup(post)
        pre_drawdown, pre_drawup = _drawdown_drawup(pre)
        block = np.column_stack(
            [
                post_drawdown,
                post_drawup,
                post_drawdown - pre_drawdown,
                post_drawup - pre_drawup,
                np.abs(post_drawdown - pre_drawdown),
                np.abs(post_drawup - pre_drawup),
            ]
        )
        return FeatureBlock(block, ["post_drawdown", "post_drawup", "drawdown_delta", "drawup_delta", "drawdown_delta_abs", "drawup_delta_abs"])

    return NodeAlternative(alternative_id, parents, fn, "Post-period drawdown and drawup profile statistics.")


def shape_post_concentration_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        post_centered = post - np.mean(post, axis=1, keepdims=True)
        pre_centered = pre - np.mean(pre, axis=1, keepdims=True)
        post_energy = np.abs(post_centered)
        pre_energy = np.abs(pre_centered)
        post_total = np.maximum(np.sum(post_energy, axis=1), 1e-8)
        pre_total = np.maximum(np.sum(pre_energy, axis=1), 1e-8)
        tail = max(2, post.shape[1] // 4)
        post_tail_share = np.sum(post_energy[:, -tail:], axis=1) / post_total
        pre_tail_share = np.sum(pre_energy[:, -tail:], axis=1) / pre_total
        time = np.linspace(0.0, 1.0, post.shape[1])
        post_centroid = np.sum(post_energy * time.reshape(1, -1), axis=1) / post_total
        post_peak = np.argmax(post_energy, axis=1).astype(np.float64) / max(post.shape[1] - 1, 1)
        block = np.column_stack(
            [
                post_tail_share,
                post_tail_share - pre_tail_share,
                post_centroid,
                post_peak,
                np.maximum(post_centroid - 0.5, 0.0),
            ]
        )
        return FeatureBlock(block, ["post_tail_energy_share", "tail_energy_delta", "post_energy_centroid", "post_peak_location", "late_centroid_excess"])

    return NodeAlternative(alternative_id, parents, fn, "Concentration of post-period movement near the late horizon.")


def row_recent_change_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        older, recent = split_segments(series, boundary)
        older_tail = _tail_window(older)
        recent_head = _head_window(recent)
        recent_tail = _tail_window(recent)
        older_tail_mean = np.mean(older_tail, axis=1)
        recent_head_mean = np.mean(recent_head, axis=1)
        recent_tail_mean = np.mean(recent_tail, axis=1)
        recent_full_mean = np.mean(recent, axis=1)
        older_full_mean = np.mean(older, axis=1)
        jump = recent_head_mean - older_tail_mean
        drift = recent_tail_mean - recent_head_mean
        block = np.column_stack(
            [
                recent_full_mean - older_full_mean,
                np.abs(recent_full_mean - older_full_mean),
                jump,
                np.abs(jump),
                drift,
                np.abs(drift),
                series[:, -1] - older_tail_mean,
                series[:, -1] - recent_head_mean,
            ]
        )
        return FeatureBlock(
            block,
            [
                "recent_mean_delta",
                "recent_mean_delta_abs",
                "recent_entry_jump",
                "recent_entry_jump_abs",
                "recent_tail_drift",
                "recent_tail_drift_abs",
                "last_vs_older_tail",
                "last_vs_recent_head",
            ],
        )

    return NodeAlternative(alternative_id, parents, fn, "Row-local older-vs-recent level-change features for causal target rows.")


def row_volatility_burst_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        older, recent = split_segments(series, boundary)
        older_diff = np.diff(older, axis=1)
        recent_diff = np.diff(recent, axis=1)
        older_abs = np.abs(older_diff)
        recent_abs = np.abs(recent_diff)
        older_abs_mean = np.mean(older_abs, axis=1)
        recent_abs_mean = np.mean(recent_abs, axis=1)
        older_diff_std = safe_std(older_diff)
        recent_diff_std = safe_std(recent_diff)
        tail = max(2, recent.shape[1] // 4)
        recent_tail_abs = np.mean(np.abs(np.diff(recent[:, -tail:], axis=1)), axis=1)
        older_tail_abs = np.mean(np.abs(np.diff(older[:, -tail:], axis=1)), axis=1)
        burst = np.log((recent_abs_mean + 1e-8) / (older_abs_mean + 1e-8))
        burst_std = np.log(recent_diff_std / older_diff_std)
        tail_burst = np.log((recent_tail_abs + 1e-8) / (older_tail_abs + 1e-8))
        block = np.column_stack([burst, np.abs(burst), burst_std, np.abs(burst_std), tail_burst, np.abs(tail_burst)])
        return FeatureBlock(block, ["absdiff_log_ratio", "absdiff_log_ratio_abs", "diff_std_log_ratio", "diff_std_log_ratio_abs", "tail_absdiff_log_ratio", "tail_absdiff_log_ratio_abs"])

    return NodeAlternative(alternative_id, parents, fn, "Row-local volatility burst features from recent lookback increments.")


def row_cusum_local_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        centered = series - np.mean(series, axis=1, keepdims=True)
        cumulative = np.cumsum(centered, axis=1)
        peak_abs = np.max(np.abs(cumulative), axis=1)
        scale = np.maximum(safe_std(series), 1e-8)
        peak_location = np.argmax(np.abs(cumulative), axis=1).astype(np.float64) / max(series.shape[1] - 1, 1)
        recent_peak = np.max(np.abs(cumulative[:, boundary:]), axis=1)
        recent_share = recent_peak / np.maximum(peak_abs, 1e-8)
        terminal = cumulative[:, -1] / np.maximum(peak_abs, 1e-8)
        block = np.column_stack(
            [
                peak_abs / scale,
                peak_location,
                recent_share,
                terminal,
                np.maximum(peak_location - 0.5, 0.0),
            ]
        )
        return FeatureBlock(block, ["local_cusum_peak", "local_cusum_peak_location", "recent_cusum_share", "terminal_cusum_balance", "late_cusum_excess"])

    return NodeAlternative(alternative_id, parents, fn, "Row-local CUSUM profile features over the causal lookback window.")


def row_context_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        block, names = _row_context_features(ctx, series.shape[0], series.shape[1])
        return FeatureBlock(block, names)

    return NodeAlternative(alternative_id, parents, fn, "Optional row metadata features such as sample time, period, and observed lookback length.")


def row_local_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        older, recent = split_segments(series, boundary)
        recent_tail = _tail_window(recent)
        older_tail = _tail_window(older)
        full_slope = slope(series)
        older_slope = slope(older)
        recent_slope = slope(recent)
        recent_tail_slope = slope(recent_tail)
        drawdown, drawup = _drawdown_drawup(recent)
        recent_std = safe_std(recent)
        older_std = safe_std(older)
        last = series[:, -1]
        older_mean = np.mean(older, axis=1)
        recent_mean = np.mean(recent, axis=1)
        tail_delta = np.mean(recent_tail, axis=1) - np.mean(older_tail, axis=1)
        std_delta = np.log(recent_std / older_std)
        context_block, context_names = _row_context_features(ctx, series.shape[0], series.shape[1])
        block = np.column_stack(
            [
                last,
                last - older_mean,
                last - recent_mean,
                recent_mean - older_mean,
                tail_delta,
                np.abs(tail_delta),
                std_delta,
                np.abs(std_delta),
                full_slope,
                recent_slope - older_slope,
                recent_tail_slope - recent_slope,
                drawdown,
                drawup,
                context_block,
            ]
        )
        names = [
            "last_value",
            "last_vs_older_mean",
            "last_vs_recent_mean",
            "recent_vs_older_mean",
            "recent_tail_vs_older_tail",
            "recent_tail_vs_older_tail_abs",
            "recent_std_log_ratio",
            "recent_std_log_ratio_abs",
            "full_window_slope",
            "recent_slope_delta",
            "recent_tail_slope_delta",
            "recent_drawdown",
            "recent_drawup",
            *context_names,
        ]
        return FeatureBlock(block, names)

    return NodeAlternative(alternative_id, parents, fn, "Always-evaluated row-local features for target-time lookback windows.")


def row_time_basis_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        block, names = _row_time_basis_features(ctx, series.shape[0], series.shape[1])
        return FeatureBlock(block, names)

    return NodeAlternative(alternative_id, parents, fn, "Expanded row time and observed-lookback basis features.")


def row_multiscale_tail_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        older, recent = split_segments(series, boundary)
        features, names = _row_multiscale_tail_features(older, recent)
        return FeatureBlock(features, names)

    return NodeAlternative(alternative_id, parents, fn, "Multiscale recent-tail drift, volatility, slope, and drawdown features.")


def adia_row_baseline_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        older, recent = split_segments(series, boundary)
        older_tail = _tail_window(older)
        recent_head = _head_window(recent)
        recent_tail = _tail_window(recent)
        older_mean = np.mean(older, axis=1)
        recent_mean = np.mean(recent, axis=1)
        older_std = safe_std(older)
        recent_std = safe_std(recent)
        older_diff = np.diff(older, axis=1)
        recent_diff = np.diff(recent, axis=1)
        recent_drawdown, recent_drawup = _drawdown_drawup(recent)
        centered = series - np.mean(series, axis=1, keepdims=True)
        cumulative = np.cumsum(centered, axis=1)
        peak_abs = np.maximum(np.max(np.abs(cumulative), axis=1), 1e-8)
        context_block, context_names = _row_context_features(ctx, series.shape[0], series.shape[1])
        features = np.column_stack(
            [
                series[:, -1],
                series[:, -1] - older_mean,
                series[:, -1] - recent_mean,
                recent_mean - older_mean,
                np.mean(recent_head, axis=1) - np.mean(older_tail, axis=1),
                np.mean(recent_tail, axis=1) - np.mean(recent_head, axis=1),
                np.log(recent_std / older_std),
                np.mean(np.abs(recent_diff), axis=1) - np.mean(np.abs(older_diff), axis=1),
                np.log((np.mean(np.abs(recent_diff), axis=1) + 1e-8) / (np.mean(np.abs(older_diff), axis=1) + 1e-8)),
                slope(series),
                slope(recent) - slope(older),
                slope(recent_tail),
                recent_drawdown,
                recent_drawup,
                peak_abs / np.maximum(safe_std(series), 1e-8),
                np.argmax(np.abs(cumulative), axis=1).astype(np.float64) / max(series.shape[1] - 1, 1),
                np.max(np.abs(cumulative[:, boundary:]), axis=1) / peak_abs,
                context_block,
            ]
        )
        names = [
            "last_value",
            "last_vs_older_mean",
            "last_vs_recent_mean",
            "recent_vs_older_mean",
            "recent_entry_jump",
            "recent_tail_drift",
            "recent_std_log_ratio",
            "absdiff_delta",
            "absdiff_log_ratio",
            "full_window_slope",
            "recent_slope_delta",
            "recent_tail_slope",
            "recent_drawdown",
            "recent_drawup",
            "local_cusum_peak",
            "local_cusum_peak_location",
            "recent_cusum_share",
            *context_names,
        ]
        return FeatureBlock(features, names)

    return NodeAlternative(alternative_id, parents, fn, "Always-evaluated ADIA row-level baseline feature block.")


def spectral_basic_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        pre_fft = np.abs(np.fft.rfft(pre, axis=1))
        post_fft = np.abs(np.fft.rfft(post, axis=1))
        pre_low = np.mean(pre_fft[:, 1:4], axis=1)
        post_low = np.mean(post_fft[:, 1:4], axis=1)
        pre_high = np.mean(pre_fft[:, 4:], axis=1)
        post_high = np.mean(post_fft[:, 4:], axis=1)
        ratio_delta = np.log((post_high + 1e-8) / (post_low + 1e-8)) - np.log((pre_high + 1e-8) / (pre_low + 1e-8))
        return FeatureBlock(np.column_stack([ratio_delta, np.abs(ratio_delta)]), ["high_low_delta", "high_low_delta_abs"])

    def tfn(ctx: EvalContext, values: dict[str, object]) -> object:
        import torch

        series = values[parents[0]]
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments_torch(series, boundary)
        pre_fft = torch.abs(torch.fft.rfft(pre, dim=1))
        post_fft = torch.abs(torch.fft.rfft(post, dim=1))
        pre_low = torch.mean(pre_fft[:, 1:4], dim=1)
        post_low = torch.mean(post_fft[:, 1:4], dim=1)
        pre_high = torch.mean(pre_fft[:, 4:], dim=1)
        post_high = torch.mean(post_fft[:, 4:], dim=1)
        ratio_delta = torch.log((post_high + 1e-8) / (post_low + 1e-8)) - torch.log((pre_high + 1e-8) / (pre_low + 1e-8))
        return torch.column_stack([ratio_delta, torch.abs(ratio_delta)])

    return NodeAlternative(alternative_id, parents, fn, "Frequency-band ratio change statistics.", torch_fn=tfn)


def identity_callable_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(_ctx: EvalContext, _values: dict[str, object]) -> CallableFamily:
        return CallableFamily("identity", lambda x: x, "No-op callable family.")

    def tfn(_ctx: EvalContext, _values: dict[str, object]) -> CallableFamily:
        return CallableFamily("identity", lambda x: x, "No-op callable family.")

    return NodeAlternative(alternative_id, parents, fn, "Identity callable.", torch_fn=tfn)


def sigmoid_gate_callable_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, _values: dict[str, object]) -> CallableFamily:
        scale = float(ctx.globals.get("gate_scale")[0])
        return CallableFamily("sigmoid_gate", lambda x: 1.0 / (1.0 + np.exp(-scale * x)), "Sigmoid gate callable.")

    def tfn(ctx: EvalContext, _values: dict[str, object]) -> CallableFamily:
        import torch

        scale = ctx.globals.get("gate_scale").reshape(())
        return CallableFamily("sigmoid_gate", lambda x: 1.0 / (1.0 + torch.exp(-scale * x)), "Sigmoid gate callable.")

    return NodeAlternative(alternative_id, parents, fn, "Sigmoid gate callable.", global_refs=("gate_scale",), torch_fn=tfn)


def clipped_linear_callable_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, _values: dict[str, object]) -> CallableFamily:
        scale = float(ctx.globals.get("gate_scale")[0])
        return CallableFamily("clipped_linear", lambda x: np.clip(0.5 + scale * x, 0.0, 1.0), "Clipped linear activation.")

    def tfn(ctx: EvalContext, _values: dict[str, object]) -> CallableFamily:
        import torch

        scale = ctx.globals.get("gate_scale").reshape(())
        return CallableFamily("clipped_linear", lambda x: torch.clamp(0.5 + scale * x, 0.0, 1.0), "Clipped linear activation.")

    return NodeAlternative(alternative_id, parents, fn, "Clipped linear callable.", global_refs=("gate_scale",), torch_fn=tfn)


def combine_blocks(values: dict[str, object], parents: tuple[str, ...]) -> FeatureBlock:
    blocks = [value if isinstance(value, FeatureBlock) else FeatureBlock(np.asarray(value), [parent]) for parent, value in ((parent, values[parent]) for parent in parents)]
    return FeatureBlock(
        np.column_stack([block.values for block in blocks]),
        [name for block in blocks for name in block.names],
    )


def combine_torch_blocks(values: dict[str, object], parents: tuple[str, ...]) -> object:
    import torch

    blocks = []
    for parent in parents:
        value = values[parent]
        if value.ndim == 1:
            value = value.reshape(-1, 1)
        blocks.append(value)
    return torch.column_stack(blocks)


def pass_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        return combine_blocks(values, parents)

    def tfn(_ctx: EvalContext, values: dict[str, object]) -> object:
        return combine_torch_blocks(values, parents)

    return NodeAlternative(alternative_id, parents, fn, "Concatenate parent feature blocks.", torch_fn=tfn)


def activated_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        block = combine_blocks(values, parents[:-1])
        family = values[parents[-1]]
        if not isinstance(family, CallableFamily):
            raise TypeError("Last parent must be a CallableFamily.")
        activated = family.apply(block.values)
        return FeatureBlock(activated, [f"{family.name}_{name}" for name in block.names])

    def tfn(_ctx: EvalContext, values: dict[str, object]) -> object:
        block = combine_torch_blocks(values, parents[:-1])
        family = values[parents[-1]]
        if not isinstance(family, CallableFamily):
            raise TypeError("Last parent must be a CallableFamily.")
        return family.apply(block)

    return NodeAlternative(alternative_id, parents, fn, "Apply callable family to parent features.", torch_fn=tfn)


def projection_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        block = combine_blocks(values, parents)
        vector = ctx.globals.get("projection_vector")
        usable = vector[: block.values.shape[1]]
        projected = block.values[:, : usable.shape[0]] @ usable.reshape(-1, 1)
        return FeatureBlock(projected, ["global_projection"])

    def tfn(ctx: EvalContext, values: dict[str, object]) -> object:
        block = combine_torch_blocks(values, parents)
        vector = ctx.globals.get("projection_vector")
        usable = vector[: block.shape[1]]
        return block[:, : usable.shape[0]] @ usable.reshape(-1, 1)

    return NodeAlternative(alternative_id, parents, fn, "Projection using persistent global parameters.", global_refs=("projection_vector",), torch_fn=tfn)


def competition_event_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        pre_tail = _tail_window(pre)
        post_head = _head_window(post)
        post_tail = _tail_window(post)
        pre_tail_mean = np.mean(pre_tail, axis=1)
        post_head_mean = np.mean(post_head, axis=1)
        post_tail_mean = np.mean(post_tail, axis=1)
        tail_jump = post_head_mean - pre_tail_mean
        late_drift = post_tail_mean - post_head_mean
        post_drawdown, post_drawup = _drawdown_drawup(post)
        pre_drawdown, pre_drawup = _drawdown_drawup(pre)
        post_centered = post - np.mean(post, axis=1, keepdims=True)
        pre_centered = pre - np.mean(pre, axis=1, keepdims=True)
        post_energy = np.abs(post_centered)
        pre_energy = np.abs(pre_centered)
        post_total = np.maximum(np.sum(post_energy, axis=1), 1e-8)
        pre_total = np.maximum(np.sum(pre_energy, axis=1), 1e-8)
        tail = max(2, post.shape[1] // 4)
        post_tail_share = np.sum(post_energy[:, -tail:], axis=1) / post_total
        pre_tail_share = np.sum(pre_energy[:, -tail:], axis=1) / pre_total
        post_time = np.linspace(0.0, 1.0, post.shape[1])
        post_centroid = np.sum(post_energy * post_time.reshape(1, -1), axis=1) / post_total
        block = np.column_stack(
            [
                tail_jump,
                np.abs(tail_jump),
                late_drift,
                np.abs(late_drift),
                post_drawdown,
                post_drawup,
                post_drawdown - pre_drawdown,
                post_drawup - pre_drawup,
                post_tail_share,
                post_tail_share - pre_tail_share,
                post_centroid,
                np.maximum(post_centroid - 0.5, 0.0),
            ]
        )
        return FeatureBlock(
            block,
            [
                "tail_jump",
                "tail_jump_abs",
                "post_late_drift",
                "post_late_drift_abs",
                "post_drawdown",
                "post_drawup",
                "drawdown_delta",
                "drawup_delta",
                "post_tail_energy_share",
                "tail_energy_delta",
                "post_energy_centroid",
                "late_centroid_excess",
            ],
        )

    return NodeAlternative(alternative_id, parents, fn, "Always-evaluated competition event detection output features.")


def adia_structural_break_baseline_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        pre, post = split_segments(series, boundary)
        pre_tail = _tail_window(pre)
        post_head = _head_window(post)
        post_tail = _tail_window(post)
        pre_d = np.diff(pre, axis=1)
        post_d = np.diff(post, axis=1)
        pre_mean = np.mean(pre, axis=1)
        post_mean = np.mean(post, axis=1)
        pre_std = safe_std(pre)
        post_std = safe_std(post)
        pre_iqr = np.quantile(pre, 0.75, axis=1) - np.quantile(pre, 0.25, axis=1)
        post_iqr = np.quantile(post, 0.75, axis=1) - np.quantile(post, 0.25, axis=1)
        pre_c = pre - pre_mean.reshape(-1, 1)
        post_c = post - post_mean.reshape(-1, 1)
        pre_cusum = np.cumsum(pre_c, axis=1)
        post_cusum = np.cumsum(post_c, axis=1)
        pre_fft = np.abs(np.fft.rfft(pre, axis=1))
        post_fft = np.abs(np.fft.rfft(post, axis=1))
        pre_low = _safe_band_mean(pre_fft, 1, 4)
        post_low = _safe_band_mean(post_fft, 1, 4)
        pre_high = _safe_band_mean(pre_fft, 4, pre_fft.shape[1])
        post_high = _safe_band_mean(post_fft, 4, post_fft.shape[1])
        quantiles = np.linspace(0.1, 0.9, 9)
        pre_q = np.quantile(pre, quantiles, axis=1).T
        post_q = np.quantile(post, quantiles, axis=1).T
        paired = max(1, min(pre.shape[1], post.shape[1]))
        sorted_pre = np.sort(pre, axis=1)[:, :paired]
        sorted_post = np.sort(post, axis=1)[:, :paired]
        post_drawdown, post_drawup = _drawdown_drawup(post)
        pre_drawdown, pre_drawup = _drawdown_drawup(pre)
        features = np.column_stack(
            [
                post_mean - pre_mean,
                np.abs(post_mean - pre_mean),
                (post_mean - pre_mean) / pre_std,
                np.median(post, axis=1) - np.median(pre, axis=1),
                np.log((post_std + 1e-8) / (pre_std + 1e-8)),
                post_iqr - pre_iqr,
                np.log((post_iqr + 1e-8) / (pre_iqr + 1e-8)),
                np.mean(np.abs(post_d), axis=1) - np.mean(np.abs(pre_d), axis=1),
                np.log((safe_std(post_d) + 1e-8) / (safe_std(pre_d) + 1e-8)),
                _autocorr1(post) - _autocorr1(pre),
                slope(post) - slope(pre),
                slope(post_tail) - slope(pre_tail),
                np.mean(post_head, axis=1) - np.mean(pre_tail, axis=1),
                np.mean(post_tail, axis=1) - np.mean(post_head, axis=1),
                np.mean(np.abs(post_q - pre_q), axis=1),
                np.max(np.abs(post_q - pre_q), axis=1),
                np.mean(sorted_post - sorted_pre, axis=1),
                np.mean(np.abs(sorted_post - sorted_pre), axis=1),
                np.max(np.abs(post_cusum), axis=1) / post_std,
                np.max(np.abs(pre_cusum), axis=1) / pre_std,
                np.max(np.abs(post_cusum), axis=1) / np.maximum(np.max(np.abs(pre_cusum), axis=1), 1e-8),
                np.log((post_low + 1e-8) / (pre_low + 1e-8)),
                np.log((post_high + 1e-8) / (pre_high + 1e-8)),
                np.log((post_high + 1e-8) / (post_low + 1e-8)) - np.log((pre_high + 1e-8) / (pre_low + 1e-8)),
                post_drawdown - pre_drawdown,
                post_drawup - pre_drawup,
                series[:, -1] - pre_mean,
                series[:, -1] - np.mean(pre_tail, axis=1),
            ]
        )
        names = [
            "adia_mean_delta",
            "adia_mean_delta_abs",
            "adia_mean_delta_scaled",
            "adia_median_delta",
            "adia_std_log_ratio",
            "adia_iqr_delta",
            "adia_iqr_log_ratio",
            "adia_absdiff_delta",
            "adia_diff_std_log_ratio",
            "adia_autocorr_delta",
            "adia_slope_delta",
            "adia_tail_slope_delta",
            "adia_boundary_jump",
            "adia_post_tail_drift",
            "adia_quantile_l1_distance",
            "adia_quantile_max_distance",
            "adia_sorted_signed_distance",
            "adia_sorted_l1_distance",
            "adia_post_cusum_peak",
            "adia_pre_cusum_peak",
            "adia_cusum_peak_ratio",
            "adia_low_freq_log_ratio",
            "adia_high_freq_log_ratio",
            "adia_spectral_shape_delta",
            "adia_drawdown_delta",
            "adia_drawup_delta",
            "adia_last_vs_pre_mean",
            "adia_last_vs_pre_tail",
        ]
        return FeatureBlock(features, names)

    return NodeAlternative(alternative_id, parents, fn, "Always-evaluated full ADIA structural-break baseline feature block.")


def interaction_outputs_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        block = combine_blocks(values, parents)
        x = block.values
        if x.shape[1] == 0:
            return FeatureBlock(x, [])
        centered = x - np.mean(x, axis=0, keepdims=True)
        std = np.maximum(np.std(centered, axis=0, keepdims=True), 1e-8)
        z = np.clip(centered / std, -5.0, 5.0)
        k = min(6, z.shape[1])
        selected = z[:, :k]
        aggregate = np.column_stack(
            [
                np.mean(selected, axis=1),
                np.max(selected, axis=1),
                np.min(selected, axis=1),
                np.mean(np.abs(selected), axis=1),
            ]
        )
        columns = [selected[:, idx] * selected[:, idx + 1] for idx in range(k - 1)]
        names = [f"z_interaction_{idx}_{idx + 1}" for idx in range(k - 1)]
        values_out = np.column_stack([aggregate, *columns]) if columns else aggregate
        return FeatureBlock(values_out, ["z_mean", "z_max", "z_min", "z_abs_mean", *names])

    return NodeAlternative(alternative_id, parents, fn, "Standardized aggregate and pairwise interaction output features.")


def uniform_sample_weight_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, _values: dict[str, object]) -> np.ndarray:
        series = np.asarray(ctx.read_input("series"), dtype=np.float64)
        return np.ones(series.shape[0], dtype=np.float64)

    return NodeAlternative(alternative_id, parents, fn, "Uniform Ridge sample weights.")


def boundary_energy_weight_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> np.ndarray:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        centered = series - np.mean(series, axis=1, keepdims=True)
        cumulative = np.cumsum(centered, axis=1)
        boundary_energy = np.abs(cumulative[:, boundary - 1])
        peak_energy = np.maximum(np.max(np.abs(cumulative), axis=1), 1e-8)
        weights = 0.5 + boundary_energy / peak_energy
        return np.clip(weights, 0.25, 3.0)

    return NodeAlternative(alternative_id, parents, fn, "Emphasize samples whose CUSUM energy concentrates near the boundary.")


def late_energy_weight_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, values: dict[str, object]) -> np.ndarray:
        series = np.asarray(values[parents[0]], dtype=np.float64)
        boundary = int(ctx.read_input("boundary"))
        _pre, post = split_segments(series, boundary)
        tail = max(2, post.shape[1] // 4)
        centered = post - np.mean(post, axis=1, keepdims=True)
        total = np.maximum(np.sum(np.abs(centered), axis=1), 1e-8)
        late_share = np.sum(np.abs(centered[:, -tail:]), axis=1) / total
        weights = 0.75 + late_share
        return np.clip(weights, 0.5, 2.5)

    return NodeAlternative(alternative_id, parents, fn, "Emphasize samples with movement concentrated late in the post period.")


def identity_residual_weight_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(_ctx: EvalContext, _values: dict[str, object]) -> ResidualWeightRule:
        return ResidualWeightRule("identity", lambda residual: np.ones_like(residual, dtype=np.float64), "No residual reweighting.")

    return NodeAlternative(alternative_id, parents, fn, "Identity residual weighting rule.")


def huber_residual_weight_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
    def fn(ctx: EvalContext, _values: dict[str, object]) -> ResidualWeightRule:
        scale = float(np.maximum(ctx.globals.get("residual_huber_scale")[0], 1e-6))

        def apply(residual: np.ndarray) -> np.ndarray:
            abs_residual = np.abs(np.asarray(residual, dtype=np.float64))
            return np.where(abs_residual <= scale, 1.0, scale / np.maximum(abs_residual, 1e-6))

        return ResidualWeightRule("huber", apply, "Huber-style residual downweighting.")

    return NodeAlternative(alternative_id, parents, fn, "Huber-style residual weighting rule.", global_refs=("residual_huber_scale",))


def _head_window(x: np.ndarray) -> np.ndarray:
    width = max(2, x.shape[1] // 4)
    return x[:, :width]


def _tail_window(x: np.ndarray) -> np.ndarray:
    width = max(2, x.shape[1] // 4)
    return x[:, -width:]


def _drawdown_drawup(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cumulative_max = np.maximum.accumulate(x, axis=1)
    cumulative_min = np.minimum.accumulate(x, axis=1)
    drawdown = np.max(cumulative_max - x, axis=1)
    drawup = np.max(x - cumulative_min, axis=1)
    scale = np.maximum(safe_std(x), 1e-8)
    return drawdown / scale, drawup / scale


def _autocorr1(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=1, keepdims=True)
    return np.sum(centered[:, 1:] * centered[:, :-1], axis=1) / np.maximum(np.sum(centered[:, :-1] ** 2, axis=1), 1e-8)


def _safe_band_mean(x: np.ndarray, start: int, stop: int) -> np.ndarray:
    start = min(max(0, int(start)), x.shape[1] - 1)
    stop = min(max(start + 1, int(stop)), x.shape[1])
    return np.mean(x[:, start:stop], axis=1)


def _row_context_features(ctx: EvalContext, n_rows: int, series_length: int) -> tuple[np.ndarray, list[str]]:
    sample_time = _optional_input_vector(ctx, "sample_time", n_rows)
    sample_period = _optional_input_vector(ctx, "sample_period", n_rows)
    observed = _optional_input_vector(ctx, "lookback_observed", n_rows)
    sample_time_scale = _optional_input_vector(ctx, "sample_time_scale", n_rows)
    if np.any(sample_time_scale > 0.0):
        time_scale = max(float(np.max(sample_time_scale)), 1.0)
    else:
        time_scale = max(float(np.max(sample_time)) if sample_time.size else 0.0, 1.0)
    time_norm = sample_time / time_scale
    log_time_norm = np.log1p(np.maximum(sample_time, 0.0)) / max(float(np.log1p(time_scale)), 1.0)
    period_two = (sample_period >= 2.0).astype(np.float64)
    observed_fraction = np.clip(observed / max(float(series_length), 1.0), 0.0, 1.0)
    return (
        np.column_stack([time_norm, log_time_norm, period_two, observed_fraction]),
        ["sample_time_norm", "sample_time_log_norm", "sample_period_two", "lookback_observed_fraction"],
    )


def _row_time_basis_features(ctx: EvalContext, n_rows: int, series_length: int) -> tuple[np.ndarray, list[str]]:
    sample_time = _optional_input_vector(ctx, "sample_time", n_rows)
    sample_period = _optional_input_vector(ctx, "sample_period", n_rows)
    observed = _optional_input_vector(ctx, "lookback_observed", n_rows)
    sample_time_scale = _optional_input_vector(ctx, "sample_time_scale", n_rows)
    if np.any(sample_time_scale > 0.0):
        time_scale = max(float(np.max(sample_time_scale)), 1.0)
    else:
        time_scale = max(float(np.max(sample_time)) if sample_time.size else 0.0, 1.0)
    time_norm = sample_time / time_scale
    log_time_norm = np.log1p(np.maximum(sample_time, 0.0)) / max(float(np.log1p(time_scale)), 1.0)
    sqrt_time_norm = np.sqrt(np.maximum(time_norm, 0.0))
    observed_fraction = np.clip(observed / max(float(series_length), 1.0), 0.0, 1.0)
    block = np.column_stack(
        [
            time_norm,
            log_time_norm,
            time_norm * time_norm,
            time_norm * time_norm * time_norm,
            sqrt_time_norm,
            np.sin(np.pi * time_norm),
            np.cos(np.pi * time_norm),
            np.sin(2.0 * np.pi * time_norm),
            np.cos(2.0 * np.pi * time_norm),
            observed_fraction,
            observed_fraction * observed_fraction,
            time_norm * observed_fraction,
            (sample_period >= 2.0).astype(np.float64),
        ]
    )
    return (
        block,
        [
            "time_basis_norm",
            "time_basis_log_norm",
            "time_basis_sq",
            "time_basis_cube",
            "time_basis_sqrt",
            "time_basis_sin1",
            "time_basis_cos1",
            "time_basis_sin2",
            "time_basis_cos2",
            "observed_basis_fraction",
            "observed_basis_sq",
            "time_observed_interaction",
            "time_basis_period_two",
        ],
    )


def _row_multiscale_tail_features(older: np.ndarray, recent: np.ndarray) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    last = recent[:, -1]
    seen_widths: set[int] = set()
    for requested_width in (4, 8, 16, 32, 64, 128, 240):
        width = min(int(requested_width), recent.shape[1])
        if width < 2 or width in seen_widths:
            continue
        seen_widths.add(width)
        tail = recent[:, -width:]
        previous = recent[:, max(0, recent.shape[1] - 2 * width) : recent.shape[1] - width]
        if previous.shape[1] < 2:
            previous = older[:, -min(width, older.shape[1]) :]
        old = older[:, -min(width, older.shape[1]) :]
        tail_mean = np.mean(tail, axis=1)
        previous_mean = np.mean(previous, axis=1)
        old_mean = np.mean(old, axis=1)
        tail_std = safe_std(tail)
        previous_std = safe_std(previous)
        old_std = safe_std(old)
        tail_diff = np.diff(tail, axis=1)
        previous_diff = np.diff(previous, axis=1)
        old_diff = np.diff(old, axis=1)
        tail_absdiff = np.mean(np.abs(tail_diff), axis=1)
        previous_absdiff = np.mean(np.abs(previous_diff), axis=1)
        old_absdiff = np.mean(np.abs(old_diff), axis=1)
        drawdown, drawup = _drawdown_drawup(tail)
        tail_slope = slope(tail)
        previous_slope = slope(previous)
        columns.extend(
            [
                last - tail_mean,
                tail_mean - previous_mean,
                tail_mean - old_mean,
                np.log(tail_std / previous_std),
                np.log(tail_std / old_std),
                tail_absdiff - previous_absdiff,
                np.log((tail_absdiff + 1e-8) / (previous_absdiff + 1e-8)),
                np.log((tail_absdiff + 1e-8) / (old_absdiff + 1e-8)),
                tail_slope,
                tail_slope - previous_slope,
                drawdown,
                drawup,
                np.max(tail, axis=1) - np.min(tail, axis=1),
            ]
        )
        names.extend(
            [
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
        )
    if not columns:
        return np.zeros((recent.shape[0], 0), dtype=np.float64), []
    return np.column_stack(columns), names


def _optional_input_vector(ctx: EvalContext, name: str, n_rows: int) -> np.ndarray:
    if name not in ctx.inputs:
        return np.zeros(n_rows, dtype=np.float64)
    value = np.asarray(ctx.inputs[name], dtype=np.float64).reshape(-1)
    if value.shape[0] != n_rows:
        return np.zeros(n_rows, dtype=np.float64)
    return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
