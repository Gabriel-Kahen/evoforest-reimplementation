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
        registry.register("cusum_basic", cusum_basic_factory)
        registry.register("spectral_basic", spectral_basic_factory)
        registry.register("identity_callable", identity_callable_factory)
        registry.register("sigmoid_gate_callable", sigmoid_gate_callable_factory)
        registry.register("clipped_linear_callable", clipped_linear_callable_factory)
        registry.register("pass_outputs", pass_outputs_factory)
        registry.register("activated_outputs", activated_outputs_factory)
        registry.register("projection_outputs", projection_outputs_factory)
        registry.register("uniform_sample_weight", uniform_sample_weight_factory)
        registry.register("boundary_energy_weight", boundary_energy_weight_factory)
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
