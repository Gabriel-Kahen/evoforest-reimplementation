from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evoforest_arch.graph import Graph
from evoforest_arch.mutations import MutationEngine, MutationSpec
from evoforest_arch.splits import subset_inputs


@dataclass(frozen=True)
class SourceMutationCheck:
    alternative_id: str
    passed: bool
    error: str = ""
    n_features: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "alternative_id": self.alternative_id,
            "passed": bool(self.passed),
            "error": self.error,
            "n_features": int(self.n_features),
        }


def structural_break_source_mutations() -> tuple[MutationSpec, ...]:
    """Source-backed structural-break feature templates.

    These templates approximate the paper's source-backed mutation surface while
    staying deterministic and reviewable. Real LLM mutations can use the same
    `MutationSpec(source=...)` path.
    """

    return (
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="source",
            alternative_id="source_distribution_shift",
            parents=("series",),
            source=SOURCE_SIMPLE_DISTRIBUTION_SHIFT,
            description="Source-backed robust distribution-shift features across the known boundary.",
            output_contract={"type": "feature_block", "n_columns": 4},
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="source",
            alternative_id="source_autocorr_volatility_shift",
            parents=("series",),
            source=SOURCE_SIMPLE_VOLATILITY_SHIFT,
            description="Source-backed autocorrelation, volatility, and increment-regime change features.",
            output_contract={"type": "feature_block", "n_columns": 3},
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="source",
            alternative_id="source_multiscale_tail_shift",
            parents=("series",),
            source=SOURCE_SIMPLE_TAIL_SHIFT,
            description="Source-backed multiscale tail/head structural-break features.",
            output_contract={"type": "feature_block", "n_columns": 3},
        ),
    )


def validate_source_mutations(
    graph: Graph,
    specs: tuple[MutationSpec, ...],
    inputs: dict[str, object],
    *,
    n_rows: int = 24,
) -> list[SourceMutationCheck]:
    n_samples = int(np.asarray(inputs["series"]).shape[0])
    indices = np.arange(min(max(1, int(n_rows)), n_samples), dtype=np.int64)
    smoke_inputs = subset_inputs(inputs, indices, n_samples=n_samples)
    checks: list[SourceMutationCheck] = []
    for spec in specs:
        try:
            candidate = MutationEngine(allow_source=True).apply(graph, spec)
            config = candidate.default_config()
            x, _names, _ctx = candidate.evaluate_features(smoke_inputs, config=config)
            checks.append(SourceMutationCheck(spec.alternative_id, True, n_features=int(x.shape[1])))
        except Exception as exc:
            checks.append(SourceMutationCheck(spec.alternative_id, False, error=str(exc)))
    return checks


SOURCE_SIMPLE_DISTRIBUTION_SHIFT = """lambda ctx, values: FeatureBlock(
    np.column_stack([
        np.mean(values["series"][:, int(ctx.read_input("boundary")):], axis=1) - np.mean(values["series"][:, :int(ctx.read_input("boundary"))], axis=1),
        np.abs(np.mean(values["series"][:, int(ctx.read_input("boundary")):], axis=1) - np.mean(values["series"][:, :int(ctx.read_input("boundary"))], axis=1)),
        np.log((np.std(values["series"][:, int(ctx.read_input("boundary")):], axis=1) + 1e-8) / (np.std(values["series"][:, :int(ctx.read_input("boundary"))], axis=1) + 1e-8)),
        np.median(values["series"][:, int(ctx.read_input("boundary")):], axis=1) - np.median(values["series"][:, :int(ctx.read_input("boundary"))], axis=1)
    ]),
    ["src_mean_delta", "src_mean_delta_abs", "src_std_log_ratio", "src_median_delta"]
)"""


SOURCE_SIMPLE_VOLATILITY_SHIFT = """lambda ctx, values: FeatureBlock(
    np.column_stack([
        np.std(np.diff(values["series"][:, int(ctx.read_input("boundary")):], axis=1), axis=1) - np.std(np.diff(values["series"][:, :int(ctx.read_input("boundary"))], axis=1), axis=1),
        np.mean(np.abs(np.diff(values["series"][:, int(ctx.read_input("boundary")):], axis=1)), axis=1) - np.mean(np.abs(np.diff(values["series"][:, :int(ctx.read_input("boundary"))], axis=1)), axis=1),
        np.max(np.abs(values["series"][:, int(ctx.read_input("boundary")):]), axis=1) - np.max(np.abs(values["series"][:, :int(ctx.read_input("boundary"))]), axis=1)
    ]),
    ["src_diff_std_delta", "src_abs_diff_mean_delta", "src_abs_max_delta"]
)"""


SOURCE_SIMPLE_TAIL_SHIFT = """lambda ctx, values: FeatureBlock(
    np.column_stack([
        np.mean(values["series"][:, -max(2, values["series"].shape[1] // 8):], axis=1) - np.mean(values["series"][:, :max(2, values["series"].shape[1] // 8)], axis=1),
        np.std(values["series"][:, -max(2, values["series"].shape[1] // 8):], axis=1) - np.std(values["series"][:, :max(2, values["series"].shape[1] // 8)], axis=1),
        np.mean(np.abs(values["series"][:, -max(2, values["series"].shape[1] // 8):]), axis=1) - np.mean(np.abs(values["series"][:, :max(2, values["series"].shape[1] // 8)]), axis=1)
    ]),
    ["src_tail_head_mean_delta", "src_tail_head_std_delta", "src_tail_head_abs_delta"]
)"""


SOURCE_DISTRIBUTION_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post: FeatureBlock(
            np.column_stack([
                np.mean(post, axis=1) - np.mean(pre, axis=1),
                np.abs(np.mean(post, axis=1) - np.mean(pre, axis=1)),
                (np.median(post, axis=1) - np.median(pre, axis=1)) / (np.maximum(np.std(pre, axis=1), 1e-8)),
                np.log((np.std(post, axis=1) + 1e-8) / (np.std(pre, axis=1) + 1e-8)),
                (np.quantile(post, 0.75, axis=1) - np.quantile(post, 0.25, axis=1)) - (np.quantile(pre, 0.75, axis=1) - np.quantile(pre, 0.25, axis=1)),
                np.mean(np.abs(np.sort(post, axis=1) - np.sort(pre, axis=1)), axis=1),
                np.max(np.abs(np.quantile(post, np.linspace(0.1, 0.9, 9), axis=1).T - np.quantile(pre, np.linspace(0.1, 0.9, 9), axis=1).T), axis=1)
            ]),
            [
                "src_mean_delta",
                "src_mean_delta_abs",
                "src_scaled_median_delta",
                "src_std_log_ratio",
                "src_iqr_delta",
                "src_sorted_l1_distance",
                "src_quantile_max_distance"
            ]
        ))(s[:, :b], s[:, b:])
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""


SOURCE_AUTOCORR_VOLATILITY_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post, pre_d, post_d: (
            (lambda pre_c, post_c: FeatureBlock(
                np.column_stack([
                    np.mean(np.abs(post_d), axis=1) - np.mean(np.abs(pre_d), axis=1),
                    np.log((np.std(post_d, axis=1) + 1e-8) / (np.std(pre_d, axis=1) + 1e-8)),
                    (np.sum(post_c[:, 1:] * post_c[:, :-1], axis=1) / np.maximum(np.sum(post_c[:, :-1] ** 2, axis=1), 1e-8))
                        - (np.sum(pre_c[:, 1:] * pre_c[:, :-1], axis=1) / np.maximum(np.sum(pre_c[:, :-1] ** 2, axis=1), 1e-8)),
                    np.max(np.abs(np.cumsum(post - np.mean(post, axis=1, keepdims=True), axis=1)), axis=1) / np.maximum(np.std(post, axis=1), 1e-8),
                    np.max(np.abs(np.cumsum(pre - np.mean(pre, axis=1, keepdims=True), axis=1)), axis=1) / np.maximum(np.std(pre, axis=1), 1e-8),
                    np.max(np.abs(np.cumsum(post - np.mean(post, axis=1, keepdims=True), axis=1)), axis=1) / np.maximum(np.max(np.abs(np.cumsum(pre - np.mean(pre, axis=1, keepdims=True), axis=1)), axis=1), 1e-8)
                ]),
                [
                    "src_absdiff_delta",
                    "src_diff_std_log_ratio",
                    "src_autocorr_delta",
                    "src_post_cusum_peak",
                    "src_pre_cusum_peak",
                    "src_cusum_peak_ratio"
                ]
            ))(pre - np.mean(pre, axis=1, keepdims=True), post - np.mean(post, axis=1, keepdims=True))
        ))(s[:, :b], s[:, b:], np.diff(s[:, :b], axis=1), np.diff(s[:, b:], axis=1))
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""


SOURCE_MULTISCALE_TAIL_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post, w1, w2: FeatureBlock(
            np.column_stack([
                np.mean(post[:, :w1], axis=1) - np.mean(pre[:, -w1:], axis=1),
                np.mean(post[:, -w1:], axis=1) - np.mean(post[:, :w1], axis=1),
                np.mean(post[:, -w2:], axis=1) - np.mean(pre[:, -w2:], axis=1),
                np.std(post[:, -w1:], axis=1) - np.std(pre[:, -w1:], axis=1),
                np.max(post, axis=1) - np.max(pre, axis=1),
                np.min(post, axis=1) - np.min(pre, axis=1),
                (s[:, -1] - np.mean(pre[:, -w2:], axis=1)) / np.maximum(np.std(pre[:, -w2:], axis=1), 1e-8)
            ]),
            [
                "src_boundary_jump",
                "src_post_tail_drift",
                "src_wide_tail_delta",
                "src_tail_std_delta",
                "src_max_delta",
                "src_min_delta",
                "src_last_pre_tail_z"
            ]
        ))(s[:, :b], s[:, b:], max(2, (s.shape[1] - b) // 4), max(3, (s.shape[1] - b) // 2))
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""


SOURCE_RANK_SHAPE_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post: (
            (lambda combined, pre_rank, post_rank: FeatureBlock(
                np.column_stack([
                    np.mean(post_rank, axis=1) - np.mean(pre_rank, axis=1),
                    np.std(post_rank, axis=1) - np.std(pre_rank, axis=1),
                    np.mean(np.maximum(post_rank - 0.75, 0.0), axis=1) - np.mean(np.maximum(pre_rank - 0.75, 0.0), axis=1),
                    np.mean(np.maximum(0.25 - post_rank, 0.0), axis=1) - np.mean(np.maximum(0.25 - pre_rank, 0.0), axis=1),
                    np.mean(np.abs(np.diff(post_rank, axis=1)), axis=1) - np.mean(np.abs(np.diff(pre_rank, axis=1)), axis=1)
                ]),
                [
                    "src_rank_mean_delta",
                    "src_rank_std_delta",
                    "src_upper_rank_excess_delta",
                    "src_lower_rank_excess_delta",
                    "src_rank_path_variation_delta"
                ]
            ))(
                np.concatenate([pre, post], axis=1),
                np.argsort(np.argsort(np.concatenate([pre, post], axis=1), axis=1), axis=1)[:, :pre.shape[1]] / np.maximum(np.concatenate([pre, post], axis=1).shape[1] - 1, 1),
                np.argsort(np.argsort(np.concatenate([pre, post], axis=1), axis=1), axis=1)[:, pre.shape[1]:] / np.maximum(np.concatenate([pre, post], axis=1).shape[1] - 1, 1)
            )
        ))(s[:, :b], s[:, b:])
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""


SOURCE_SPECTRAL_ENTROPY_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post: (
            (lambda pre_fft, post_fft: (
                (lambda pre_p, post_p, freq: FeatureBlock(
                    np.column_stack([
                        np.sum(post_p * freq.reshape(1, -1), axis=1) - np.sum(pre_p * freq.reshape(1, -1), axis=1),
                        -np.sum(post_p * np.log(post_p + 1e-8), axis=1) + np.sum(pre_p * np.log(pre_p + 1e-8), axis=1),
                        np.max(post_p, axis=1) - np.max(pre_p, axis=1),
                        np.sum(post_p[:, :max(2, post_p.shape[1] // 3)], axis=1) - np.sum(pre_p[:, :max(2, pre_p.shape[1] // 3)], axis=1),
                        np.sum(post_p[:, -max(2, post_p.shape[1] // 3):], axis=1) - np.sum(pre_p[:, -max(2, pre_p.shape[1] // 3):], axis=1)
                    ]),
                    [
                        "src_spectral_centroid_delta",
                        "src_spectral_entropy_delta",
                        "src_spectral_peak_share_delta",
                        "src_spectral_low_share_delta",
                        "src_spectral_high_share_delta"
                    ]
                ))(
                    pre_fft / np.maximum(np.sum(pre_fft, axis=1, keepdims=True), 1e-8),
                    post_fft / np.maximum(np.sum(post_fft, axis=1, keepdims=True), 1e-8),
                    np.linspace(0.0, 1.0, pre_fft.shape[1])
                )
            ))(np.abs(np.fft.rfft(pre, axis=1))[:, 1:] + 1e-8, np.abs(np.fft.rfft(post, axis=1))[:, 1:] + 1e-8)
        ))(s[:, :b], s[:, b:])
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""


SOURCE_TAIL_EXTREMA_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post, w: (
            (lambda pre_tail, post_tail, pre_med, post_med, pre_scale, post_scale: FeatureBlock(
                np.column_stack([
                    (np.max(post_tail, axis=1) - post_med) / post_scale - (np.max(pre_tail, axis=1) - pre_med) / pre_scale,
                    (post_med - np.min(post_tail, axis=1)) / post_scale - (pre_med - np.min(pre_tail, axis=1)) / pre_scale,
                    (np.max(post_tail, axis=1) - np.min(post_tail, axis=1)) - (np.max(pre_tail, axis=1) - np.min(pre_tail, axis=1)),
                    np.mean(post_tail > (post_med.reshape(-1, 1) + post_scale.reshape(-1, 1)), axis=1) - np.mean(pre_tail > (pre_med.reshape(-1, 1) + pre_scale.reshape(-1, 1)), axis=1),
                    np.mean(post_tail < (post_med.reshape(-1, 1) - post_scale.reshape(-1, 1)), axis=1) - np.mean(pre_tail < (pre_med.reshape(-1, 1) - pre_scale.reshape(-1, 1)), axis=1)
                ]),
                [
                    "src_tail_upper_extreme_delta",
                    "src_tail_lower_extreme_delta",
                    "src_tail_range_delta",
                    "src_tail_high_exceedance_delta",
                    "src_tail_low_exceedance_delta"
                ]
            ))(
                pre[:, -w:],
                post[:, -w:],
                np.median(pre, axis=1),
                np.median(post, axis=1),
                np.maximum(np.std(pre, axis=1), 1e-8),
                np.maximum(np.std(post, axis=1), 1e-8)
            )
        ))(s[:, :b], s[:, b:], max(3, (s.shape[1] - b) // 3))
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""


SOURCE_MOMENT_SHAPE_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post: (
            (lambda pre_z, post_z: FeatureBlock(
                np.column_stack([
                    np.mean(post_z ** 3, axis=1) - np.mean(pre_z ** 3, axis=1),
                    np.mean(post_z ** 4, axis=1) - np.mean(pre_z ** 4, axis=1),
                    np.mean(np.abs(post_z), axis=1) - np.mean(np.abs(pre_z), axis=1),
                    np.mean(np.maximum(post_z, 0.0) ** 2, axis=1) - np.mean(np.maximum(pre_z, 0.0) ** 2, axis=1),
                    np.mean(np.minimum(post_z, 0.0) ** 2, axis=1) - np.mean(np.minimum(pre_z, 0.0) ** 2, axis=1)
                ]),
                [
                    "src_skew_delta",
                    "src_kurtosis_delta",
                    "src_abs_z_delta",
                    "src_positive_energy_delta",
                    "src_negative_energy_delta"
                ]
            ))(
                (pre - np.mean(pre, axis=1, keepdims=True)) / np.maximum(np.std(pre, axis=1, keepdims=True), 1e-8),
                (post - np.mean(post, axis=1, keepdims=True)) / np.maximum(np.std(post, axis=1, keepdims=True), 1e-8)
            )
        ))(s[:, :b], s[:, b:])
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""


SOURCE_MULTIRESOLUTION_CUSUM_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post: (
            (lambda pre_c, post_c: (
                (lambda pre_abs, post_abs, pre_scale, post_scale: FeatureBlock(
                    np.column_stack([
                        np.max(post_abs, axis=1) / post_scale - np.max(pre_abs, axis=1) / pre_scale,
                        np.argmax(post_abs, axis=1) / np.maximum(post_abs.shape[1] - 1, 1) - np.argmax(pre_abs, axis=1) / np.maximum(pre_abs.shape[1] - 1, 1),
                        np.mean(post_abs[:, :max(2, post_abs.shape[1] // 3)], axis=1) / post_scale - np.mean(pre_abs[:, :max(2, pre_abs.shape[1] // 3)], axis=1) / pre_scale,
                        np.mean(post_abs[:, -max(2, post_abs.shape[1] // 3):], axis=1) / post_scale - np.mean(pre_abs[:, -max(2, pre_abs.shape[1] // 3):], axis=1) / pre_scale,
                        np.max(np.abs(np.cumsum(s - np.mean(s, axis=1, keepdims=True), axis=1))[:, b - 1:], axis=1)
                            / np.maximum(np.max(np.abs(np.cumsum(s - np.mean(s, axis=1, keepdims=True), axis=1)), axis=1), 1e-8)
                    ]),
                    [
                        "src_cusum_peak_delta",
                        "src_cusum_peak_location_delta",
                        "src_cusum_early_energy_delta",
                        "src_cusum_late_energy_delta",
                        "src_global_post_boundary_cusum_share"
                    ]
                ))(
                    np.abs(np.cumsum(pre_c, axis=1)),
                    np.abs(np.cumsum(post_c, axis=1)),
                    np.maximum(np.std(pre, axis=1), 1e-8),
                    np.maximum(np.std(post, axis=1), 1e-8)
                )
            ))(pre - np.mean(pre, axis=1, keepdims=True), post - np.mean(post, axis=1, keepdims=True))
        ))(s[:, :b], s[:, b:])
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""


SOURCE_HAAR_SCALE_ENERGY_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post, scales: FeatureBlock(
            np.column_stack([
                *[
                    np.log(
                        (np.mean(np.abs(post[:, scale:] - post[:, :-scale]), axis=1) + 1e-8)
                        / (np.mean(np.abs(pre[:, scale:] - pre[:, :-scale]), axis=1) + 1e-8)
                    )
                    for scale in scales
                ],
                *[
                    np.mean(post[:, scale:] - post[:, :-scale], axis=1) - np.mean(pre[:, scale:] - pre[:, :-scale], axis=1)
                    for scale in scales
                ]
            ]),
            [
                "src_haar_energy_log_ratio_s2",
                "src_haar_energy_log_ratio_s4",
                "src_haar_energy_log_ratio_s8",
                "src_haar_signed_delta_s2",
                "src_haar_signed_delta_s4",
                "src_haar_signed_delta_s8"
            ]
        ))(
            s[:, :b],
            s[:, b:],
            np.asarray([2, min(4, max(2, s[:, b:].shape[1] - 1)), min(8, max(2, s[:, b:].shape[1] - 1))], dtype=np.int64)
        )
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""


SOURCE_QUANTILE_TRANSPORT_SHIFT = """lambda ctx, values: (
    (lambda s, b: (
        (lambda pre, post, q: (
            (lambda pre_q, post_q, diff: FeatureBlock(
                np.column_stack([
                    np.mean(diff, axis=1),
                    np.mean(np.abs(diff), axis=1),
                    np.max(np.abs(diff), axis=1),
                    np.mean(diff[:, -4:], axis=1) - np.mean(diff[:, :4], axis=1),
                    np.log((np.mean(np.maximum(post_q[:, -4:] - np.median(post, axis=1, keepdims=True), 0.0), axis=1) + 1e-8)
                        / (np.mean(np.maximum(pre_q[:, -4:] - np.median(pre, axis=1, keepdims=True), 0.0), axis=1) + 1e-8)),
                    np.log((np.mean(np.maximum(np.median(post, axis=1, keepdims=True) - post_q[:, :4], 0.0), axis=1) + 1e-8)
                        / (np.mean(np.maximum(np.median(pre, axis=1, keepdims=True) - pre_q[:, :4], 0.0), axis=1) + 1e-8))
                ]),
                [
                    "src_quantile_transport_signed_mean",
                    "src_quantile_transport_l1",
                    "src_quantile_transport_linf",
                    "src_quantile_tail_balance_delta",
                    "src_upper_tail_spread_log_ratio",
                    "src_lower_tail_spread_log_ratio"
                ]
            ))(
                np.quantile(pre, q, axis=1).T,
                np.quantile(post, q, axis=1).T,
                np.quantile(post, q, axis=1).T - np.quantile(pre, q, axis=1).T
            )
        ))(s[:, :b], s[:, b:], np.linspace(0.05, 0.95, 19))
    ))(np.asarray(values["series"], dtype=np.float64), int(ctx.read_input("boundary")))
)"""
