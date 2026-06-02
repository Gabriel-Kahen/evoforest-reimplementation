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
    """Trusted source-backed structural-break feature templates.

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
            source=SOURCE_DISTRIBUTION_SHIFT,
            description="Source-backed robust distribution-shift features across the known boundary.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="source",
            alternative_id="source_autocorr_volatility_shift",
            parents=("series",),
            source=SOURCE_AUTOCORR_VOLATILITY_SHIFT,
            description="Source-backed autocorrelation, volatility, and increment-regime change features.",
        ),
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="source",
            alternative_id="source_multiscale_tail_shift",
            parents=("series",),
            source=SOURCE_MULTISCALE_TAIL_SHIFT,
            description="Source-backed multiscale tail/head structural-break features.",
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
