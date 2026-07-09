from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import time
from typing import Any, Callable

import numpy as np

from evoforest_arch.evaluation_cache import PersistentEvaluationCache, fingerprint_inputs
from evoforest_arch.graph import EvalContext, Graph, ResidualWeightRule
from evoforest_arch.metrics import DEFAULT_SCORER, FoldStrategy, ScoreFunction, TaskScorer, coerce_fold_strategy, coerce_scorer, safe_corr
from evoforest_arch.readout import DEFAULT_ALPHAS, RidgeModel, Standardizer, combine_sample_weights, normalize_sample_weight, select_alpha_and_fit_ridge
from evoforest_arch.task import TaskSchema


@dataclass
class EvaluationResult:
    score: float
    config: dict[str, str]
    feature_names: list[str]
    predictions: np.ndarray
    alphas: list[float]
    diagnostics: dict[str, object]
    feature_matrix: np.ndarray | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "config": self.config,
            "feature_names": self.feature_names,
            "alphas": self.alphas,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class LinearDiagnosticFit:
    alpha: float
    intercept: float
    coef: np.ndarray
    predictions: np.ndarray
    contributions: np.ndarray
    residual_reweighted: bool
    sample_weight: np.ndarray
    irls_iterations: list[dict[str, object]]

    def to_dict(self, y: np.ndarray, scorer: TaskScorer = DEFAULT_SCORER) -> dict[str, object]:
        reconstructed = self.intercept + np.sum(self.contributions, axis=1)
        score, raw_score = _score_and_raw(scorer, y, self.predictions)
        return {
            "alpha": float(self.alpha),
            "score": float(score),
            "raw_score": float(raw_score),
            "raw_metric": scorer.raw_name or scorer.name,
            "intercept": float(self.intercept),
            "prediction_std": float(np.std(self.predictions)),
            "residual_std": float(np.std(y - self.predictions)),
            "residual_reweighted": bool(self.residual_reweighted),
            "sample_weight_min": float(np.min(self.sample_weight)),
            "sample_weight_max": float(np.max(self.sample_weight)),
            "sample_weight_mean": float(np.mean(self.sample_weight)),
            "sample_weight_std": float(np.std(self.sample_weight)),
            "irls_steps_used": len(self.irls_iterations),
            "irls_iterations": self.irls_iterations,
            "contribution_reconstruction_error": float(np.max(np.abs(self.predictions - reconstructed))) if self.predictions.size else 0.0,
            "mean_abs_contribution": float(np.mean(np.abs(self.contributions))) if self.contributions.size else 0.0,
        }


@dataclass(frozen=True)
class FeatureCorrelationSummary:
    max_corr: np.ndarray
    high_corr_counts: np.ndarray
    most_correlated: list[str]
    effective_rank: float


@dataclass(frozen=True)
class ReweightedRidgeFit:
    model: RidgeModel
    alpha: float
    sample_weight: np.ndarray | None
    irls_iterations: list[dict[str, object]]


def _score_and_raw(scorer: TaskScorer, y: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    raw_score = float(scorer.raw_score(y, preds))
    if type(scorer) is TaskScorer:
        return float(raw_score if scorer.higher_is_better else -raw_score), raw_score
    return float(scorer.score(y, preds)), raw_score


class RidgeEvaluator:
    def __init__(
        self,
        n_splits: int = 3,
        seed: int = 0,
        alphas: np.ndarray = DEFAULT_ALPHAS,
        max_configurations: int = 64,
        refine_globals: bool = True,
        refine_steps: int = 20,
        refine_backend: str = "auto",
        irls_steps: int = 2,
        group_key: str | None = None,
        fold_strategy: FoldStrategy | str | None = None,
        time_key: str | None = None,
        stratify_bins: int = 5,
        diagnostics_mode: str = "full",
        feature_pool_diagnostics: bool = True,
        retain_feature_matrix: bool = True,
        torch_device: str | None = None,
        scorer: TaskScorer | ScoreFunction | str | None = None,
        task_schema: TaskSchema | dict[str, object] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        persistent_cache: bool = True,
        cache_max_entries: int = 4096,
        cache_max_bytes: int = 1_073_741_824,
    ) -> None:
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.max_configurations = max(1, int(max_configurations))
        self.refine_globals = bool(refine_globals)
        self.refine_steps = int(refine_steps)
        self.refine_backend = refine_backend
        self.torch_device = torch_device
        self.irls_steps = max(0, int(irls_steps))
        self.task_schema = _coerce_task_schema(task_schema)
        inferred_group_key = group_key or _schema_role_key(self.task_schema, "group", "unit", "engine", "entity")
        inferred_time_key = time_key or _schema_role_key(self.task_schema, "time", "cycle", "sequence_index")
        inferred_fold_strategy = fold_strategy
        if inferred_fold_strategy is None:
            if inferred_group_key:
                inferred_fold_strategy = "group_random"
            elif inferred_time_key:
                inferred_fold_strategy = "time_blocked"
        self.group_key = inferred_group_key
        self.time_key = inferred_time_key
        self.fold_strategy = coerce_fold_strategy(inferred_fold_strategy, group_key=inferred_group_key, time_key=inferred_time_key, stratify_bins=stratify_bins)
        self.scorer = coerce_scorer(scorer)
        if diagnostics_mode not in {"full", "basic"}:
            raise ValueError("diagnostics_mode must be 'full' or 'basic'.")
        self.diagnostics_mode = diagnostics_mode
        self.feature_pool_diagnostics = bool(feature_pool_diagnostics)
        self.retain_feature_matrix = bool(retain_feature_matrix)
        self.progress_callback = progress_callback
        self.persistent_cache = bool(persistent_cache)
        self.evaluation_cache = PersistentEvaluationCache(max_entries=cache_max_entries, max_bytes=cache_max_bytes)

    def clear_evaluation_cache(self) -> None:
        self.evaluation_cache.clear()

    def evaluation_cache_diagnostics(self) -> dict[str, int | bool]:
        return {"enabled": self.persistent_cache, **self.evaluation_cache.stats()}

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(dict(payload))

    def evaluate(
        self,
        graph: Graph,
        inputs: dict[str, object],
        y: np.ndarray,
        config: dict[str, str] | None = None,
        update_graph: bool = False,
    ) -> EvaluationResult:
        working_graph = graph if update_graph else graph.clone()
        y = np.asarray(y, dtype=np.float64)
        if self.persistent_cache:
            dataset_fingerprint = fingerprint_inputs(inputs)
            self.evaluation_cache.begin_evaluation()
            shared_cache: dict[object, object] = self.evaluation_cache
        else:
            dataset_fingerprint = ""
            shared_cache = {}
        persistent_stats_before = self.evaluation_cache.stats()
        started_at = time.monotonic()
        search_cache_hits = 0
        search_cache_misses = 0
        feature_matrix_reuse_hits = 0
        folds, fold_diagnostics = self._folds(inputs, y)
        refinement_diagnostics: dict[str, object] = {"enabled": False}
        if self.refine_globals and working_graph.globals.trainable_names():
            from evoforest_arch.refinement import GlobalRefiner

            base_config = working_graph.default_config()
            if config:
                base_config.update(config)
            refinement = GlobalRefiner(
                steps=self.refine_steps,
                seed=self.seed,
                backend=self.refine_backend,
                device=self.torch_device,
            ).refine(working_graph, inputs, y, base_config)
            refinement_diagnostics = refinement.to_dict()

        if config is not None:
            selected = working_graph.default_config()
            selected.update(config)
            result = self._evaluate_single_config(
                working_graph,
                inputs,
                y,
                selected,
                shared_cache,
                cache_namespace=dataset_fingerprint,
                folds=folds,
                fold_diagnostics=fold_diagnostics,
                progress_context={
                    "config_index": 1,
                    "configurations_evaluated": 1,
                    "configurations_limit": 1,
                    "configurations_total": 1,
                },
            )
            feature_pool = [(result.feature_matrix, result.feature_names)] if self.feature_pool_diagnostics else []
            cache_row = result.diagnostics.get("cache", {})
            if isinstance(cache_row, dict):
                search_cache_hits += int(cache_row.get("hits", 0))
                search_cache_misses += int(cache_row.get("misses", 0))
            result.diagnostics["configuration_search"] = self._configuration_search_diagnostics(
                [{"score": result.score, "config": result.config, "n_features": len(result.feature_names)}],
                total_configurations=1,
            )
        else:
            configs, total_configurations = self._configuration_candidates(working_graph)
            best: EvaluationResult | None = None
            best_feature_result: tuple[np.ndarray, list[str]] | None = None
            config_rows: list[dict[str, object]] = []
            feature_pool_by_signature: dict[tuple[object, ...], tuple[np.ndarray | None, list[str]]] = {}
            feature_results_by_signature: dict[tuple[object, ...], tuple[np.ndarray, list[str]]] | None = (
                {} if self.feature_pool_diagnostics else None
            )
            seen_feature_signatures: set[tuple[object, ...]] = set()
            last_feature_signature: tuple[object, ...] | None = None
            last_feature_result: tuple[np.ndarray, list[str]] | None = None
            for candidate_config in configs:
                config_index = len(config_rows) + 1
                feature_signature = self._feature_signature(working_graph, candidate_config)
                seen_feature_signatures.add(feature_signature)
                precomputed_features = (
                    feature_results_by_signature.get(feature_signature)
                    if feature_results_by_signature is not None
                    else last_feature_result if feature_signature == last_feature_signature else None
                )
                if precomputed_features is not None:
                    feature_matrix_reuse_hits += 1
                result = self._evaluate_single_config(
                    working_graph,
                    inputs,
                    y,
                    candidate_config,
                    shared_cache,
                    cache_namespace=dataset_fingerprint,
                    precomputed_features=precomputed_features,
                    folds=folds,
                    fold_diagnostics=fold_diagnostics,
                    diagnostics_mode="score",
                    progress_context={
                        "config_index": config_index,
                        "configurations_evaluated": config_index,
                        "configurations_limit": len(configs),
                        "configurations_total": total_configurations,
                    },
                )
                if result.feature_matrix is not None:
                    last_feature_signature = feature_signature
                    last_feature_result = (result.feature_matrix, result.feature_names)
                    if feature_results_by_signature is not None:
                        feature_results_by_signature[feature_signature] = last_feature_result
                if self.feature_pool_diagnostics:
                    feature_pool_by_signature.setdefault(feature_signature, (result.feature_matrix, result.feature_names))
                cache_row = result.diagnostics.get("cache", {})
                if isinstance(cache_row, dict):
                    search_cache_hits += int(cache_row.get("hits", 0))
                    search_cache_misses += int(cache_row.get("misses", 0))
                config_rows.append(
                    {
                        "score": result.score,
                        "config": result.config,
                        "n_features": len(result.feature_names),
                        "cache_hits": int(cache_row.get("hits", 0)) if isinstance(cache_row, dict) else 0,
                        "cache_misses": int(cache_row.get("misses", 0)) if isinstance(cache_row, dict) else 0,
                    }
                )
                self._emit_progress(
                    {
                        "phase": "configuration_evaluated",
                        "config_index": config_index,
                        "configurations_evaluated": config_index,
                        "configurations_limit": len(configs),
                        "configurations_total": total_configurations,
                        "score": float(result.score),
                        "n_features": len(result.feature_names),
                        "elapsed_seconds": time.monotonic() - started_at,
                    }
                )
                if best is None or result.score > best.score:
                    best = result
                    best_feature_result = last_feature_result
            if best is None:
                raise ValueError("No graph configurations were available for evaluation.")
            if self.diagnostics_mode == "full":
                result = self._evaluate_single_config(
                    working_graph,
                    inputs,
                    y,
                    best.config,
                    shared_cache,
                    cache_namespace=dataset_fingerprint,
                    precomputed_features=best_feature_result,
                    folds=folds,
                    fold_diagnostics=fold_diagnostics,
                    diagnostics_mode="full",
                    progress_context={
                        "phase_detail": "winner_diagnostics",
                        "config_index": len(config_rows),
                        "configurations_evaluated": len(config_rows),
                        "configurations_limit": len(configs),
                        "configurations_total": total_configurations,
                    },
                )
                result.diagnostics["evaluation_passes"] = {
                    "configuration_scoring": "score",
                    "winner_diagnostics": "full",
                    "winner_rerun": True,
                }
            else:
                result = best
                scoring_diagnostics = result.diagnostics
                result.diagnostics = self._basic_diagnostics(
                    result.predictions,
                    y,
                    inputs,
                    result.feature_names,
                    working_graph.selected_alternatives(result.config),
                    result.score,
                )
                for key in ("folds", "fitting", "cache", "graph"):
                    result.diagnostics[key] = scoring_diagnostics[key]
                result.diagnostics["evaluation_passes"] = {
                    "configuration_scoring": "score",
                    "winner_diagnostics": "basic",
                    "winner_rerun": False,
                }
            feature_pool = list(feature_pool_by_signature.values())
            result.diagnostics["configuration_search"] = self._configuration_search_diagnostics(config_rows, total_configurations)
            result.diagnostics["configuration_search"]["unique_feature_matrices"] = len(seen_feature_signatures)
        search_cache = {
            "hits": int(search_cache_hits),
            "misses": int(search_cache_misses),
            "entries": int(len(shared_cache)),
            "feature_matrix_reuse_hits": int(feature_matrix_reuse_hits),
            "shared_across_configurations": True,
            "key": "ancestor_conditioned_subpath",
            "key_version": "content_addressed_v2",
            "dataset_fingerprint": dataset_fingerprint,
        }
        persistent_stats_after = self.evaluation_cache.stats()
        search_cache["persistent"] = {
            "enabled": self.persistent_cache,
            "cross_evaluation_hits": int(persistent_stats_after["cross_evaluation_hits"] - persistent_stats_before["cross_evaluation_hits"]),
            "stores": int(persistent_stats_after["stores"] - persistent_stats_before["stores"]),
            "evictions": int(persistent_stats_after["evictions"] - persistent_stats_before["evictions"]),
            "entries": int(persistent_stats_after["entries"]),
            "bytes": int(persistent_stats_after["bytes"]),
            "max_entries": int(persistent_stats_after["max_entries"]),
            "max_bytes": int(persistent_stats_after["max_bytes"]),
        }
        result.diagnostics["configuration_search"]["cache"] = search_cache
        result.diagnostics["cache"] = {
            **(result.diagnostics.get("cache", {}) if isinstance(result.diagnostics.get("cache", {}), dict) else {}),
            "search_hits": search_cache["hits"],
            "search_misses": search_cache["misses"],
            "search_entries": search_cache["entries"],
            "shared_across_configurations": True,
            "key": "ancestor_conditioned_subpath",
            "key_version": search_cache["key_version"],
            "persistent": search_cache["persistent"],
        }
        if self.feature_pool_diagnostics:
            self._attach_valid_feature_pool_diagnostics(result, y, feature_pool)
        else:
            result.diagnostics["valid_feature_pool"] = {
                "enabled": False,
                "reason": "feature_pool_diagnostics=false",
            }
        result.diagnostics["scoring_context"] = self._scoring_context(result)
        result.diagnostics["refinement"] = refinement_diagnostics
        if update_graph:
            alternatives = result.diagnostics.get("alternatives", [])
            if isinstance(alternatives, list):
                working_graph.update_alternative_statistics([row for row in alternatives if isinstance(row, dict)])
        result.diagnostics["alternative_stats"] = working_graph.alternative_statistics_snapshot()
        if not self.retain_feature_matrix:
            result.feature_matrix = None
        return result

    def _evaluate_single_config(
        self,
        graph: Graph,
        inputs: dict[str, object],
        y: np.ndarray,
        config: dict[str, str],
        shared_cache: dict[object, object] | None = None,
        cache_namespace: str = "",
        precomputed_features: tuple[np.ndarray, list[str]] | None = None,
        folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
        fold_diagnostics: dict[str, object] | None = None,
        progress_context: dict[str, Any] | None = None,
        diagnostics_mode: str | None = None,
    ) -> EvaluationResult:
        progress_context = dict(progress_context or {})
        resolved_diagnostics_mode = diagnostics_mode or self.diagnostics_mode
        if precomputed_features is None:
            x, names, ctx = graph.evaluate_features(inputs, config=config, cache=shared_cache, cache_namespace=cache_namespace)
            feature_matrix_reused = False
        else:
            x, names = precomputed_features
            ctx = EvalContext(
                inputs=inputs,
                globals=graph.globals.clone(),
                cache=shared_cache if shared_cache is not None else {},
                cache_namespace=cache_namespace,
            )
            feature_matrix_reused = True
        self._emit_progress(
            {
                "phase": "features_evaluated",
                **progress_context,
                "n_rows": int(x.shape[0]),
                "n_features": int(x.shape[1]),
                "feature_matrix_reused": bool(feature_matrix_reused),
            }
        )
        sample_weight, residual_rule, fitting_diagnostics = self._evaluate_fitting_rules(graph, inputs, y, config, ctx)
        preds = np.zeros_like(y, dtype=np.float64)
        collect_full_diagnostics = resolved_diagnostics_mode == "full"
        fold_abs_contribution_sum = np.zeros(x.shape[1], dtype=np.float64) if collect_full_diagnostics else None
        cv_reconstructed = np.zeros_like(y, dtype=np.float64) if collect_full_diagnostics else None
        alphas: list[float] = []
        coefs: list[np.ndarray] = []
        fold_scores: list[float] = []
        fold_raw_scores: list[float] = []
        fold_irls: list[dict[str, object]] = []
        if folds is None or fold_diagnostics is None:
            folds, fold_diagnostics = self._folds(inputs, y)
        for fold_index, (train_idx, val_idx) in enumerate(folds):
            x_train_raw = x[train_idx]
            x_val_raw = x[val_idx]
            std = Standardizer.fit(x_train_raw)
            x_train = std.transform(x_train_raw)
            x_val = std.transform(x_val_raw)
            train_weight = sample_weight[train_idx] if sample_weight is not None else None
            ridge_fit = self._fit_reweighted_ridge(x_train, y[train_idx], train_weight, residual_rule)
            model = ridge_fit.model
            preds[val_idx] = model.predict(x_val)
            if collect_full_diagnostics:
                assert fold_abs_contribution_sum is not None
                assert cv_reconstructed is not None
                contribution = x_val * model.coef.reshape(1, -1)
                fold_abs_contribution_sum += np.sum(np.abs(contribution), axis=0)
                cv_reconstructed[val_idx] = float(model.intercept) + np.sum(contribution, axis=1)
            fold_score, fold_raw_score = _score_and_raw(self.scorer, y[val_idx], preds[val_idx])
            fold_scores.append(float(fold_score))
            fold_raw_scores.append(float(fold_raw_score))
            self._emit_progress(
                {
                    "phase": "fold_evaluated",
                    **progress_context,
                    "fold": int(fold_index + 1),
                    "folds": int(len(folds)),
                    "fold_score": float(fold_score),
                    "fold_raw_score": float(fold_raw_score),
                    "n_features": int(x.shape[1]),
                }
            )
            alphas.append(float(ridge_fit.alpha))
            if collect_full_diagnostics:
                coefs.append(model.coef)
            fold_irls.append(
                {
                    "fold": int(fold_index),
                    "steps_used": len(ridge_fit.irls_iterations),
                    "final_alpha": float(ridge_fit.alpha),
                    "iterations": ridge_fit.irls_iterations,
                }
            )
        score = float(np.mean(fold_scores)) if fold_scores else _score_and_raw(self.scorer, y, preds)[0]
        selected_alternatives = graph.selected_alternatives(config)
        if resolved_diagnostics_mode == "full":
            assert fold_abs_contribution_sum is not None
            assert cv_reconstructed is not None
            feature_dependencies = feature_dependency_rows(names, graph.output_dependency_map(config))
            global_fit = self._fit_global_diagnostic_model(x, y, sample_weight, residual_rule)
            diagnostics = self._diagnostics(
                x,
                y,
                preds,
                inputs,
                names,
                coefs,
                feature_dependencies,
                fold_abs_contribution_sum / max(int(y.shape[0]), 1),
                cv_reconstructed,
                global_fit,
            )
            diagnostics["alternatives"] = alternative_diagnostics(diagnostics["features"], selected_alternatives, score)
        elif resolved_diagnostics_mode == "basic":
            diagnostics = self._basic_diagnostics(preds, y, inputs, names, selected_alternatives, score)
        else:
            diagnostics = {
                "diagnostics_mode": "score",
                "n_features": int(len(names)),
                "features": [],
                "subnodes": [],
                "alternatives": [],
            }
        diagnostics["folds"] = {
            "score": fold_scores,
            "score_mean": float(np.mean(fold_scores)) if fold_scores else 0.0,
            "score_std": float(np.std(fold_scores)) if fold_scores else 0.0,
            "raw_score": fold_raw_scores,
            "raw_score_mean": float(np.mean(fold_raw_scores)) if fold_raw_scores else 0.0,
            "raw_score_std": float(np.std(fold_raw_scores)) if fold_raw_scores else 0.0,
            "raw_metric": self.scorer.raw_name or self.scorer.name,
            "raw_higher_is_better": bool(self.scorer.higher_is_better),
            "alphas": alphas,
            **fold_diagnostics,
        }
        diagnostics["fitting"] = fitting_diagnostics
        if isinstance(diagnostics["fitting"].get("ridge_g"), dict):
            diagnostics["fitting"]["ridge_g"]["irls_steps_requested"] = int(self.irls_steps)
            diagnostics["fitting"]["ridge_g"]["irls_steps_used_per_fold"] = [int(row["steps_used"]) for row in fold_irls]
            diagnostics["fitting"]["ridge_g"]["irls"] = fold_irls
        diagnostics["cache"] = {
            "hits": int(ctx.cache_hits),
            "misses": int(ctx.cache_misses),
            "entries": int(len(ctx.cache)),
            "feature_matrix_reused": bool(feature_matrix_reused),
        }
        diagnostics["graph"] = {
            "nodes": len(graph.nodes),
            "outputs": graph.output_nodes(),
            "alternatives": sum(len(node.alternatives) for node in graph.nodes.values()),
            "selected_alternatives": selected_alternatives,
        }
        return EvaluationResult(
            score=float(score),
            config=dict(config),
            feature_names=names,
            predictions=preds,
            alphas=alphas,
            diagnostics=diagnostics,
            feature_matrix=x,
        )

    def _feature_signature(self, graph: Graph, config: dict[str, str]) -> tuple[object, ...]:
        selected = graph.selected_config(config)
        memo: dict[tuple[str, str], tuple[object, ...]] = {}
        return tuple(
            graph.alternative_cache_key(node_name, alternative.id, selected, memo=memo)
            for node_name in graph.output_nodes()
            for alternative in graph.nodes[node_name].alternatives
        )

    def _attach_valid_feature_pool_diagnostics(
        self,
        result: EvaluationResult,
        y: np.ndarray,
        feature_pool: list[tuple[np.ndarray | None, list[str]]],
    ) -> None:
        matrices = [matrix for matrix, _names in feature_pool if matrix is not None and matrix.size]
        if not matrices:
            return
        n_configurations = len(matrices)
        n_feature_names = sum(len(names_) for _matrix, names_ in feature_pool)
        x_pool = matrices[0] if len(matrices) == 1 else np.column_stack(matrices)
        feature_pool.clear()
        matrices.clear()
        pool_fit = self._fit_global_diagnostic_model(x_pool, y, None, None)
        result.diagnostics["valid_feature_pool"] = {
            "n_configurations": int(n_configurations),
            "n_features": int(x_pool.shape[1]),
            "n_feature_names": int(n_feature_names),
        }
        result.diagnostics["global_ridge_pool"] = pool_fit.to_dict(y, self.scorer)

    def _folds(self, inputs: dict[str, object], y: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
        return self.fold_strategy.split(inputs, y, self.n_splits, self.seed)

    def _fit_global_diagnostic_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        residual_rule: ResidualWeightRule | None,
    ) -> LinearDiagnosticFit:
        std = Standardizer.fit(x)
        xs = std.transform(x)
        ridge_fit = self._fit_reweighted_ridge(xs, y, sample_weight, residual_rule)
        model = ridge_fit.model
        predictions = model.predict(xs)
        contributions = xs * model.coef.reshape(1, -1)
        return LinearDiagnosticFit(
            alpha=float(ridge_fit.alpha),
            intercept=float(model.intercept),
            coef=model.coef,
            predictions=predictions,
            contributions=contributions,
            residual_reweighted=bool(ridge_fit.irls_iterations),
            sample_weight=normalize_sample_weight(ridge_fit.sample_weight, y.shape[0]),
            irls_iterations=ridge_fit.irls_iterations,
        )

    def _fit_reweighted_ridge(
        self,
        x: np.ndarray,
        y: np.ndarray,
        base_weight: np.ndarray | None,
        residual_rule: ResidualWeightRule | None,
    ) -> ReweightedRidgeFit:
        current_weight = base_weight
        alpha, model = select_alpha_and_fit_ridge(x, y, self.alphas, sample_weight=current_weight)
        irls_iterations: list[dict[str, object]] = []
        if residual_rule is None or residual_rule.name == "identity" or self.irls_steps == 0:
            return ReweightedRidgeFit(model=model, alpha=float(alpha), sample_weight=current_weight, irls_iterations=irls_iterations)

        for step in range(1, self.irls_steps + 1):
            residual = y - model.predict(x)
            residual_weight = normalize_sample_weight(residual_rule.apply(residual), y.shape[0])
            current_weight = combine_sample_weights(base_weight, residual_weight)
            alpha, model = select_alpha_and_fit_ridge(x, y, self.alphas, sample_weight=current_weight)
            irls_iterations.append(
                {
                    "step": int(step),
                    "alpha": float(alpha),
                    "residual_std": float(np.std(residual)),
                    "weight_min": float(np.min(residual_weight)),
                    "weight_max": float(np.max(residual_weight)),
                    "weight_mean": float(np.mean(residual_weight)),
                    "weight_std": float(np.std(residual_weight)),
                }
            )
        return ReweightedRidgeFit(model=model, alpha=float(alpha), sample_weight=current_weight, irls_iterations=irls_iterations)

    def _configuration_candidates(self, graph: Graph) -> tuple[list[dict[str, str]], int]:
        space = graph.configuration_space()
        keys = sorted(space)
        if not keys:
            return [{}], 1
        total = int(np.prod([len(space[key]) for key in keys]))
        if total <= self.max_configurations:
            return [dict(zip(keys, values, strict=True)) for values in product(*(space[key] for key in keys))], total

        rng = np.random.default_rng(self.seed)
        default = graph.default_config()
        configs = [default]
        seen = {tuple(default[key] for key in keys)}
        while len(configs) < self.max_configurations:
            candidate = {key: str(rng.choice(space[key])) for key in keys}
            signature = tuple(candidate[key] for key in keys)
            if signature in seen:
                continue
            seen.add(signature)
            configs.append(candidate)
        return configs, total

    @staticmethod
    def _configuration_search_diagnostics(config_rows: list[dict[str, object]], total_configurations: int) -> dict[str, object]:
        scores = np.asarray([float(row["score"]) for row in config_rows], dtype=np.float64)
        n_features = [int(row.get("n_features", 0)) for row in config_rows]
        top_rows = sorted(config_rows, key=lambda row: float(row["score"]), reverse=True)[:8]
        if scores.size == 0:
            return {
                "evaluated": 0,
                "total": int(total_configurations),
                "capped": False,
                "score_range": [0.0, 0.0],
                "score_mean": 0.0,
                "score_std": 0.0,
                "best_config_score": 0.0,
                "n_features_global": 0,
                "n_features_best_config": 0,
                "top_configs": [],
            }
        return {
            "evaluated": len(config_rows),
            "total": int(total_configurations),
            "capped": int(total_configurations) > len(config_rows),
            "score_range": [float(np.min(scores)), float(np.max(scores))],
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "best_config_score": float(np.max(scores)),
            "n_features_global": int(np.sum(n_features)),
            "n_features_best_config": int(top_rows[0].get("n_features", 0)) if top_rows else 0,
            "top_configs": [
                {
                    "score": float(row["score"]),
                    "n_features": int(row.get("n_features", 0)),
                    "config": row["config"],
                }
                for row in top_rows
            ],
        }

    def _scoring_context(self, result: EvaluationResult) -> dict[str, object]:
        search = result.diagnostics.get("configuration_search", {})
        folds = result.diagnostics.get("folds", {})
        global_ridge = result.diagnostics.get("global_ridge", {})
        global_ridge_pool = result.diagnostics.get("global_ridge_pool", {})
        linear_shap = result.diagnostics.get("linear_shap", {})
        return {
            "scoring": f"configuration-based ({self.scorer.name}; best config score = evoforest score)",
            "scorer": self.scorer.to_dict(),
            "best_config_score": float(result.score),
            "best_config_raw_score": float(folds.get("raw_score_mean", 0.0)) if isinstance(folds, dict) else 0.0,
            "global_ridge_score": float(global_ridge.get("score", 0.0)) if isinstance(global_ridge, dict) else 0.0,
            "global_ridge_raw_score": float(global_ridge.get("raw_score", 0.0)) if isinstance(global_ridge, dict) else 0.0,
            "global_feature_pool_ridge_score": float(global_ridge_pool.get("score", 0.0)) if isinstance(global_ridge_pool, dict) else 0.0,
            "config_score_range": search.get("score_range", [float(result.score), float(result.score)]),
            "fold_score_std": float(folds.get("score_std", 0.0)) if isinstance(folds, dict) else 0.0,
            "effective_rank": float(result.diagnostics.get("effective_rank", 0.0)),
            "mean_max_corr": float(result.diagnostics.get("mean_max_corr", 0.0)),
            "shap_reconstruction_error": float(linear_shap.get("global_reconstruction_error", 0.0)) if isinstance(linear_shap, dict) else 0.0,
            "n_features_global": int(search.get("n_features_global", len(result.feature_names))) if isinstance(search, dict) else len(result.feature_names),
            "n_features_best_config": len(result.feature_names),
            "n_configs": int(search.get("evaluated", 1)) if isinstance(search, dict) else 1,
            "n_configs_total": int(search.get("total", 1)) if isinstance(search, dict) else 1,
        }

    @staticmethod
    def _evaluate_fitting_rules(
        graph: Graph,
        inputs: dict[str, object],
        y: np.ndarray,
        config: dict[str, str],
        ctx: EvalContext,
    ) -> tuple[np.ndarray | None, ResidualWeightRule | None, dict[str, object]]:
        del inputs
        sample_weight = None
        residual_rule = None
        diagnostics: dict[str, object] = {
            "ridge_w": None,
            "ridge_g": None,
        }
        if "ridge_w" in graph.nodes:
            raw_weight = graph.evaluate_node("ridge_w", config, ctx)
            sample_weight = normalize_sample_weight(np.asarray(raw_weight, dtype=np.float64), y.shape[0])
            diagnostics["ridge_w"] = {
                "alternative": config.get("ridge_w", graph.nodes["ridge_w"].default_alternative_id()),
                "min": float(np.min(sample_weight)),
                "max": float(np.max(sample_weight)),
                "mean": float(np.mean(sample_weight)),
                "std": float(np.std(sample_weight)),
            }
        if "ridge_g" in graph.nodes:
            raw_rule = graph.evaluate_node("ridge_g", config, ctx)
            if isinstance(raw_rule, ResidualWeightRule):
                residual_rule = raw_rule
            elif callable(raw_rule):
                residual_rule = ResidualWeightRule("callable", raw_rule, "Callable residual weighting rule.")
            else:
                raise TypeError("ridge_g alternatives must return a ResidualWeightRule or callable.")
            diagnostics["ridge_g"] = {
                "alternative": config.get("ridge_g", graph.nodes["ridge_g"].default_alternative_id()),
                "rule": residual_rule.name,
                "description": residual_rule.description,
            }
        return sample_weight, residual_rule, diagnostics

    def _basic_diagnostics(
        self,
        preds: np.ndarray,
        y: np.ndarray,
        inputs: dict[str, object],
        names: list[str],
        selected_alternatives: dict[str, str],
        score: float,
    ) -> dict[str, object]:
        residual = y - preds
        return {
            "prediction_std": float(np.std(preds)),
            "residual_std": float(np.std(residual)),
            "effective_rank": 0.0,
            "mean_max_corr": 0.0,
            "global_ridge": {},
            "linear_shap": {
                "basis": "skipped",
                "global_reconstruction_error": 0.0,
                "cv_reconstruction_error": 0.0,
                "mean_abs_contribution_total": 0.0,
                "top_feature": "",
            },
            "features": [],
            "subnodes": [],
            "alternatives": [
                {
                    "name": f"{node}.{alternative}",
                    "selected": True,
                    "last_score": float(score),
                }
                for node, alternative in selected_alternatives.items()
            ],
            "diagnostics_mode": "basic",
            "n_features": int(len(names)),
            "objective": self._objective_diagnostics(y, preds, inputs),
        }

    def _diagnostics(
        self,
        x: np.ndarray,
        y: np.ndarray,
        preds: np.ndarray,
        inputs: dict[str, object],
        names: list[str],
        coefs: list[np.ndarray],
        feature_dependencies: dict[str, list[str]],
        fold_abs_contribution: np.ndarray,
        cv_reconstructed: np.ndarray,
        global_fit: LinearDiagnosticFit,
    ) -> dict[str, object]:
        coef_matrix = np.vstack(coefs) if coefs else np.zeros((1, x.shape[1]))
        mean_coef = np.mean(coef_matrix, axis=0)
        coef_std = np.std(coef_matrix, axis=0)
        importance = np.mean(np.abs(coef_matrix), axis=0)
        total = max(float(np.sum(importance)), 1e-8)
        global_abs_contribution = np.mean(np.abs(global_fit.contributions), axis=0) if global_fit.contributions.size else np.zeros(x.shape[1], dtype=np.float64)
        shap_total = max(float(np.sum(global_abs_contribution)), 1e-8)
        feature_rows = []
        residual = y - preds
        corr_summary = feature_correlation_summary(x, names)
        redundancy = corr_summary.max_corr
        high_corr_counts = corr_summary.high_corr_counts
        most_correlated = corr_summary.most_correlated
        target_corr = safe_corr_columns(x, y)
        target_quadratic_corr = safe_corr_columns(x, y, square=True)
        residual_corr = safe_corr_columns(x, residual)
        residual_quadratic_corr = safe_corr_columns(x, residual, square=True)
        shap_target_corr = safe_corr_columns(global_fit.contributions, y) if global_fit.contributions.size else np.zeros(x.shape[1], dtype=np.float64)
        for idx, name in enumerate(names):
            contribution = global_fit.contributions[:, idx]
            dependencies = feature_dependencies.get(name, [])
            feature_rows.append(
                {
                    "name": name,
                    "dependencies": dependencies,
                    "depth": len(dependencies),
                    "importance": float(importance[idx] / total),
                    "coef_sign": int(np.sign(mean_coef[idx])),
                    "target_alignment": float(abs(target_corr[idx])),
                    "target_corr": float(target_corr[idx]),
                    "target_quadratic_corr": float(target_quadratic_corr[idx]),
                    "max_corr": float(redundancy[idx]),
                    "n_high_corr": int(high_corr_counts[idx]),
                    "most_correlated": most_correlated[idx],
                    "redundancy": float(redundancy[idx]),
                    "residual_corr": float(residual_corr[idx]),
                    "residual_quadratic_corr": float(residual_quadratic_corr[idx]),
                    "weight_stability": float(abs(mean_coef[idx]) / max(coef_std[idx], 1e-8)),
                    "mean_abs_coef": float(importance[idx]),
                    "global_coef": float(global_fit.coef[idx]),
                    "shap_mean": float(np.mean(contribution)),
                    "shap_mean_abs": float(global_abs_contribution[idx]),
                    "shap_std": float(np.std(contribution)),
                    "shap_importance": float(global_abs_contribution[idx] / shap_total),
                    "shap_target_corr": float(shap_target_corr[idx]),
                    "cv_shap_mean_abs": float(fold_abs_contribution[idx]),
                }
            )
        feature_rows.sort(key=lambda row: row["importance"], reverse=True)
        global_ridge = global_fit.to_dict(y, self.scorer)
        return {
            "prediction_std": float(np.std(preds)),
            "residual_std": float(np.std(residual)),
            "objective": self._objective_diagnostics(y, preds, inputs),
            "effective_rank": float(corr_summary.effective_rank),
            "mean_max_corr": float(np.mean(redundancy)) if redundancy.size else 0.0,
            "global_ridge": global_ridge,
            "linear_shap": {
                "basis": "standardized linear Ridge contribution z_j * coefficient_j",
                "global_reconstruction_error": global_ridge["contribution_reconstruction_error"],
                "cv_reconstruction_error": float(np.max(np.abs(preds - cv_reconstructed))) if preds.size else 0.0,
                "mean_abs_contribution_total": float(np.sum(global_abs_contribution)),
                "top_feature": str(names[int(np.argmax(global_abs_contribution))]) if global_abs_contribution.size else "",
            },
            "features": feature_rows,
            "subnodes": subnode_diagnostics(feature_rows),
        }

    def _objective_diagnostics(self, y: np.ndarray, preds: np.ndarray, inputs: dict[str, object]) -> dict[str, object]:
        residual = np.asarray(y, dtype=np.float64) - np.asarray(preds, dtype=np.float64)
        score, raw_score = _score_and_raw(self.scorer, y, preds)
        rows: dict[str, object] = {
            "score": float(score),
            "raw_score": float(raw_score),
            "raw_metric": self.scorer.raw_name or self.scorer.name,
            "raw_higher_is_better": bool(self.scorer.higher_is_better),
            "residual_mean": float(np.mean(residual)) if residual.size else 0.0,
            "residual_std": float(np.std(residual)) if residual.size else 0.0,
            "residual_abs_mean": float(np.mean(np.abs(residual))) if residual.size else 0.0,
            "target_bins": self._target_bin_diagnostics(y, preds, residual),
        }
        schema_roles = self._schema_role_map()
        if schema_roles:
            rows["schema_roles"] = schema_roles
        group_key = self.fold_strategy.group_key or self.group_key
        if group_key and group_key in inputs:
            group_values = np.asarray(inputs[group_key])
            if group_values.ndim == 1 and group_values.shape[0] == y.shape[0]:
                rows["group_key"] = group_key
                rows["groups"] = self._group_objective_diagnostics(y, preds, residual, group_values)
                rows["split_leakage_check"] = "fold diagnostics report group overlap when grouped folds are active"
        time_key = self.fold_strategy.time_key or self.time_key
        if time_key and time_key in inputs:
            time_values = np.asarray(inputs[time_key])
            if time_values.ndim == 1 and time_values.shape[0] == y.shape[0]:
                rows["time_key"] = time_key
                rows["time_bins"] = self._time_objective_diagnostics(y, preds, residual, time_values)
        role_groups: dict[str, object] = {}
        for role in ("regime", "fault_mode", "event", "censoring"):
            key = _schema_role_key(self.task_schema, role)
            if not key or key == group_key or key not in inputs:
                continue
            values = np.asarray(inputs[key])
            if values.ndim == 1 and values.shape[0] == y.shape[0]:
                role_groups[role] = {"key": key, "groups": self._group_objective_diagnostics(y, preds, residual, values)}
        if role_groups:
            rows["role_groups"] = role_groups
        return rows

    def _schema_role_map(self) -> dict[str, list[str]]:
        return self.task_schema.role_map() if self.task_schema is not None else {}

    def _target_bin_diagnostics(self, y: np.ndarray, preds: np.ndarray, residual: np.ndarray, bins: int = 5) -> list[dict[str, object]]:
        y_array = np.asarray(y, dtype=np.float64)
        pred_array = np.asarray(preds)
        if y_array.size == 0:
            return []
        edges = np.unique(np.quantile(y_array, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size <= 2:
            labels = np.zeros(y_array.shape[0], dtype=np.int64)
            n_bins = 1
        else:
            labels = np.searchsorted(edges[1:-1], y_array, side="right")
            n_bins = int(edges.size - 1)
        rows: list[dict[str, object]] = []
        for label in range(n_bins):
            mask = labels == label
            if not np.any(mask):
                continue
            bin_score, bin_raw_score = _score_and_raw(self.scorer, y_array[mask], pred_array[mask])
            rows.append(
                {
                    "bin": int(label),
                    "n": int(np.sum(mask)),
                    "target_min": float(np.min(y_array[mask])),
                    "target_max": float(np.max(y_array[mask])),
                    "score": float(bin_score),
                    "raw_score": float(bin_raw_score),
                    "residual_abs_mean": float(np.mean(np.abs(residual[mask]))),
                }
            )
        return rows

    def _group_objective_diagnostics(
        self,
        y: np.ndarray,
        preds: np.ndarray,
        residual: np.ndarray,
        groups: np.ndarray,
        *,
        max_groups: int = 20,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        unique = np.unique(groups)
        for group in unique[:max_groups]:
            mask = groups == group
            if not np.any(mask):
                continue
            group_score, group_raw_score = _score_and_raw(self.scorer, y[mask], preds[mask])
            rows.append(
                {
                    "group": _jsonable_scalar(group),
                    "n": int(np.sum(mask)),
                    "score": float(group_score),
                    "raw_score": float(group_raw_score),
                    "residual_mean": float(np.mean(residual[mask])),
                    "residual_abs_mean": float(np.mean(np.abs(residual[mask]))),
                }
            )
        rows.sort(key=lambda row: float(row["raw_score"]), reverse=bool(self.scorer.higher_is_better))
        return rows

    def _time_objective_diagnostics(self, y: np.ndarray, preds: np.ndarray, residual: np.ndarray, time_values: np.ndarray, bins: int = 5) -> list[dict[str, object]]:
        y_array = np.asarray(y, dtype=np.float64)
        time_array = np.asarray(time_values)
        if time_array.size == 0:
            return []
        if np.issubdtype(time_array.dtype, np.number):
            order_values = np.asarray(time_array, dtype=np.float64)
        else:
            order_values = np.arange(time_array.shape[0], dtype=np.float64)
        edges = np.unique(np.quantile(order_values, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size <= 2:
            labels = np.zeros(order_values.shape[0], dtype=np.int64)
            n_bins = 1
        else:
            labels = np.searchsorted(edges[1:-1], order_values, side="right")
            n_bins = int(edges.size - 1)
        rows: list[dict[str, object]] = []
        pred_array = np.asarray(preds)
        for label in range(n_bins):
            mask = labels == label
            if not np.any(mask):
                continue
            bin_score, bin_raw_score = _score_and_raw(self.scorer, y_array[mask], pred_array[mask])
            rows.append(
                {
                    "bin": int(label),
                    "n": int(np.sum(mask)),
                    "time_min": _jsonable_scalar(np.min(time_array[mask])),
                    "time_max": _jsonable_scalar(np.max(time_array[mask])),
                    "score": float(bin_score),
                    "raw_score": float(bin_raw_score),
                    "prediction_mean": float(np.mean(pred_array[mask])),
                    "residual_mean": float(np.mean(residual[mask])),
                    "residual_abs_mean": float(np.mean(np.abs(residual[mask]))),
                }
            )
        return rows


def effective_rank(x: np.ndarray) -> float:
    if x.size == 0 or x.shape[1] == 0:
        return 0.0
    centered = x - np.mean(x, axis=0, keepdims=True)
    scale = np.std(centered, axis=0)
    valid = scale >= 1e-12
    if not np.any(valid):
        return 0.0
    normalized = centered[:, valid] / scale[valid].reshape(1, -1)
    singular_values = np.linalg.svd(normalized, full_matrices=False, compute_uv=False)
    total = float(np.sum(singular_values))
    if total < 1e-12:
        return 0.0
    probabilities = singular_values / total
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return float(np.exp(entropy))


def abs_feature_correlation(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 1:
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    centered = x - np.mean(x, axis=0, keepdims=True)
    scale = np.std(centered, axis=0)
    valid = scale >= 1e-12
    corr = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    if np.sum(valid) <= 1:
        return corr
    normalized = centered[:, valid] / scale[valid].reshape(1, -1)
    valid_corr = np.abs(np.corrcoef(normalized, rowvar=False))
    valid_corr = np.nan_to_num(valid_corr, nan=0.0, posinf=0.0, neginf=0.0)
    valid_indices = np.flatnonzero(valid)
    corr[np.ix_(valid_indices, valid_indices)] = valid_corr
    np.fill_diagonal(corr, 0.0)
    return corr


def feature_correlation_summary(x: np.ndarray, names: list[str], *, threshold: float = 0.9, block_size: int = 512) -> FeatureCorrelationSummary:
    n_features = int(x.shape[1])
    if n_features <= 1 or x.shape[0] == 0:
        return FeatureCorrelationSummary(
            max_corr=np.zeros(n_features, dtype=np.float64),
            high_corr_counts=np.zeros(n_features, dtype=np.int64),
            most_correlated=["" for _name in names],
            effective_rank=0.0,
        )
    mean = np.mean(x, axis=0, keepdims=True)
    scale = np.std(x, axis=0)
    valid = scale >= 1e-12
    valid_indices = np.flatnonzero(valid)
    if valid_indices.shape[0] <= 1:
        return FeatureCorrelationSummary(
            max_corr=np.zeros(n_features, dtype=np.float64),
            high_corr_counts=np.zeros(n_features, dtype=np.int64),
            most_correlated=["" for _name in names],
            effective_rank=0.0,
        )

    normalized = (x[:, valid_indices] - mean[:, valid_indices]) / scale[valid_indices].reshape(1, -1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    singular_values = np.linalg.svd(normalized, full_matrices=False, compute_uv=False)
    total = float(np.sum(singular_values))
    if total < 1e-12:
        rank = 0.0
    else:
        probabilities = singular_values / total
        rank = float(np.exp(-float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))))
    max_corr = np.zeros(n_features, dtype=np.float64)
    high_corr_counts = np.zeros(n_features, dtype=np.int64)
    best_indices = np.full(n_features, -1, dtype=np.int64)
    n_rows = max(int(x.shape[0]), 1)
    effective_block = max(1, int(block_size))
    for start in range(0, valid_indices.shape[0], effective_block):
        end = min(start + effective_block, valid_indices.shape[0])
        corr_block = np.abs((normalized[:, start:end].T @ normalized) / n_rows)
        corr_block = np.nan_to_num(corr_block, nan=0.0, posinf=0.0, neginf=0.0)
        for local_row, valid_col in enumerate(range(start, end)):
            corr_block[local_row, valid_col] = 0.0
        row_max = np.max(corr_block, axis=1)
        row_argmax = np.argmax(corr_block, axis=1)
        original_indices = valid_indices[start:end]
        max_corr[original_indices] = row_max
        high_corr_counts[original_indices] = np.sum(corr_block > threshold, axis=1)
        best_indices[original_indices] = valid_indices[row_argmax]

    most_correlated = [
        names[int(index)] if index >= 0 and float(max_corr[row]) > 0.0 else ""
        for row, index in enumerate(best_indices)
    ]
    return FeatureCorrelationSummary(max_corr=max_corr, high_corr_counts=high_corr_counts, most_correlated=most_correlated, effective_rank=rank)


def safe_corr_columns(x: np.ndarray, y: np.ndarray, *, square: bool = False, block_size: int = 1024) -> np.ndarray:
    x_array = np.asarray(x, dtype=np.float64)
    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError(f"Correlation arrays must have identical row counts, got {x_array.shape[0]} and {y_array.shape[0]}.")
    out = np.zeros(x_array.shape[1], dtype=np.float64)
    if x_array.shape[0] == 0 or x_array.shape[1] == 0:
        return out
    y_clean = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
    y_std = float(np.std(y_clean))
    if y_std < 1e-12:
        return out
    y_centered = y_clean - np.mean(y_clean)
    y_energy = float(np.sum(y_centered * y_centered))
    effective_block = max(1, int(block_size))
    for start in range(0, x_array.shape[1], effective_block):
        end = min(start + effective_block, x_array.shape[1])
        block = np.asarray(x_array[:, start:end], dtype=np.float64)
        if square:
            block = block * block
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        block_std = np.std(block, axis=0)
        valid = block_std >= 1e-12
        if not np.any(valid):
            continue
        centered = block - np.mean(block, axis=0, keepdims=True)
        x_energy = np.sum(centered * centered, axis=0)
        denom = np.sqrt(x_energy * y_energy)
        valid &= denom >= 1e-12
        if not np.any(valid):
            continue
        values = (centered[:, valid].T @ y_centered) / denom[valid]
        out[start:end][valid] = values
    return out


def max_correlation_scores(corr: np.ndarray) -> np.ndarray:
    if corr.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.max(corr, axis=1)


def most_correlated_names(corr: np.ndarray, names: list[str]) -> list[str]:
    if corr.size == 0:
        return ["" for _name in names]
    indices = np.argmax(corr, axis=1)
    return [names[int(index)] if float(corr[row, int(index)]) > 0.0 else "" for row, index in enumerate(indices)]


def redundancy_scores(x: np.ndarray) -> np.ndarray:
    return max_correlation_scores(abs_feature_correlation(x))


def feature_dependency_rows(feature_names: list[str], dependency_map: dict[str, list[str]]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for name in feature_names:
        matched = ""
        for prefix in dependency_map:
            if name == prefix or name.startswith(f"{prefix}."):
                if len(prefix) > len(matched):
                    matched = prefix
        rows[name] = dependency_map.get(matched, [])
    return rows


def subnode_diagnostics(features: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregates: dict[str, dict[str, object]] = {}
    for feature in features:
        dependencies = feature.get("dependencies", [])
        if not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            row = aggregates.setdefault(
                str(dependency),
                {
                    "name": str(dependency),
                    "feature_count": 0,
                    "importance": 0.0,
                    "shap_importance": 0.0,
                    "max_target_alignment": 0.0,
                    "mean_abs_shap": 0.0,
                    "mean_abs_residual_corr": 0.0,
                    "mean_redundancy": 0.0,
                    "mean_weight_stability": 0.0,
                },
            )
            row["feature_count"] = int(row["feature_count"]) + 1
            row["importance"] = float(row["importance"]) + float(feature.get("importance", 0.0))
            row["shap_importance"] = float(row["shap_importance"]) + float(feature.get("shap_importance", 0.0))
            row["max_target_alignment"] = max(float(row["max_target_alignment"]), float(feature.get("target_alignment", 0.0)))
            row["mean_abs_shap"] = float(row["mean_abs_shap"]) + float(feature.get("shap_mean_abs", 0.0))
            row["mean_abs_residual_corr"] = float(row["mean_abs_residual_corr"]) + abs(float(feature.get("residual_corr", 0.0)))
            row["mean_redundancy"] = float(row["mean_redundancy"]) + float(feature.get("redundancy", 0.0))
            row["mean_weight_stability"] = float(row["mean_weight_stability"]) + float(feature.get("weight_stability", 0.0))
    for row in aggregates.values():
        count = max(int(row["feature_count"]), 1)
        row["mean_abs_shap"] = float(row["mean_abs_shap"]) / count
        row["mean_abs_residual_corr"] = float(row["mean_abs_residual_corr"]) / count
        row["mean_redundancy"] = float(row["mean_redundancy"]) / count
        row["mean_weight_stability"] = float(row["mean_weight_stability"]) / count
    return sorted(aggregates.values(), key=lambda row: float(row["importance"]), reverse=True)


def alternative_diagnostics(
    features: list[dict[str, object]],
    selected_alternatives: dict[str, str],
    config_score: float,
) -> list[dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for row in subnode_diagnostics(features):
        name = str(row["name"])
        node, alternative = split_alternative_name(name)
        rows[name] = {
            **row,
            "node": node,
            "alternative": alternative,
            "selected": False,
            "config_score": float(config_score),
        }

    for node, alternative in selected_alternatives.items():
        name = f"{node}.{alternative}"
        rows.setdefault(
            name,
            {
                "name": name,
                "node": node,
                "alternative": alternative,
                "feature_count": 0,
                "importance": 0.0,
                "shap_importance": 0.0,
                "max_target_alignment": 0.0,
                "mean_abs_shap": 0.0,
                "mean_abs_residual_corr": 0.0,
                "mean_redundancy": 0.0,
                "mean_weight_stability": 0.0,
                "config_score": float(config_score),
            },
        )
        rows[name]["selected"] = True
        rows[name]["config_score"] = float(config_score)

    return sorted(
        rows.values(),
        key=lambda row: (
            -float(row.get("importance", 0.0)),
            -float(row.get("max_target_alignment", 0.0)),
            str(row.get("name", "")),
        ),
    )


def split_alternative_name(name: str) -> tuple[str, str]:
    if "." not in name:
        return name, ""
    return name.split(".", maxsplit=1)


def _coerce_task_schema(task_schema: TaskSchema | dict[str, object] | None) -> TaskSchema | None:
    if task_schema is None:
        return None
    if isinstance(task_schema, TaskSchema):
        return task_schema
    return TaskSchema.from_dict(dict(task_schema))


def _schema_role_key(task_schema: TaskSchema | None, *roles: str) -> str | None:
    if task_schema is None:
        return None
    return task_schema.input_name_with_role(*roles)


def _jsonable_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
