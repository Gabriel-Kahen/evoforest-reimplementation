from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from evoforest_arch.graph import EvalContext, Graph, ResidualWeightRule
from evoforest_arch.metrics import DEFAULT_SCORER, FoldStrategy, ScoreFunction, TaskScorer, coerce_fold_strategy, coerce_scorer, safe_corr, target_alignment
from evoforest_arch.readout import DEFAULT_ALPHAS, RidgeModel, Standardizer, combine_sample_weights, fit_ridge, normalize_sample_weight, select_alpha


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
        raw_score = scorer.raw_score(y, self.predictions)
        return {
            "alpha": float(self.alpha),
            "score": float(scorer.score(y, self.predictions)),
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
class ReweightedRidgeFit:
    model: RidgeModel
    alpha: float
    sample_weight: np.ndarray | None
    irls_iterations: list[dict[str, object]]


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
        torch_device: str | None = None,
        scorer: TaskScorer | ScoreFunction | str | None = None,
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
        self.group_key = group_key
        self.fold_strategy = coerce_fold_strategy(fold_strategy, group_key=group_key, time_key=time_key, stratify_bins=stratify_bins)
        self.scorer = coerce_scorer(scorer)
        if diagnostics_mode not in {"full", "basic"}:
            raise ValueError("diagnostics_mode must be 'full' or 'basic'.")
        self.diagnostics_mode = diagnostics_mode

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
        shared_cache: dict[object, object] = {}
        search_cache_hits = 0
        search_cache_misses = 0
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
            result = self._evaluate_single_config(working_graph, inputs, y, selected, shared_cache)
            feature_pool = [(result.feature_matrix, result.feature_names)]
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
            config_rows: list[dict[str, object]] = []
            feature_pool: list[tuple[np.ndarray | None, list[str]]] = []
            for candidate_config in configs:
                result = self._evaluate_single_config(working_graph, inputs, y, candidate_config, shared_cache)
                feature_pool.append((result.feature_matrix, result.feature_names))
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
                if best is None or result.score > best.score:
                    best = result
            if best is None:
                raise ValueError("No graph configurations were available for evaluation.")
            result = best
            result.diagnostics["configuration_search"] = self._configuration_search_diagnostics(config_rows, total_configurations)
        search_cache = {
            "hits": int(search_cache_hits),
            "misses": int(search_cache_misses),
            "entries": int(len(shared_cache)),
            "shared_across_configurations": True,
            "key": "ancestor_conditioned_subpath",
        }
        result.diagnostics["configuration_search"]["cache"] = search_cache
        result.diagnostics["cache"] = {
            **(result.diagnostics.get("cache", {}) if isinstance(result.diagnostics.get("cache", {}), dict) else {}),
            "search_hits": search_cache["hits"],
            "search_misses": search_cache["misses"],
            "search_entries": search_cache["entries"],
            "shared_across_configurations": True,
            "key": "ancestor_conditioned_subpath",
        }
        result.diagnostics["scoring_context"] = self._scoring_context(result)
        self._attach_valid_feature_pool_diagnostics(result, y, feature_pool)
        result.diagnostics["scoring_context"] = self._scoring_context(result)
        result.diagnostics["refinement"] = refinement_diagnostics
        if update_graph:
            alternatives = result.diagnostics.get("alternatives", [])
            if isinstance(alternatives, list):
                working_graph.update_alternative_statistics([row for row in alternatives if isinstance(row, dict)])
        result.diagnostics["alternative_stats"] = working_graph.alternative_statistics_snapshot()
        return result

    def _evaluate_single_config(
        self,
        graph: Graph,
        inputs: dict[str, object],
        y: np.ndarray,
        config: dict[str, str],
        shared_cache: dict[object, object] | None = None,
    ) -> EvaluationResult:
        x, names, ctx = graph.evaluate_features(inputs, config=config, cache=shared_cache)
        sample_weight, residual_rule, fitting_diagnostics = self._evaluate_fitting_rules(graph, inputs, y, config, ctx)
        preds = np.zeros_like(y, dtype=np.float64)
        fold_contributions = np.zeros_like(x, dtype=np.float64)
        fold_intercepts = np.zeros_like(y, dtype=np.float64)
        alphas: list[float] = []
        coefs: list[np.ndarray] = []
        fold_scores: list[float] = []
        fold_raw_scores: list[float] = []
        fold_irls: list[dict[str, object]] = []
        folds, fold_diagnostics = self._folds(inputs, y)
        for fold_index, (train_idx, val_idx) in enumerate(folds):
            std = Standardizer.fit(x[train_idx])
            x_train = std.transform(x[train_idx])
            x_val = std.transform(x[val_idx])
            train_weight = sample_weight[train_idx] if sample_weight is not None else None
            ridge_fit = self._fit_reweighted_ridge(x_train, y[train_idx], train_weight, residual_rule)
            model = ridge_fit.model
            preds[val_idx] = model.predict(x_val)
            fold_contributions[val_idx] = x_val * model.coef.reshape(1, -1)
            fold_intercepts[val_idx] = float(model.intercept)
            fold_scores.append(float(self.scorer.score(y[val_idx], preds[val_idx])))
            fold_raw_scores.append(float(self.scorer.raw_score(y[val_idx], preds[val_idx])))
            alphas.append(float(ridge_fit.alpha))
            coefs.append(model.coef)
            fold_irls.append(
                {
                    "fold": int(fold_index),
                    "steps_used": len(ridge_fit.irls_iterations),
                    "final_alpha": float(ridge_fit.alpha),
                    "iterations": ridge_fit.irls_iterations,
                }
            )
        score = float(np.mean(fold_scores)) if fold_scores else self.scorer.score(y, preds)
        selected_alternatives = graph.selected_alternatives(config)
        if self.diagnostics_mode == "full":
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
                fold_contributions,
                fold_intercepts,
                global_fit,
            )
            diagnostics["alternatives"] = alternative_diagnostics(diagnostics["features"], selected_alternatives, score)
        else:
            diagnostics = self._basic_diagnostics(preds, y, inputs, names, selected_alternatives, score)
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

    def _attach_valid_feature_pool_diagnostics(
        self,
        result: EvaluationResult,
        y: np.ndarray,
        feature_pool: list[tuple[np.ndarray | None, list[str]]],
    ) -> None:
        matrices = [matrix for matrix, _names in feature_pool if matrix is not None and matrix.size]
        if not matrices:
            return
        x_pool = np.column_stack(matrices)
        names = [name for _matrix, names_ in feature_pool for name in names_]
        pool_fit = self._fit_global_diagnostic_model(x_pool, y, None, None)
        result.diagnostics["valid_feature_pool"] = {
            "n_configurations": int(len(matrices)),
            "n_features": int(x_pool.shape[1]),
            "n_feature_names": int(len(names)),
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
        alpha = select_alpha(x, y, self.alphas, sample_weight=current_weight)
        model = fit_ridge(x, y, alpha, sample_weight=current_weight)
        irls_iterations: list[dict[str, object]] = []
        if residual_rule is None or residual_rule.name == "identity" or self.irls_steps == 0:
            return ReweightedRidgeFit(model=model, alpha=float(alpha), sample_weight=current_weight, irls_iterations=irls_iterations)

        for step in range(1, self.irls_steps + 1):
            residual = y - model.predict(x)
            residual_weight = normalize_sample_weight(residual_rule.apply(residual), y.shape[0])
            current_weight = combine_sample_weights(base_weight, residual_weight)
            alpha = select_alpha(x, y, self.alphas, sample_weight=current_weight)
            model = fit_ridge(x, y, alpha, sample_weight=current_weight)
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
        fold_contributions: np.ndarray,
        fold_intercepts: np.ndarray,
        global_fit: LinearDiagnosticFit,
    ) -> dict[str, object]:
        coef_matrix = np.vstack(coefs) if coefs else np.zeros((1, x.shape[1]))
        mean_coef = np.mean(coef_matrix, axis=0)
        coef_std = np.std(coef_matrix, axis=0)
        importance = np.mean(np.abs(coef_matrix), axis=0)
        total = max(float(np.sum(importance)), 1e-8)
        global_abs_contribution = np.mean(np.abs(global_fit.contributions), axis=0) if global_fit.contributions.size else np.zeros(x.shape[1], dtype=np.float64)
        fold_abs_contribution = np.mean(np.abs(fold_contributions), axis=0) if fold_contributions.size else np.zeros(x.shape[1], dtype=np.float64)
        shap_total = max(float(np.sum(global_abs_contribution)), 1e-8)
        feature_rows = []
        residual = y - preds
        corr = abs_feature_correlation(x)
        redundancy = max_correlation_scores(corr)
        high_corr_counts = np.sum(corr > 0.9, axis=1) if corr.size else np.zeros(x.shape[1], dtype=np.int64)
        most_correlated = most_correlated_names(corr, names)
        for idx, name in enumerate(names):
            feature = x[:, idx]
            contribution = global_fit.contributions[:, idx]
            dependencies = feature_dependencies.get(name, [])
            feature_rows.append(
                {
                    "name": name,
                    "dependencies": dependencies,
                    "depth": len(dependencies),
                    "importance": float(importance[idx] / total),
                    "coef_sign": int(np.sign(mean_coef[idx])),
                    "target_alignment": float(target_alignment(feature, y)),
                    "target_corr": float(safe_corr(feature, y)),
                    "target_quadratic_corr": float(safe_corr(feature * feature, y)),
                    "max_corr": float(redundancy[idx]),
                    "n_high_corr": int(high_corr_counts[idx]),
                    "most_correlated": most_correlated[idx],
                    "redundancy": float(redundancy[idx]),
                    "residual_corr": float(safe_corr(feature, residual)),
                    "residual_quadratic_corr": float(safe_corr(feature * feature, residual)),
                    "weight_stability": float(abs(mean_coef[idx]) / max(coef_std[idx], 1e-8)),
                    "mean_abs_coef": float(importance[idx]),
                    "global_coef": float(global_fit.coef[idx]),
                    "shap_mean": float(np.mean(contribution)),
                    "shap_mean_abs": float(global_abs_contribution[idx]),
                    "shap_std": float(np.std(contribution)),
                    "shap_importance": float(global_abs_contribution[idx] / shap_total),
                    "shap_target_corr": float(safe_corr(contribution, y)),
                    "cv_shap_mean_abs": float(fold_abs_contribution[idx]),
                }
            )
        feature_rows.sort(key=lambda row: row["importance"], reverse=True)
        cv_reconstructed = fold_intercepts + np.sum(fold_contributions, axis=1)
        global_ridge = global_fit.to_dict(y, self.scorer)
        return {
            "prediction_std": float(np.std(preds)),
            "residual_std": float(np.std(residual)),
            "objective": self._objective_diagnostics(y, preds, inputs),
            "effective_rank": float(effective_rank(x)),
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
        rows: dict[str, object] = {
            "score": float(self.scorer.score(y, preds)),
            "raw_score": float(self.scorer.raw_score(y, preds)),
            "raw_metric": self.scorer.raw_name or self.scorer.name,
            "raw_higher_is_better": bool(self.scorer.higher_is_better),
            "residual_mean": float(np.mean(residual)) if residual.size else 0.0,
            "residual_std": float(np.std(residual)) if residual.size else 0.0,
            "residual_abs_mean": float(np.mean(np.abs(residual))) if residual.size else 0.0,
            "target_bins": self._target_bin_diagnostics(y, preds, residual),
        }
        group_key = self.fold_strategy.group_key or self.group_key
        if group_key and group_key in inputs:
            group_values = np.asarray(inputs[group_key])
            if group_values.ndim == 1 and group_values.shape[0] == y.shape[0]:
                rows["group_key"] = group_key
                rows["groups"] = self._group_objective_diagnostics(y, preds, residual, group_values)
                rows["split_leakage_check"] = "fold diagnostics report group overlap when grouped folds are active"
        return rows

    def _target_bin_diagnostics(self, y: np.ndarray, preds: np.ndarray, residual: np.ndarray, bins: int = 5) -> list[dict[str, object]]:
        y_array = np.asarray(y, dtype=np.float64)
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
            rows.append(
                {
                    "bin": int(label),
                    "n": int(np.sum(mask)),
                    "target_min": float(np.min(y_array[mask])),
                    "target_max": float(np.max(y_array[mask])),
                    "score": float(self.scorer.score(y_array[mask], np.asarray(preds)[mask])),
                    "raw_score": float(self.scorer.raw_score(y_array[mask], np.asarray(preds)[mask])),
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
            rows.append(
                {
                    "group": _jsonable_scalar(group),
                    "n": int(np.sum(mask)),
                    "score": float(self.scorer.score(y[mask], preds[mask])),
                    "raw_score": float(self.scorer.raw_score(y[mask], preds[mask])),
                    "residual_mean": float(np.mean(residual[mask])),
                    "residual_abs_mean": float(np.mean(np.abs(residual[mask]))),
                }
            )
        rows.sort(key=lambda row: float(row["raw_score"]), reverse=bool(self.scorer.higher_is_better))
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


def _jsonable_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
