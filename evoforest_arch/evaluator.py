from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from evoforest_arch.graph import EvalContext, Graph, ResidualWeightRule
from evoforest_arch.metrics import roc_auc_score, stratified_folds, stratified_group_folds
from evoforest_arch.readout import DEFAULT_ALPHAS, RidgeModel, Standardizer, combine_sample_weights, fit_ridge, normalize_sample_weight, select_alpha


@dataclass
class EvaluationResult:
    auc: float
    config: dict[str, str]
    feature_names: list[str]
    predictions: np.ndarray
    alphas: list[float]
    diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "auc": self.auc,
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

    def to_dict(self, y: np.ndarray) -> dict[str, object]:
        reconstructed = self.intercept + np.sum(self.contributions, axis=1)
        return {
            "alpha": float(self.alpha),
            "auc": float(roc_auc_score(y, self.predictions)),
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
        refine_globals: bool = False,
        refine_steps: int = 20,
        refine_backend: str = "auto",
        irls_steps: int = 2,
        group_key: str | None = None,
        diagnostics_mode: str = "full",
    ) -> None:
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.max_configurations = max(1, int(max_configurations))
        self.refine_globals = bool(refine_globals)
        self.refine_steps = int(refine_steps)
        self.refine_backend = refine_backend
        self.irls_steps = max(0, int(irls_steps))
        self.group_key = group_key
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
        if self.refine_globals:
            from evoforest_arch.refinement import GlobalRefiner

            base_config = working_graph.default_config()
            if config:
                base_config.update(config)
            refinement = GlobalRefiner(steps=self.refine_steps, seed=self.seed, backend=self.refine_backend).refine(working_graph, inputs, y, base_config)
            refinement_diagnostics = refinement.to_dict()

        if config is not None:
            selected = working_graph.default_config()
            selected.update(config)
            result = self._evaluate_single_config(working_graph, inputs, y, selected, shared_cache)
            cache_row = result.diagnostics.get("cache", {})
            if isinstance(cache_row, dict):
                search_cache_hits += int(cache_row.get("hits", 0))
                search_cache_misses += int(cache_row.get("misses", 0))
            result.diagnostics["configuration_search"] = self._configuration_search_diagnostics(
                [{"auc": result.auc, "config": result.config, "n_features": len(result.feature_names)}],
                total_configurations=1,
            )
        else:
            configs, total_configurations = self._configuration_candidates(working_graph)
            best: EvaluationResult | None = None
            config_rows: list[dict[str, object]] = []
            for candidate_config in configs:
                result = self._evaluate_single_config(working_graph, inputs, y, candidate_config, shared_cache)
                cache_row = result.diagnostics.get("cache", {})
                if isinstance(cache_row, dict):
                    search_cache_hits += int(cache_row.get("hits", 0))
                    search_cache_misses += int(cache_row.get("misses", 0))
                config_rows.append(
                    {
                        "auc": result.auc,
                        "config": result.config,
                        "n_features": len(result.feature_names),
                        "cache_hits": int(cache_row.get("hits", 0)) if isinstance(cache_row, dict) else 0,
                        "cache_misses": int(cache_row.get("misses", 0)) if isinstance(cache_row, dict) else 0,
                    }
                )
                if best is None or result.auc > best.auc:
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
        fold_aucs: list[float] = []
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
            fold_aucs.append(float(roc_auc_score(y[val_idx], preds[val_idx])))
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
        auc = roc_auc_score(y, preds)
        selected_alternatives = graph.selected_alternatives(config)
        if self.diagnostics_mode == "full":
            feature_dependencies = feature_dependency_rows(names, graph.output_dependency_map(config))
            global_fit = self._fit_global_diagnostic_model(x, y, sample_weight, residual_rule)
            diagnostics = self._diagnostics(
                x,
                y,
                preds,
                names,
                coefs,
                feature_dependencies,
                fold_contributions,
                fold_intercepts,
                global_fit,
            )
            diagnostics["alternatives"] = alternative_diagnostics(diagnostics["features"], selected_alternatives, auc)
        else:
            diagnostics = self._basic_diagnostics(preds, y, names, selected_alternatives, auc)
        diagnostics["folds"] = {
            "auc": fold_aucs,
            "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else 0.0,
            "auc_std": float(np.std(fold_aucs)) if fold_aucs else 0.0,
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
            auc=float(auc),
            config=dict(config),
            feature_names=names,
            predictions=preds,
            alphas=alphas,
            diagnostics=diagnostics,
        )

    def _folds(self, inputs: dict[str, object], y: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
        if self.group_key is None or self.group_key not in inputs:
            return stratified_folds(y, self.n_splits, self.seed), {"method": "stratified_random", "group_key": None}
        groups = np.asarray(inputs[self.group_key])
        if groups.ndim != 1 or groups.shape[0] != y.shape[0]:
            return stratified_folds(y, self.n_splits, self.seed), {
                "method": "stratified_random",
                "group_key": self.group_key,
                "grouped": False,
                "reason": "group array shape did not match y",
            }
        folds = stratified_group_folds(y, groups, self.n_splits, self.seed)
        overlap_count = 0
        validation_group_counts: list[int] = []
        for train_idx, val_idx in folds:
            train_groups = set(np.asarray(groups)[train_idx].tolist())
            val_groups = set(np.asarray(groups)[val_idx].tolist())
            overlap_count += len(train_groups & val_groups)
            validation_group_counts.append(len(val_groups))
        return folds, {
            "method": "stratified_group_random",
            "group_key": self.group_key,
            "grouped": True,
            "fold_group_overlap_count": int(overlap_count),
            "validation_group_counts": validation_group_counts,
            "n_groups": int(np.unique(groups).shape[0]),
        }

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
        aucs = np.asarray([float(row["auc"]) for row in config_rows], dtype=np.float64)
        n_features = [int(row.get("n_features", 0)) for row in config_rows]
        top_rows = sorted(config_rows, key=lambda row: float(row["auc"]), reverse=True)[:8]
        if aucs.size == 0:
            return {
                "evaluated": 0,
                "total": int(total_configurations),
                "capped": False,
                "auc_range": [0.0, 0.0],
                "auc_mean": 0.0,
                "auc_std": 0.0,
                "best_config_auc": 0.0,
                "n_features_global": 0,
                "n_features_best_config": 0,
                "top_configs": [],
            }
        return {
            "evaluated": len(config_rows),
            "total": int(total_configurations),
            "capped": int(total_configurations) > len(config_rows),
            "auc_range": [float(np.min(aucs)), float(np.max(aucs))],
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "best_config_auc": float(np.max(aucs)),
            "n_features_global": int(np.sum(n_features)),
            "n_features_best_config": int(top_rows[0].get("n_features", 0)) if top_rows else 0,
            "top_configs": [
                {
                    "auc": float(row["auc"]),
                    "n_features": int(row.get("n_features", 0)),
                    "config": row["config"],
                }
                for row in top_rows
            ],
        }

    @staticmethod
    def _scoring_context(result: EvaluationResult) -> dict[str, object]:
        search = result.diagnostics.get("configuration_search", {})
        folds = result.diagnostics.get("folds", {})
        global_ridge = result.diagnostics.get("global_ridge", {})
        linear_shap = result.diagnostics.get("linear_shap", {})
        return {
            "scoring": "configuration-based (best config AUC = evoforest score)",
            "best_config_auc": float(result.auc),
            "global_ridge_auc": float(global_ridge.get("auc", 0.0)) if isinstance(global_ridge, dict) else 0.0,
            "config_auc_range": search.get("auc_range", [float(result.auc), float(result.auc)]),
            "fold_auc_std": float(folds.get("auc_std", 0.0)) if isinstance(folds, dict) else 0.0,
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

    @staticmethod
    def _basic_diagnostics(
        preds: np.ndarray,
        y: np.ndarray,
        names: list[str],
        selected_alternatives: dict[str, str],
        auc: float,
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
                    "last_auc": float(auc),
                }
                for node, alternative in selected_alternatives.items()
            ],
            "diagnostics_mode": "basic",
            "n_features": int(len(names)),
        }

    @staticmethod
    def _diagnostics(
        x: np.ndarray,
        y: np.ndarray,
        preds: np.ndarray,
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
                    "individual_auc": float(max(roc_auc_score(y, feature), 1.0 - roc_auc_score(y, feature))),
                    "rank_corr": float(rank_corr(feature, y)),
                    "class_effect": float(class_effect_size(feature, y)),
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
                    "shap_class_effect": float(class_effect_size(contribution, y)),
                    "cv_shap_mean_abs": float(fold_abs_contribution[idx]),
                }
            )
        feature_rows.sort(key=lambda row: row["importance"], reverse=True)
        cv_reconstructed = fold_intercepts + np.sum(fold_contributions, axis=1)
        global_ridge = global_fit.to_dict(y)
        return {
            "prediction_std": float(np.std(preds)),
            "residual_std": float(np.std(residual)),
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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return safe_corr(rankdata(a), rankdata(b))


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    unique_values, first_indices, counts = np.unique(values[order], return_index=True, return_counts=True)
    del unique_values
    for first, count in zip(first_indices, counts, strict=True):
        if count > 1:
            tied = order[first : first + count]
            ranks[tied] = float(np.mean(ranks[tied]))
    return ranks


def class_effect_size(feature: np.ndarray, y: np.ndarray) -> float:
    feature = np.asarray(feature, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pos = feature[y > 0.5]
    neg = feature[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    pooled = np.sqrt(0.5 * (np.var(pos) + np.var(neg)))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(pos) - np.mean(neg)) / pooled)


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
                    "max_feature_auc": 0.0,
                    "mean_abs_shap": 0.0,
                    "mean_abs_residual_corr": 0.0,
                    "mean_redundancy": 0.0,
                    "mean_weight_stability": 0.0,
                },
            )
            row["feature_count"] = int(row["feature_count"]) + 1
            row["importance"] = float(row["importance"]) + float(feature.get("importance", 0.0))
            row["shap_importance"] = float(row["shap_importance"]) + float(feature.get("shap_importance", 0.0))
            row["max_feature_auc"] = max(float(row["max_feature_auc"]), float(feature.get("individual_auc", 0.0)))
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
    config_auc: float,
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
            "config_auc": float(config_auc),
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
                "max_feature_auc": 0.0,
                "mean_abs_shap": 0.0,
                "mean_abs_residual_corr": 0.0,
                "mean_redundancy": 0.0,
                "mean_weight_stability": 0.0,
                "config_auc": float(config_auc),
            },
        )
        rows[name]["selected"] = True
        rows[name]["config_auc"] = float(config_auc)

    return sorted(
        rows.values(),
        key=lambda row: (
            -float(row.get("importance", 0.0)),
            -float(row.get("max_feature_auc", 0.0)),
            str(row.get("name", "")),
        ),
    )


def split_alternative_name(name: str) -> tuple[str, str]:
    if "." not in name:
        return name, ""
    return name.split(".", maxsplit=1)
