from __future__ import annotations

from evoforest_arch.evaluator import EvaluationResult


def feedback_summary(result: EvaluationResult, max_features: int = 8) -> dict[str, object]:
    features = result.diagnostics.get("features", [])
    subnodes = result.diagnostics.get("subnodes", [])
    alternatives = alternative_rows(result)
    risky = [
        row
        for row in features
        if float(row.get("redundancy", 0.0)) > 0.98 or float(row.get("weight_stability", 0.0)) < 0.25
    ]
    return {
        "auc": float(result.auc),
        "config": result.config,
        "top_features": features[:max_features],
        "top_subnodes": subnodes[:max_features],
        "top_alternatives": alternatives[:max_features],
        "risky_features": risky[:max_features],
        "cache": result.diagnostics.get("cache", {}),
        "graph": result.diagnostics.get("graph", {}),
        "configuration_search": result.diagnostics.get("configuration_search", {}),
        "scoring_context": result.diagnostics.get("scoring_context", {}),
        "folds": result.diagnostics.get("folds", {}),
        "fitting": result.diagnostics.get("fitting", {}),
        "guidance": [
            "Prefer mutations that increase residual-correlation coverage.",
            "Use SHAP-style contribution fields to separate high-weight features from high-impact features.",
            "Add alternatives to underexplored intermediate nodes.",
            "Prune or rewrite highly redundant output alternatives.",
            "Use ridge_w/ridge_g alternatives when residuals show unstable or heavy-tailed behavior.",
            "Keep output dimensionality small enough for stable Ridge evaluation.",
        ],
    }


def toon_report(result: EvaluationResult, max_features: int = 12) -> str:
    features = result.diagnostics.get("features", [])[:max_features]
    subnodes = result.diagnostics.get("subnodes", [])[:max_features]
    alternatives = alternative_rows(result)[:max_features]
    scoring = result.diagnostics.get("scoring_context", {})
    rows = [
        "context:",
        f"  scoring: {scoring.get('scoring', 'configuration-based (best config AUC = evoforest score)')}",
        f"  best_config_auc: {float(scoring.get('best_config_auc', result.auc)):.6f}",
        f"  global_ridge_auc: {float(scoring.get('global_ridge_auc', 0.0)):.6f}",
        f"  config_auc_range: {scoring.get('config_auc_range', [result.auc, result.auc])}",
        f"  fold_auc_std: {float(scoring.get('fold_auc_std', 0.0)):.6f}",
        f"  effective_rank: {float(scoring.get('effective_rank', 0.0)):.4f}",
        f"  mean_max_corr: {float(scoring.get('mean_max_corr', 0.0)):.4f}",
        f"  shap_reconstruction_error: {float(scoring.get('shap_reconstruction_error', 0.0)):.8f}",
        f"  n_features_global: {int(scoring.get('n_features_global', len(result.feature_names)))}",
        f"  n_features_best_config: {int(scoring.get('n_features_best_config', len(result.feature_names)))}",
        f"  n_configs: {int(scoring.get('n_configs', 1))}",
        f"  n_configs_total: {int(scoring.get('n_configs_total', 1))}",
        "scoring:",
        f"  auc: {result.auc:.6f}",
        f"  config: {result.config}",
        f"  search: {result.diagnostics.get('configuration_search', {})}",
        f"  fitting: {result.diagnostics.get('fitting', {})}",
        f"  global_ridge: {result.diagnostics.get('global_ridge', {})}",
        "features[name,depth,imp,auc,sign,max_corr,n_hi_corr,most_corr,res,res2,rank,effect,stab,shap,cv_shap]:",
    ]
    for feature in features:
        rows.append(
            "  "
            + ",".join(
                [
                    str(feature.get("name", "")),
                    str(feature.get("depth", 0)),
                    f"{float(feature.get('importance', 0.0)):.4f}",
                    f"{float(feature.get('individual_auc', 0.0)):.4f}",
                    str(feature.get("coef_sign", 0)),
                    f"{float(feature.get('max_corr', feature.get('redundancy', 0.0))):.4f}",
                    str(feature.get("n_high_corr", 0)),
                    str(feature.get("most_correlated", "")),
                    f"{float(feature.get('residual_corr', 0.0)):.4f}",
                    f"{float(feature.get('residual_quadratic_corr', 0.0)):.4f}",
                    f"{float(feature.get('rank_corr', 0.0)):.4f}",
                    f"{float(feature.get('class_effect', 0.0)):.4f}",
                    f"{float(feature.get('weight_stability', 0.0)):.4f}",
                    f"{float(feature.get('shap_importance', 0.0)):.4f}",
                    f"{float(feature.get('cv_shap_mean_abs', 0.0)):.4f}",
                ]
            )
        )
    rows.append("subnodes[name,n,imp,shap,max_auc,abs_shap,res_abs,red,stab]:")
    for subnode in subnodes:
        rows.append(
            "  "
            + ",".join(
                [
                    str(subnode.get("name", "")),
                    str(subnode.get("feature_count", 0)),
                    f"{float(subnode.get('importance', 0.0)):.4f}",
                    f"{float(subnode.get('shap_importance', 0.0)):.4f}",
                    f"{float(subnode.get('max_feature_auc', 0.0)):.4f}",
                    f"{float(subnode.get('mean_abs_shap', 0.0)):.4f}",
                    f"{float(subnode.get('mean_abs_residual_corr', 0.0)):.4f}",
                    f"{float(subnode.get('mean_redundancy', 0.0)):.4f}",
                    f"{float(subnode.get('mean_weight_stability', 0.0)):.4f}",
                ]
            )
        )
    rows.append("alternatives[name,age,evals,sel,n,imp,shap,max_auc,abs_shap,res_abs,red,stab,last_auc]:")
    for alternative in alternatives:
        rows.append(
            "  "
            + ",".join(
                [
                    str(alternative.get("name", "")),
                    str(alternative.get("age", 0)),
                    str(alternative.get("participation_count", alternative.get("eval_count", 0))),
                    str(alternative.get("selected_count", 0)),
                    str(alternative.get("last_feature_count", alternative.get("feature_count", 0))),
                    f"{float(alternative.get('mean_importance', alternative.get('importance', 0.0))):.4f}",
                    f"{float(alternative.get('mean_shap_importance', alternative.get('shap_importance', 0.0))):.4f}",
                    f"{float(alternative.get('max_feature_auc', 0.0)):.4f}",
                    f"{float(alternative.get('mean_abs_shap', 0.0)):.4f}",
                    f"{float(alternative.get('mean_abs_residual_corr', 0.0)):.4f}",
                    f"{float(alternative.get('mean_redundancy', 0.0)):.4f}",
                    f"{float(alternative.get('mean_weight_stability', 0.0)):.4f}",
                    f"{float(alternative.get('last_config_auc', alternative.get('config_auc', result.auc))):.6f}",
                ]
            )
        )
    return "\n".join(rows)


def alternative_rows(result: EvaluationResult) -> list[dict[str, object]]:
    stats = result.diagnostics.get("alternative_stats", [])
    if isinstance(stats, list) and any(isinstance(row, dict) and "participation_count" in row for row in stats):
        return [row for row in stats if isinstance(row, dict)]
    current = result.diagnostics.get("alternatives", [])
    if isinstance(current, list):
        return [row for row in current if isinstance(row, dict)]
    return []
