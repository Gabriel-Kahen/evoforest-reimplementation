from __future__ import annotations

import io
import json
import urllib.error

import numpy as np
import pytest

from evoforest_arch import llm as llm_module
from evoforest_arch.agents import EngineerAgent, ScientistAgent
from evoforest_arch.evaluator import RidgeEvaluator, abs_feature_correlation, effective_rank, feature_correlation_summary, max_correlation_scores, most_correlated_names, safe_corr_columns
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.feedback import toon_report
from evoforest_arch.graph import CallableFamily, EvalContext, FeatureBlock, Graph, ResidualWeightRule
from evoforest_arch.llm import ClaudeLLMClient, GeminiLLMClient, LLMEngineerAgent, LLMMemorandumAgent, LLMScientistAgent, OpenAILLMClient, PromptBuilder, StaticLLMClient, llm_client_from_env, llm_provider_from_env
from evoforest_arch.maintenance import GraphMaintenance
from evoforest_arch.metrics import FoldStrategy, RMSE_SCORER, safe_corr
from evoforest_arch.mutations import GlobalSpec, MutationDocument, MutationEngine, MutationSpec, NodeSpec, RemoveSpec
from evoforest_arch.rank_readout import fit_rank_feature_expansion, select_rank_ensemble
from evoforest_arch.readout import fit_ridge, select_alpha, select_alpha_and_fit_ridge
from evoforest_arch.seed import build_seed_graph, build_structural_break_seed_graph
from evoforest_arch.source import SourceExecutionError
from evoforest_arch.synthetic import make_structural_break_data, make_tabular_data
from evoforest_arch.task import InputSpec, TaskSchema


def test_default_seed_graph_is_task_independent_tabular() -> None:
    dataset = make_tabular_data(n_samples=80, n_features=8, seed=4)
    graph = build_seed_graph()
    assert graph.task_schema is not None
    assert graph.task_schema["kind"] == "tabular"
    assert "x" in graph.nodes
    assert "series" not in graph.nodes
    assert "boundary" not in graph.nodes
    assert graph.nodes["base_features"].kind == "intermediate"
    assert graph.nodes["output"].kind == "output"

    result = RidgeEvaluator(n_splits=3, seed=4, max_configurations=8).evaluate(graph, dataset.inputs(), dataset.y)

    assert result.score > 0.5
    assert result.diagnostics["features"]


def test_seed_graph_exposes_paper_architecture_motifs() -> None:
    graph = build_structural_break_seed_graph()
    assert graph.nodes["segment_stats"].kind == "intermediate"
    assert graph.nodes["activation"].kind == "callable"
    assert graph.nodes["output"].kind == "output"
    assert graph.nodes["ridge_w"].kind == "fitting"
    assert graph.nodes["ridge_g"].kind == "fitting"
    assert len(graph.nodes["segment_stats"].alternatives) >= 2
    assert len(graph.nodes["activation"].alternatives) >= 3
    assert len(graph.nodes["output"].alternatives) >= 3
    assert "gate_scale" in graph.globals.names()
    assert "projection_vector" in graph.globals.names()
    assert "residual_huber_scale" in graph.globals.names()
    assert "output" not in graph.configuration_space()
    assert "ridge_w" in graph.configuration_space()
    assert "ridge_g" in graph.configuration_space()


def test_graph_evaluates_alternatives_callables_and_globals() -> None:
    dataset = make_structural_break_data(n_series=40, length=80, seed=3)
    graph = build_structural_break_seed_graph()
    config = graph.default_config()
    config["activation"] = "sigmoid_gate"
    x, names, ctx = graph.evaluate_features(dataset.inputs(), config)
    assert x.shape[0] == 40
    assert len(names) == x.shape[1]
    assert any(name.startswith("output.activated") for name in names)
    assert any(name.startswith("output.projection") for name in names)
    callable_value = graph.evaluate_node("activation", config, ctx)
    assert isinstance(callable_value, CallableFamily)
    residual_rule = graph.evaluate_node("ridge_g", {**config, "ridge_g": "huber"}, ctx)
    assert isinstance(residual_rule, ResidualWeightRule)
    assert ctx.cache_hits >= 0
    assert ctx.cache_misses > 0


def test_alternative_cache_key_memo_preserves_key_format() -> None:
    graph = build_structural_break_seed_graph()
    config = graph.default_config()
    config["activation"] = "sigmoid_gate"
    selected = graph.selected_config(config)
    memo: dict[tuple[str, str], tuple[object, ...]] = {}

    uncached = [
        graph.alternative_cache_key(node_name, alternative.id, selected)
        for node_name in graph.output_nodes()
        for alternative in graph.nodes[node_name].alternatives
    ]
    cached = [
        graph.alternative_cache_key(node_name, alternative.id, selected, memo=memo)
        for node_name in graph.output_nodes()
        for alternative in graph.nodes[node_name].alternatives
    ]

    assert cached == uncached
    assert memo


def test_configuration_search_uses_ancestor_conditioned_shared_cache() -> None:
    dataset = make_structural_break_data(n_series=48, length=60, seed=9)
    graph = Graph("cache_test")
    graph.add_input("series", "Synthetic series.")
    graph.add_node("a", "intermediate", "Shared ancestor.")
    graph.add_node("b", "intermediate", "Child depending on selected ancestor.")
    graph.add_node("output", "output", "Scored output.")
    counts = {"a0": 0, "a1": 0, "b0": 0, "b1": 0, "out": 0}

    def a_fn(label: str, offset: float):
        def fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
            counts[label] += 1
            series = np.asarray(values["series"], dtype=np.float64)
            return FeatureBlock(series[:, :1] + offset, [label])

        return fn

    def b_fn(label: str, scale: float):
        def fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
            counts[label] += 1
            block = values["a"]
            assert isinstance(block, FeatureBlock)
            return FeatureBlock(block.values * scale, [label])

        return fn

    def out_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        counts["out"] += 1
        a_block = values["a"]
        b_block = values["b"]
        assert isinstance(a_block, FeatureBlock)
        assert isinstance(b_block, FeatureBlock)
        return FeatureBlock(np.column_stack([a_block.values[:, 0], b_block.values[:, 0]]), ["a_value", "b_value"])

    graph.add_alternative("a", "a0", ("series",), a_fn("a0", 0.0))
    graph.add_alternative("a", "a1", ("series",), a_fn("a1", 0.25))
    graph.add_alternative("b", "b0", ("a",), b_fn("b0", 1.0))
    graph.add_alternative("b", "b1", ("a",), b_fn("b1", -1.0))
    graph.add_alternative("output", "out", ("a", "b"), out_fn)

    result = RidgeEvaluator(n_splits=2, seed=9, max_configurations=4).evaluate(graph, dataset.inputs(), dataset.y)

    assert result.diagnostics["configuration_search"]["evaluated"] == 4
    cache = result.diagnostics["configuration_search"]["cache"]
    assert cache["shared_across_configurations"] is True
    assert cache["key"] == "ancestor_conditioned_subpath"
    assert cache["hits"] > 0
    assert counts["a0"] == 1
    assert counts["a1"] == 1
    assert counts["b0"] == 2
    assert counts["b1"] == 2
    assert counts["out"] == 4


def test_ridge_evaluator_scores_synthetic_breaks() -> None:
    dataset = make_structural_break_data(n_series=120, length=100, seed=11)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=11).evaluate(graph, dataset.inputs(), dataset.y)
    assert result.score > 0.8
    assert result.diagnostics["features"]
    assert result.diagnostics["graph"]["alternatives"] >= 14
    assert result.diagnostics["configuration_search"]["evaluated"] >= 12
    search = result.diagnostics["configuration_search"]
    assert search["cache"]["shared_across_configurations"] is True
    assert search["cache"]["key"] == "ancestor_conditioned_subpath"
    assert search["score_range"][0] <= result.score <= search["score_range"][1]
    assert search["best_config_score"] == result.score
    scoring = result.diagnostics["scoring_context"]
    assert scoring["best_config_score"] == result.score
    assert scoring["global_ridge_score"] > 0.8
    assert scoring["shap_reconstruction_error"] < 1e-8
    assert scoring["n_configs"] == search["evaluated"]
    assert scoring["n_features_best_config"] == len(result.feature_names)
    assert result.diagnostics["global_ridge"]["score"] == scoring["global_ridge_score"]
    assert result.diagnostics["linear_shap"]["global_reconstruction_error"] < 1e-8
    assert result.diagnostics["linear_shap"]["cv_reconstruction_error"] < 1e-8
    assert result.diagnostics["folds"]["score_std"] >= 0.0
    assert result.diagnostics["effective_rank"] > 0.0
    assert result.diagnostics["mean_max_corr"] >= 0.0
    assert result.diagnostics["fitting"]["ridge_w"]["alternative"] in {"uniform", "boundary_energy"}
    assert result.diagnostics["fitting"]["ridge_g"]["alternative"] in {"identity", "huber"}
    assert result.diagnostics["fitting"]["ridge_g"]["irls_steps_requested"] == 2
    assert "irls_steps_used_per_fold" in result.diagnostics["fitting"]["ridge_g"]
    first_feature = result.diagnostics["features"][0]
    assert {
        "dependencies",
        "depth",
        "max_corr",
        "n_high_corr",
        "most_correlated",
        "redundancy",
        "residual_quadratic_corr",
        "weight_stability",
        "target_alignment",
        "target_corr",
        "target_quadratic_corr",
        "global_coef",
        "shap_mean_abs",
        "shap_importance",
        "cv_shap_mean_abs",
    } <= set(first_feature)
    assert result.diagnostics["subnodes"]
    assert all("." in row["name"] for row in result.diagnostics["subnodes"])
    assert "shap_importance" in result.diagnostics["subnodes"][0]
    assert result.diagnostics["alternatives"]
    assert any(row["name"] == "output.raw_concat" for row in result.diagnostics["alternatives"])
    assert result.diagnostics["alternative_stats"]
    assert result.diagnostics["graph"]["selected_alternatives"]["activation"] in {"identity", "sigmoid_gate", "clipped_linear"}
    toon = toon_report(result)
    assert "context:" in toon
    assert "global_ridge_score:" in toon
    assert "effective_rank:" in toon
    assert "features[name,depth,imp,align,sign,max_corr" in toon
    assert "shap" in toon
    assert "alternatives[name,age,evals,sel,n,imp,shap" in toon


def test_lower_is_better_scorer_preserves_raw_metric_direction() -> None:
    y = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.0, 2.0, 4.0])
    raw = RMSE_SCORER.raw_score(y, pred)

    assert raw > 0.0
    assert RMSE_SCORER.score(y, pred) == -raw
    scorer_dict = RMSE_SCORER.to_dict()
    assert scorer_dict["raw_metric"] == "rmse"
    assert scorer_dict["raw_higher_is_better"] is False
    assert scorer_dict["higher_is_better"] is True
    assert scorer_dict["optimization_score"] == "negative_raw"

    dataset = make_tabular_data(n_samples=60, n_features=5, seed=36)
    result = RidgeEvaluator(n_splits=3, seed=36, max_configurations=4, scorer="rmse").evaluate(
        build_seed_graph(),
        dataset.inputs(),
        dataset.y,
    )

    assert result.score <= 0.0
    assert result.diagnostics["folds"]["raw_metric"] == "rmse"
    assert result.diagnostics["folds"]["raw_higher_is_better"] is False
    assert result.diagnostics["folds"]["raw_score_mean"] >= 0.0
    assert result.diagnostics["scoring_context"]["scorer"]["optimization_score"] == "negative_raw"


def test_feature_pool_diagnostics_can_be_disabled() -> None:
    dataset = make_tabular_data(n_samples=60, n_features=5, seed=39)
    result = RidgeEvaluator(
        n_splits=3,
        seed=39,
        max_configurations=4,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    ).evaluate(build_seed_graph(), dataset.inputs(), dataset.y)

    assert result.diagnostics["valid_feature_pool"]["enabled"] is False
    assert result.diagnostics["valid_feature_pool"]["reason"] == "feature_pool_diagnostics=false"
    assert result.diagnostics["configuration_search"]["evaluated"] == 4


def test_feature_matrix_retention_can_be_disabled() -> None:
    dataset = make_tabular_data(n_samples=60, n_features=5, seed=47)
    retained = RidgeEvaluator(
        n_splits=2,
        seed=47,
        max_configurations=2,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    ).evaluate(build_seed_graph(), dataset.inputs(), dataset.y)
    released = RidgeEvaluator(
        n_splits=2,
        seed=47,
        max_configurations=2,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
        retain_feature_matrix=False,
    ).evaluate(build_seed_graph(), dataset.inputs(), dataset.y)

    assert retained.feature_matrix is not None
    assert released.feature_matrix is None
    assert released.diagnostics["configuration_search"]["evaluated"] == 2


def test_combined_alpha_selection_matches_public_ridge_helpers() -> None:
    rng = np.random.default_rng(44)
    x = rng.normal(size=(80, 12))
    y = rng.normal(size=80)
    weights = np.linspace(0.5, 2.0, y.shape[0])
    for sample_weight in (None, weights):
        alpha = select_alpha(x, y, sample_weight=sample_weight)
        model = fit_ridge(x, y, alpha, sample_weight=sample_weight)
        combined_alpha, combined_model = select_alpha_and_fit_ridge(x, y, sample_weight=sample_weight)

        assert combined_alpha == alpha
        np.testing.assert_allclose(combined_model.coef, model.coef)
        np.testing.assert_allclose(combined_model.predict(x), model.predict(x))


def test_configuration_search_reuses_adjacent_identical_feature_matrices() -> None:
    dataset = make_tabular_data(n_samples=64, n_features=5, seed=45)
    result = RidgeEvaluator(
        n_splits=2,
        seed=45,
        max_configurations=64,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    ).evaluate(build_seed_graph(), dataset.inputs(), dataset.y)

    cache = result.diagnostics["configuration_search"]["cache"]
    assert cache["feature_matrix_reuse_hits"] > 0
    assert result.diagnostics["configuration_search"]["evaluated"] > cache["feature_matrix_reuse_hits"]


def test_configuration_search_reuses_fold_assignments() -> None:
    class CountingFoldStrategy(FoldStrategy):
        def __init__(self) -> None:
            self.calls = 0

        def split(self, inputs: dict[str, object], y: np.ndarray, n_splits: int, seed: int):
            self.calls += 1
            return super().split(inputs, y, n_splits, seed)

    dataset = make_tabular_data(n_samples=64, n_features=5, seed=46)
    fold_strategy = CountingFoldStrategy()
    result = RidgeEvaluator(
        n_splits=2,
        seed=46,
        max_configurations=4,
        fold_strategy=fold_strategy,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    ).evaluate(build_seed_graph(), dataset.inputs(), dataset.y)

    assert result.diagnostics["configuration_search"]["evaluated"] == 4
    assert fold_strategy.calls == 1


def test_blockwise_correlation_summary_matches_full_matrix_helper() -> None:
    rng = np.random.default_rng(48)
    x = rng.normal(size=(80, 11))
    x[:, 4] = x[:, 3] * 0.98 + rng.normal(scale=0.01, size=80)
    x[:, 7] = 1.0
    names = [f"feature_{idx}" for idx in range(x.shape[1])]

    corr = abs_feature_correlation(x)
    summary = feature_correlation_summary(x, names, block_size=3)

    np.testing.assert_allclose(summary.max_corr, max_correlation_scores(corr))
    np.testing.assert_array_equal(summary.high_corr_counts, np.sum(corr > 0.9, axis=1))
    assert summary.most_correlated == most_correlated_names(corr, names)
    np.testing.assert_allclose(summary.effective_rank, effective_rank(x))


def test_vectorized_column_correlations_match_scalar_safe_corr() -> None:
    rng = np.random.default_rng(49)
    x = rng.normal(size=(72, 9))
    x[:, 2] = 3.0
    x[0, 5] = np.nan
    y = rng.normal(size=72)
    y[1] = np.nan

    corr = safe_corr_columns(x, y, block_size=2)
    square_corr = safe_corr_columns(x, y, square=True, block_size=2)

    expected = np.asarray([safe_corr(x[:, idx], y) for idx in range(x.shape[1])])
    expected_square = np.asarray([safe_corr(x[:, idx] * x[:, idx], y) for idx in range(x.shape[1])])
    np.testing.assert_allclose(corr, expected)
    np.testing.assert_allclose(square_corr, expected_square)


def test_task_aware_fold_strategies_report_group_and_time_diagnostics() -> None:
    dataset = make_tabular_data(n_samples=72, n_features=6, seed=37)
    inputs = dataset.inputs()
    inputs["engine_id"] = np.asarray([f"engine_{index // 4}" for index in range(dataset.y.shape[0])])
    inputs["cycle"] = np.arange(dataset.y.shape[0])

    result = RidgeEvaluator(
        n_splits=3,
        seed=37,
        max_configurations=4,
        fold_strategy="group_random",
        group_key="engine_id",
    ).evaluate(build_seed_graph(), inputs, dataset.y)

    folds = result.diagnostics["folds"]
    assert folds["method"] == "group_random"
    assert folds["grouped"] is True
    assert folds["fold_group_overlap_count"] == 0
    objective = result.diagnostics["objective"]
    assert objective["group_key"] == "engine_id"
    assert objective["groups"]
    assert objective["target_bins"]

    time_folds, time_diagnostics = FoldStrategy(name="time_blocked", time_key="cycle").split(inputs, dataset.y, 4, seed=37)
    assert len(time_folds) == 4
    assert time_diagnostics["method"] == "time_blocked"
    assert time_diagnostics["blocked"] is True

    leave_folds, leave_diagnostics = FoldStrategy(name="leave_group_out", group_key="engine_id").split(inputs, dataset.y, 3, seed=37)
    assert len(leave_folds) == len(np.unique(inputs["engine_id"]))
    assert leave_diagnostics["method"] == "leave_group_out"
    assert leave_diagnostics["validation_group_counts"] == [1] * len(leave_folds)
    for train_idx, val_idx in leave_folds:
        assert len(set(inputs["engine_id"][train_idx].tolist()) & set(inputs["engine_id"][val_idx].tolist())) == 0


def test_task_schema_roles_drive_evaluator_folds_and_objective_diagnostics() -> None:
    dataset = make_tabular_data(n_samples=64, n_features=5, seed=40)
    schema = TaskSchema(
        name="role-aware-tabular",
        kind="tabular",
        inputs=(
            InputSpec("features", "numeric_matrix", "Feature matrix.", ("n_samples", "n_features"), ("feature",)),
            InputSpec("cycle", "time_index", "Cycle index.", ("n_samples",), ("time", "sequence")),
            InputSpec("regime", "regime_id", "Regime label.", ("n_samples",), ("regime",)),
        ),
        default_input="features",
    )
    inputs = {
        "features": dataset.inputs()["x"],
        "cycle": np.arange(dataset.y.shape[0]),
        "regime": np.asarray(["early" if index < 32 else "late" for index in range(dataset.y.shape[0])]),
    }
    graph = build_seed_graph(schema)
    result = RidgeEvaluator(n_splits=4, seed=40, max_configurations=4, task_schema=schema).evaluate(graph, inputs, dataset.y)

    assert result.diagnostics["folds"]["method"] == "time_blocked"
    objective = result.diagnostics["objective"]
    assert objective["time_key"] == "cycle"
    assert objective["time_bins"]
    assert objective["schema_roles"]["time"] == ["cycle"]
    assert "group" not in objective["schema_roles"]
    assert objective["role_groups"]["regime"]["key"] == "regime"


def test_ridge_evaluator_basic_diagnostics_skips_feature_rows() -> None:
    dataset = make_structural_break_data(n_series=80, length=80, seed=10)
    graph = build_structural_break_seed_graph()

    result = RidgeEvaluator(n_splits=2, seed=10, max_configurations=4, diagnostics_mode="basic").evaluate(
        graph,
        dataset.inputs(),
        dataset.y,
    )

    assert result.score > 0.6
    assert result.diagnostics["diagnostics_mode"] == "basic"
    assert result.diagnostics["features"] == []
    assert result.diagnostics["subnodes"] == []
    assert result.diagnostics["configuration_search"]["evaluated"] == 4


def test_ridge_g_runs_iterative_reweighted_least_squares() -> None:
    dataset = make_structural_break_data(n_series=84, length=90, seed=16)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=16, max_configurations=4, irls_steps=3).evaluate(
        graph,
        dataset.inputs(),
        dataset.y,
        config={"ridge_g": "huber"},
    )

    ridge_g = result.diagnostics["fitting"]["ridge_g"]
    assert ridge_g["alternative"] == "huber"
    assert ridge_g["rule"] == "huber"
    assert ridge_g["irls_steps_requested"] == 3
    assert ridge_g["irls_steps_used_per_fold"] == [3, 3, 3]
    assert all(len(row["iterations"]) == 3 for row in ridge_g["irls"])
    assert all("weight_min" in iteration for row in ridge_g["irls"] for iteration in row["iterations"])
    assert result.diagnostics["global_ridge"]["residual_reweighted"] is True
    assert result.diagnostics["global_ridge"]["irls_steps_used"] == 3
    assert len(result.diagnostics["global_ridge"]["irls_iterations"]) == 3


def test_rank_interaction_readout_selects_oof_features() -> None:
    dataset = make_structural_break_data(n_series=96, length=80, seed=31)
    graph = build_structural_break_seed_graph()
    config = graph.default_config()
    x, names, _ctx = graph.evaluate_features(dataset.inputs(), config=config)
    expansion = fit_rank_feature_expansion(x, dataset.y, names, max_interaction_base=6)
    expanded = expansion.transform(x)
    groups = np.arange(dataset.y.shape[0])

    selection = select_rank_ensemble(expanded, dataset.y, groups, n_splits=3, seed=31)
    predictions = selection.model.predict(expanded)

    assert expanded.shape[1] > x.shape[1]
    assert selection.oof_score > 0.4
    assert predictions.shape == dataset.y.shape
    assert selection.model.feature_indices.size > 0


def test_update_graph_persists_alternative_statistics_and_age() -> None:
    dataset = make_structural_break_data(n_series=90, length=90, seed=12)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=12, max_configurations=8).evaluate(
        graph,
        dataset.inputs(),
        dataset.y,
        update_graph=True,
    )
    selected_segment = result.config["segment_stats"]
    selected = next(alternative for alternative in graph.nodes["segment_stats"].alternatives if alternative.id == selected_segment)
    assert selected.age == 1
    assert selected.stats["participation_count"] == 1
    assert selected.stats["selected_count"] == 1
    assert selected.stats["last_config_score"] == result.score
    assert selected.stats["last_feature_count"] > 0
    assert selected.stats["mean_shap_importance"] > 0.0
    assert selected.stats["mean_abs_shap"] > 0.0
    assert graph.nodes["output"].alternatives[0].age == 1
    assert graph.nodes["output"].alternatives[0].stats["output_count"] == 1
    serialized = graph.to_dict()
    first_alternative = next(node["alternatives"][0] for node in serialized["nodes"] if node["alternatives"])
    assert "age" in first_alternative
    assert "stats" in first_alternative
    snapshot = result.diagnostics["alternative_stats"]
    row = next(item for item in snapshot if item["name"] == f"segment_stats.{selected_segment}")
    assert row["age"] == 1
    assert row["participation_count"] == 1
    assert row["best_config_score"] == result.score


def test_scientist_and_engineer_generate_diagnostic_mutation_documents() -> None:
    dataset = make_structural_break_data(n_series=80, length=90, seed=15)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=15, max_configurations=8).evaluate(graph, dataset.inputs(), dataset.y)
    hypotheses = ScientistAgent().generate(graph, result)
    document = EngineerAgent().synthesize(graph, result, hypotheses, step=7, island=2, rng=np.random.default_rng(15))
    assert hypotheses
    assert document.add
    assert document.add[0].alternative_id.endswith("_i2_7")
    assert any("Expected:" in item for item in document.hypotheses)


def test_mutation_engine_adds_alternative_without_mutating_parent_graph() -> None:
    graph = build_structural_break_seed_graph()
    before = len(graph.nodes["shape_stats"].alternatives)
    spec = MutationSpec(
        kind="add_alternative",
        target_node="shape_stats",
        primitive="spectral_basic",
        alternative_id="spectral_extra",
        parents=("series",),
        description="Additional spectral probe with distinct lineage.",
    )
    mutated = MutationEngine().apply(graph, spec)
    assert len(graph.nodes["shape_stats"].alternatives) == before
    assert len(mutated.nodes["shape_stats"].alternatives) == before + 1
    assert "spectral_extra" in mutated.configuration_space()["shape_stats"]
    output_spec = MutationSpec(
        kind="add_alternative",
        target_node="output",
        primitive="projection_outputs",
        alternative_id="projection_extra",
        parents=("segment_stats", "trend_stats", "shape_stats"),
        description="Additional projection output lineage.",
    )
    output_mutated = MutationEngine().apply(graph, output_spec)
    assert "output" not in output_mutated.configuration_space()
    assert len(output_mutated.nodes["output"].alternatives) == len(graph.nodes["output"].alternatives) + 1


def test_mutation_document_round_trip_and_maintenance() -> None:
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        hypotheses=("Remove weak duplicate and add a distinct robust segment alternative.",),
        rationale="Exercise paper-style YAML mutation documents.",
        globals=(GlobalSpec("roundtrip_unused_global", [0.25], True, "Unused parser coverage global."),),
        remove=(RemoveSpec("segment_stats", "robust", "test removal"),),
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="segment_stats",
                primitive="segment_robust",
                alternative_id="robust_replacement",
                parents=("series",),
                description="Replacement robust segment lineage.",
            ),
        ),
    )
    parsed = MutationDocument.from_yaml(document.to_yaml())
    assert parsed.to_dict() == document.to_dict()
    applied = MutationEngine().apply_document(graph, parsed)
    assert "robust" not in applied.graph.configuration_space()["segment_stats"]
    assert "robust_replacement" in applied.graph.configuration_space()["segment_stats"]
    assert applied.maintenance.to_dict()["removed_alternatives"] == ["segment_stats.robust"]
    assert "roundtrip_unused_global" in applied.maintenance.to_dict()["removed_globals"]


def test_mutation_document_can_introduce_reachable_nodes() -> None:
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        hypotheses=("Introduce a reusable local_stats node and expose it through output.",),
        rationale="Exercise node-level graph mutation.",
        nodes=(NodeSpec("local_stats", "intermediate", "Additional reusable segment statistics."),),
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="local_stats",
                primitive="segment_basic",
                alternative_id="basic_local",
                parents=("series",),
                description="Local segment statistics on the raw series.",
            ),
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="pass_outputs",
                alternative_id="local_stats_output",
                parents=("local_stats",),
                description="Expose the new reusable node as output features.",
            ),
        ),
    )
    parsed = MutationDocument.from_yaml(document.to_yaml())
    applied = MutationEngine().apply_document(graph, parsed)
    assert "local_stats" in applied.graph.nodes
    assert "basic_local" in applied.graph.configuration_space()["local_stats"]
    assert any(alternative.id == "local_stats_output" for alternative in applied.graph.nodes["output"].alternatives)


def test_sandboxed_source_mutation_adds_lambda_alternative() -> None:
    dataset = make_structural_break_data(n_series=30, length=70, seed=29)
    graph = build_structural_break_seed_graph()
    source_payload = {
        "kind": "add_alternative",
        "target_node": "output",
        "alternative_id": "source_squared_delta",
        "parents": ["segment_stats"],
        "source": "lambda ctx, values: FeatureBlock(np.column_stack([values['segment_stats'].values[:, 0] ** 2]), ['squared_delta'])",
        "description": "Paper-style source lambda output feature.",
    }
    text = "\n".join(
        [
            'rationale: "exercise sandboxed source-backed alternatives"',
            "hypotheses:",
            "  []",
            "nodes:",
            "  []",
            "remove:",
            "  []",
            "globals:",
            "  []",
            "add:",
            f"  - {json.dumps(source_payload, sort_keys=True)}",
        ]
    )
    document = MutationDocument.from_yaml(text)
    assert document.add[0].primitive == "source"
    with pytest.raises(ValueError, match="allow_source"):
        MutationEngine().apply_document(graph, document)
    applied = MutationEngine(allow_source=True).apply_document(graph, document)
    assert applied.graph.nodes["output"].alternatives[-1].source.startswith("lambda ctx, values:")
    x, names, _ctx = applied.graph.evaluate_features(dataset.inputs(), applied.graph.default_config())
    assert x.shape[0] == dataset.y.shape[0]
    assert any(name.endswith("source_squared_delta.squared_delta") for name in names)


def test_paper_style_lambda_mutation_yaml_is_supported() -> None:
    document = MutationDocument.from_yaml(
        "\n".join(
            [
                "remove:",
                "  - output.raw_concat",
                "add:",
                "  output:",
                "    - \"lambda ctx, values: FeatureBlock(values['segment_stats'].values[:, :1] * ctx.globals.get('gate_scale')[0], ['first_value'])\"",
            ]
        )
    )

    assert document.remove[0].target_node == "output"
    assert document.remove[0].alternative_id == "raw_concat"
    assert document.add[0].target_node == "output"
    assert document.add[0].primitive == "source"
    assert document.add[0].source.startswith("lambda ctx, values:")
    assert document.add[0].parents == ("segment_stats",)
    assert document.add[0].global_refs == ("gate_scale",)


def test_paper_style_source_object_preserves_contracts_and_node_kind() -> None:
    document = MutationDocument.from_yaml(
        "\n".join(
            [
                "add:",
                "  output:",
                '    - {"source": "lambda ctx, values: FeatureBlock(values[\'segment_stats\'].values[:, :1], [\'first_value\'])", "parents": ["segment_stats"], "node_kind": "output", "output_contract": {"type": "feature_block", "n_columns": 1}, "torch_source": "lambda ctx, values: values[\'segment_stats\'][:, :1]"}',
            ]
        )
    )

    spec = document.add[0]
    assert spec.parents == ("segment_stats",)
    assert spec.node_kind == "output"
    assert spec.output_contract == {"type": "feature_block", "n_columns": 1}
    assert spec.torch_source.startswith("lambda ctx, values:")


def test_source_output_contract_rejects_wrong_shape() -> None:
    dataset = make_structural_break_data(n_series=24, length=70, seed=34)
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="source",
                alternative_id="wrong_contract_source",
                parents=("segment_stats",),
                source="lambda ctx, values: FeatureBlock(values['segment_stats'].values[:, :1], ['one_col'])",
                output_contract={"type": "feature_block", "n_columns": 2},
            ),
        )
    )
    applied = MutationEngine(allow_source=True).apply_document(graph, document).graph

    with pytest.raises(SourceExecutionError, match="columns"):
        applied.evaluate_features(dataset.inputs(), applied.default_config())


def test_source_lambda_defaults_are_rejected_before_execution() -> None:
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="source",
                alternative_id="default_arg_source",
                parents=("segment_stats",),
                source="lambda ctx, values, expensive=np.ones((10**8,)): FeatureBlock(values['segment_stats'].values[:, :1], ['x'])",
            ),
        )
    )

    with pytest.raises(ValueError, match="no defaults"):
        MutationEngine(allow_source=True).apply_document(graph, document)


def test_maintenance_keeps_source_alternatives_with_different_contracts() -> None:
    graph = build_structural_break_seed_graph()
    base_source = "lambda ctx, values: FeatureBlock(values['segment_stats'].values[:, :1], ['x'])"
    document = MutationDocument(
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="source",
                alternative_id="source_contract_one",
                parents=("segment_stats",),
                source=base_source,
                description="Same description.",
                output_contract={"type": "feature_block", "n_columns": 1},
            ),
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="source",
                alternative_id="source_contract_range",
                parents=("segment_stats",),
                source=base_source,
                description="Same description.",
                output_contract={"type": "feature_block", "min_columns": 1},
            ),
        )
    )
    applied = MutationEngine(allow_source=True).apply_document(graph, document)
    output_ids = {alternative.id for alternative in applied.graph.nodes["output"].alternatives}

    assert {"source_contract_one", "source_contract_range"} <= output_ids


def test_paper_style_lambda_mutation_ids_are_stable_across_documents() -> None:
    first = MutationDocument.from_yaml(
        "\n".join(
            [
                "add:",
                "  output:",
                "    - \"lambda ctx, values: FeatureBlock(ctx.read_input('series')[:, :1], ['first_value'])\"",
            ]
        )
    )
    second = MutationDocument.from_yaml(
        "\n".join(
            [
                "add:",
                "  output:",
                "    - \"lambda ctx, values: FeatureBlock(ctx.read_input('series')[:, 1:2], ['second_value'])\"",
            ]
        )
    )

    assert first.add[0].alternative_id != second.add[0].alternative_id
    engine = MutationEngine(allow_source=True)
    graph = engine.apply_document(build_structural_break_seed_graph(), first).graph
    graph = engine.apply_document(graph, second).graph
    output_ids = {alternative.id for alternative in graph.nodes["output"].alternatives}
    assert first.add[0].alternative_id in output_ids
    assert second.add[0].alternative_id in output_ids


def test_maintenance_prunes_unreachable_nodes_and_unused_globals() -> None:
    graph = build_structural_break_seed_graph()
    graph.add_node("dead_branch", "intermediate", "Unreachable branch.")
    graph.globals.add("unused", [1.0], trainable=True, description="Unused test parameter.")
    cleaned, report = GraphMaintenance().clean(graph)
    assert "dead_branch" not in cleaned.nodes
    assert "unused" not in cleaned.globals.names()
    assert "dead_branch" in report.removed_nodes
    assert "unused" in report.removed_globals


def test_global_refinement_phase_runs_without_mutating_parent_graph() -> None:
    dataset = make_structural_break_data(n_series=50, length=80, seed=13)
    graph = build_structural_break_seed_graph()
    before = graph.globals.get("gate_scale").copy()
    result = RidgeEvaluator(n_splits=3, seed=13, max_configurations=4, refine_globals=True, refine_steps=2).evaluate(
        graph,
        dataset.inputs(),
        dataset.y,
    )
    assert result.diagnostics["refinement"]["requested_backend"] == "auto"
    assert result.diagnostics["refinement"]["backend"] == "torch_l_bfgs"
    assert "fallback_reason" in result.diagnostics["refinement"]
    assert result.diagnostics["configuration_search"]["evaluated"] == 4
    assert np.array_equal(graph.globals.get("gate_scale"), before)


def test_numpy_refinement_backend_can_be_requested_explicitly() -> None:
    dataset = make_structural_break_data(n_series=40, length=70, seed=17)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(
        n_splits=3,
        seed=17,
        max_configurations=3,
        refine_globals=True,
        refine_steps=1,
        refine_backend="numpy",
    ).evaluate(graph, dataset.inputs(), dataset.y)
    assert result.diagnostics["refinement"]["backend"] == "numpy_coordinate"
    assert result.diagnostics["refinement"]["requested_backend"] == "numpy"


def test_torch_l_bfgs_refinement_backend_when_available() -> None:
    pytest.importorskip("torch")
    dataset = make_structural_break_data(n_series=32, length=70, seed=19)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(
        n_splits=3,
        seed=19,
        max_configurations=3,
        refine_globals=True,
        refine_steps=2,
        refine_backend="torch",
    ).evaluate(graph, dataset.inputs(), dataset.y)
    assert result.diagnostics["refinement"]["backend"] == "torch_l_bfgs"
    assert result.diagnostics["refinement"]["enabled"] is True


def test_source_torch_path_participates_in_refinement_when_available() -> None:
    pytest.importorskip("torch")
    dataset = make_tabular_data(n_samples=36, n_features=6, seed=35)
    graph = build_seed_graph()
    document = MutationDocument(
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="source",
                alternative_id="scaled_source_feature",
                parents=("base_features",),
                source="lambda ctx, values: FeatureBlock(values['base_features'].values[:, :1] * ctx.globals.get('gate_scale')[0], ['scaled_raw_0'])",
                global_refs=("gate_scale",),
                node_kind="output",
                output_contract={"type": "feature_block", "n_columns": 1},
                torch_source="lambda ctx, values: values['base_features'][:, :1] * ctx.globals.get('gate_scale')[0]",
            ),
        )
    )
    graph = MutationEngine(allow_source=True).apply_document(graph, document).graph
    source_alt = next(alternative for alternative in graph.nodes["output"].alternatives if alternative.id == "scaled_source_feature")
    assert source_alt.torch_fn is not None

    result = RidgeEvaluator(
        n_splits=3,
        seed=35,
        max_configurations=4,
        refine_globals=True,
        refine_steps=1,
        refine_backend="torch",
    ).evaluate(graph, dataset.inputs(), dataset.y)

    assert result.diagnostics["refinement"]["backend"] == "torch_l_bfgs"
    assert "gate_scale" in result.diagnostics["refinement"]["trainable_globals"]


def test_sandboxed_source_can_evolve_callable_nodes() -> None:
    dataset = make_structural_break_data(n_series=28, length=70, seed=38)
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="activation",
                primitive="source",
                alternative_id="source_sigmoid_callable",
                parents=(),
                source="lambda ctx, values: {'kind': 'callable', 'name': 'source_sigmoid', 'op': 'sigmoid', 'scale': 1.5}",
                output_contract={"type": "callable"},
            ),
        )
    )

    graph = MutationEngine(allow_source=True).apply_document(graph, document).graph
    config = graph.default_config()
    config["activation"] = "source_sigmoid_callable"
    ctx = EvalContext(dataset.inputs(), graph.globals.clone())
    callable_family = graph.evaluate_node("activation", config, ctx)

    assert isinstance(callable_family, CallableFamily)
    transformed = callable_family.apply(np.array([-1.0, 0.0, 1.0]))
    assert transformed.shape == (3,)
    assert transformed[0] < transformed[1] < transformed[2]


def test_sandboxed_source_can_evolve_fitting_nodes() -> None:
    dataset = make_structural_break_data(n_series=60, length=70, seed=39)
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="ridge_g",
                primitive="source",
                alternative_id="source_huber_rule",
                parents=(),
                source="lambda ctx, values: {'kind': 'residual_weight_rule', 'name': 'source_huber', 'op': 'huber', 'scale': 1.0}",
                output_contract={"type": "residual_weight_rule"},
            ),
        )
    )

    graph = MutationEngine(allow_source=True).apply_document(graph, document).graph
    result = RidgeEvaluator(n_splits=3, seed=39, max_configurations=4, irls_steps=1).evaluate(
        graph,
        dataset.inputs(),
        dataset.y,
        config={"ridge_g": "source_huber_rule"},
    )

    assert result.diagnostics["fitting"]["ridge_g"]["alternative"] == "source_huber_rule"
    assert result.diagnostics["fitting"]["ridge_g"]["rule"] == "source_huber"
    assert result.diagnostics["global_ridge"]["residual_reweighted"] is True


def test_numpy_source_lambda_gets_auto_torch_path_when_expression_is_differentiable() -> None:
    graph = build_seed_graph()
    document = MutationDocument(
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="source",
                alternative_id="auto_torch_source_feature",
                parents=("base_features",),
                source="lambda ctx, values: FeatureBlock(np.column_stack([values['base_features'].values[:, 0] * ctx.globals.get('gate_scale')[0]]), ['scaled_first'])",
                global_refs=("gate_scale",),
                output_contract={"type": "feature_block", "n_columns": 1},
            ),
        )
    )

    graph = MutationEngine(allow_source=True).apply_document(graph, document).graph
    source_alt = next(alternative for alternative in graph.nodes["output"].alternatives if alternative.id == "auto_torch_source_feature")

    assert source_alt.torch_fn is not None
    assert source_alt.torch_source
    assert "torch.column_stack" in source_alt.torch_source
    assert source_alt.output_contract["differentiability_source"] == "auto"


def test_evolution_loop_writes_events_checkpoint_and_memorandum(tmp_path) -> None:
    dataset = make_structural_break_data(n_series=90, length=90, seed=21)
    graph = build_structural_break_seed_graph()
    result = EvolutionLoop(graph, evaluator=RidgeEvaluator(n_splits=3, seed=21, max_configurations=12), seed=21).run(
        dataset.inputs(),
        dataset.y,
        steps=4,
        output_dir=tmp_path,
    )
    assert result.score > 0.75
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 4
    first_event = json.loads(events[0])
    assert {"step", "accepted", "score", "best_score", "mutation", "config"} <= set(first_event)
    assert {"maintenance", "salvaged"} <= set(first_event)
    assert first_event["mutation"]["hypotheses"]
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert "graph" in checkpoint
    assert "feedback" in checkpoint
    assert "diagnostics_toon" in checkpoint
    assert checkpoint["feedback"]["top_subnodes"]
    archive_rows = [
        json.loads(line)
        for line in (tmp_path / "archive" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert archive_rows[0]["version"] == 0
    assert archive_rows[0]["step"] == 0
    assert (tmp_path / "archive" / archive_rows[0]["path"]).exists()
    archived = json.loads((tmp_path / "archive" / archive_rows[0]["path"]).read_text(encoding="utf-8"))
    assert archived["feedback"]["top_subnodes"]
    assert archived["diagnostics_toon"]
    assert (tmp_path / "mutations" / "step_0001.yaml").exists()
    assert "Expected:" in (tmp_path / "mutations" / "step_0001.yaml").read_text(encoding="utf-8")
    assert (tmp_path / "memorandum.md").exists()
    assert (tmp_path / "task_context.md").exists()
    memorandum = (tmp_path / "memorandum.md").read_text(encoding="utf-8")
    assert "[OUTCOME HISTORY]" in memorandum
    assert "[STATE]" in memorandum
    assert "[WHAT WORKS]" in memorandum
    assert "[WHAT FAILED]" in memorandum
    assert "[ERROR LOG]" in memorandum
    task_context = (tmp_path / "task_context.md").read_text(encoding="utf-8")
    assert "## Tensor Inventory" in task_context
    assert "series: numeric_tensor" in task_context
    assert "## Scorer Mechanics" in task_context
    assert "random 3-fold Ridge CV" in task_context


def test_llm_agents_write_prompt_artifacts_and_mutation_yaml(tmp_path) -> None:
    dataset = make_structural_break_data(n_series=50, length=70, seed=23)
    graph = build_structural_break_seed_graph()
    llm_document = MutationDocument(
        hypotheses=("Add a nonduplicate spectral shape alternative to cover residual frequency structure.",),
        rationale="LLM engineer selected a shape_stats mutation grounded in diagnostics.",
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="shape_stats",
                primitive="spectral_basic",
                alternative_id="spectral_llm",
                parents=("series",),
                description="LLM-proposed frequency-band ratio change statistics.",
            ),
        ),
    )
    client = StaticLLMClient(
        (
            "\n".join(
                [
                    "Hypothesis: Add a spectral shape_stats alternative.",
                    "Rationale: shape_stats appears in the selected output dependencies.",
                    "Expected Improvement: more complementary residual frequency coverage.",
                    "Risk Mode: Balanced.",
                ]
            ),
            f"```yaml\n{llm_document.to_yaml()}```",
        )
    )
    result = EvolutionLoop(
        graph,
        evaluator=RidgeEvaluator(n_splits=3, seed=23, max_configurations=6),
        scientist=LLMScientistAgent(client),
        engineer=LLMEngineerAgent(client),
        seed=23,
    ).run(
        dataset.inputs(),
        dataset.y,
        steps=1,
        output_dir=tmp_path,
    )
    assert result.score > 0.4
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert events[0]["mutation"]["add"][0]["alternative_id"] == "spectral_llm"
    prompt_files = sorted((tmp_path / "prompts").glob("step_0001_*.md"))
    assert len(prompt_files) == 2
    prompt_text = "\n".join(path.read_text(encoding="utf-8") for path in prompt_files)
    assert "GLOBAL EVOFOREST RULES" in prompt_text
    assert "FEATURE DIAGNOSTICS" in prompt_text
    assert "## Tensor Inventory" in prompt_text
    assert "## Scorer Mechanics" in prompt_text
    assert "spectral_llm" in (tmp_path / "mutations" / "step_0001.yaml").read_text(encoding="utf-8")
    assert "[OUTCOME HISTORY]" in (tmp_path / "memorandum.md").read_text(encoding="utf-8")
    assert "## Tensor Inventory" in (tmp_path / "task_context.md").read_text(encoding="utf-8")
    assert len(client.requests) == 2


def test_llm_scientist_retries_unparseable_hypotheses(monkeypatch) -> None:
    monkeypatch.setenv("EVOFOREST_LLM_PARSE_MAX_RETRIES", "1")
    dataset = make_structural_break_data(n_series=35, length=70, seed=24)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=24, max_configurations=4).evaluate(graph, dataset.inputs(), dataset.y)
    valid_response = "\n".join(
        [
            "Hypothesis: Add a spectral shape_stats alternative.",
            "Rationale: shape_stats appears in the selected output dependencies.",
            "Expected Improvement: more complementary residual frequency coverage.",
            "Risk Mode: Balanced.",
        ]
    )
    client = StaticLLMClient(("", valid_response))

    hypotheses = LLMScientistAgent(client).generate(graph, result, max_hypotheses=1, step=1)

    assert len(hypotheses) == 1
    assert len(client.requests) == 2
    assert "RETRY REQUIRED: INVALID STRUCTURED OUTPUT" in str(client.requests[1]["user_prompt"])
    assert "(empty response)" in str(client.requests[1]["user_prompt"])


def test_llm_engineer_retries_empty_mutation_document(monkeypatch) -> None:
    monkeypatch.setenv("EVOFOREST_LLM_PARSE_MAX_RETRIES", "1")
    dataset = make_structural_break_data(n_series=35, length=70, seed=25)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=25, max_configurations=4).evaluate(graph, dataset.inputs(), dataset.y)
    hypotheses = (ScientistAgent().generate(graph, result, max_hypotheses=1)[0],)
    empty_document = "rationale: \"empty\"\nhypotheses:\n  []\nnodes:\n  []\nremove:\n  []\nglobals:\n  []\nadd:\n  []\n"
    valid_document = MutationDocument(
        hypotheses=("Add a nonduplicate spectral shape alternative.",),
        rationale="LLM engineer selected a shape_stats mutation grounded in diagnostics.",
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="shape_stats",
                primitive="spectral_basic",
                alternative_id="spectral_retry",
                parents=("series",),
                description="Retry-produced frequency-band ratio change statistics.",
            ),
        ),
    )
    client = StaticLLMClient((empty_document, valid_document.to_yaml()))

    document = LLMEngineerAgent(client).synthesize(
        graph,
        result,
        hypotheses,
        step=1,
        island=None,
        rng=np.random.default_rng(25),
    )

    assert document.add[0].alternative_id == "spectral_retry"
    assert len(client.requests) == 2
    assert "empty mutation document" in str(client.requests[1]["user_prompt"])


def test_llm_scientist_failure_aborts_without_deterministic_fallback(tmp_path) -> None:
    dataset = make_structural_break_data(n_series=40, length=70, seed=27)
    graph = build_structural_break_seed_graph()
    client = StaticLLMClient(())

    with pytest.raises(RuntimeError, match="no remaining responses"):
        EvolutionLoop(
            graph,
            evaluator=RidgeEvaluator(n_splits=3, seed=27, max_configurations=4),
            scientist=LLMScientistAgent(client),
            engineer=LLMEngineerAgent(client),
            seed=27,
        ).run(dataset.inputs(), dataset.y, steps=1, output_dir=tmp_path)

    prompt_files = sorted((tmp_path / "prompts").glob("step_0001_*.md"))
    assert len(prompt_files) == 1
    prompt_text = prompt_files[0].read_text(encoding="utf-8")
    assert "## Error" in prompt_text
    assert "StaticLLMClient has no remaining responses" in prompt_text
    assert not (tmp_path / "mutations" / "step_0001.yaml").exists()


def test_llm_engineer_failure_aborts_without_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setenv("EVOFOREST_LLM_PARSE_MAX_RETRIES", "0")
    dataset = make_structural_break_data(n_series=35, length=70, seed=32)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=32, max_configurations=4).evaluate(graph, dataset.inputs(), dataset.y)
    hypotheses = (
        ScientistAgent().generate(graph, result, max_hypotheses=1)[0],
    )
    client = StaticLLMClient(("rationale: \"empty\"\nhypotheses:\n  []\nnodes:\n  []\nremove:\n  []\nglobals:\n  []\nadd:\n  []\n",))

    with pytest.raises(ValueError, match="empty mutation document"):
        LLMEngineerAgent(client).synthesize(graph, result, hypotheses, step=1, island=None, rng=np.random.default_rng(32))


def test_llm_memorandum_agent_retries_missing_sections(monkeypatch) -> None:
    monkeypatch.setenv("EVOFOREST_LLM_PARSE_MAX_RETRIES", "1")
    dataset = make_structural_break_data(n_series=35, length=70, seed=31)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=31, max_configurations=4).evaluate(graph, dataset.inputs(), dataset.y)
    response = "\n".join(
        [
            "[OUTCOME HISTORY]",
            "- ACCEPTED: seed state.",
            "[STATE]",
            "- Effective rank is tracked.",
            "[WHAT WORKS]",
            "- Compact source-backed changes.",
            "[WHAT FAILED]",
            "- No failures yet.",
            "[ERROR LOG]",
            "- No runtime errors.",
        ]
    )
    client = StaticLLMClient(("missing sections", response))

    memorandum = LLMMemorandumAgent(client).update(result, history=["- ACCEPTED: seed state."], step=1)

    assert "[STATE]" in memorandum
    assert len(client.requests) == 2
    assert "missing required sections" in str(client.requests[1]["user_prompt"])


def test_llm_memorandum_agent_requires_paper_sections() -> None:
    dataset = make_structural_break_data(n_series=35, length=70, seed=33)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=33, max_configurations=4).evaluate(graph, dataset.inputs(), dataset.y)
    response = "\n".join(
        [
            "[OUTCOME HISTORY]",
            "- ACCEPTED: seed state.",
            "[STATE]",
            "- Effective rank is tracked.",
            "[WHAT WORKS]",
            "- Compact source-backed changes.",
            "[WHAT FAILED]",
            "- No failures yet.",
            "[ERROR LOG]",
            "- No runtime errors.",
        ]
    )
    client = StaticLLMClient((response,))
    memorandum = LLMMemorandumAgent(client).update(result, history=["- ACCEPTED: seed state."], step=1)

    assert "[STATE]" in memorandum
    assert client.requests[0]["temperature"] == 0.0


def test_llm_provider_can_be_loaded_from_env_file(tmp_path, monkeypatch) -> None:
    for key in (
        "EVOFOREST_LLM_PROVIDER",
        "EVOFOREST_LLM_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "EVOFOREST_LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "EVOFOREST_LLM_PROVIDER=openai",
                "OPENAI_API_KEY='secret key'",
                "EVOFOREST_LLM_MODEL=gpt-test",
                "EVOFOREST_LLM_TIMEOUT_SECONDS=7.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert llm_provider_from_env(env_file, required=True) == "openai"
    client = OpenAILLMClient.from_env(env_file)
    assert client.api_key == "secret key"
    assert client.model == "gpt-test"
    assert client.timeout_seconds == 7.5
    assert isinstance(llm_client_from_env(env_file), OpenAILLMClient)


def test_claude_and_gemini_clients_load_from_env_files(tmp_path, monkeypatch) -> None:
    for key in (
        "EVOFOREST_LLM_PROVIDER",
        "EVOFOREST_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "EVOFOREST_LLM_MAX_TOKENS",
    ):
        monkeypatch.delenv(key, raising=False)

    claude_env = tmp_path / "claude.env"
    claude_env.write_text(
        "EVOFOREST_LLM_PROVIDER=claude\nANTHROPIC_API_KEY=anthropic-secret\nEVOFOREST_LLM_MODEL=claude-test\nEVOFOREST_LLM_MAX_TOKENS=1234\n",
        encoding="utf-8",
    )
    claude = llm_client_from_env(claude_env)
    assert isinstance(claude, ClaudeLLMClient)
    assert claude.api_key == "anthropic-secret"
    assert claude.model == "claude-test"
    assert claude.max_tokens == 1234

    monkeypatch.delenv("EVOFOREST_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EVOFOREST_LLM_MODEL", raising=False)
    monkeypatch.delenv("EVOFOREST_LLM_MAX_TOKENS", raising=False)
    gemini_env = tmp_path / "gemini.env"
    gemini_env.write_text(
        "EVOFOREST_LLM_PROVIDER=gemini\nGEMINI_API_KEY=gemini-secret\nEVOFOREST_LLM_MODEL=gemini-test\n",
        encoding="utf-8",
    )
    gemini = llm_client_from_env(gemini_env)
    assert isinstance(gemini, GeminiLLMClient)
    assert gemini.api_key == "gemini-secret"
    assert gemini.model == "gemini-test"


def test_llm_post_json_retries_transient_http_errors(monkeypatch) -> None:
    monkeypatch.setenv("EVOFOREST_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("EVOFOREST_LLM_RETRY_INITIAL_SECONDS", "0.25")
    monkeypatch.setenv("EVOFOREST_LLM_RETRY_MAX_SECONDS", "1.0")
    sleeps: list[float] = []
    responses: list[object] = [
        urllib.error.HTTPError(
            "https://example.test/llm",
            503,
            "unavailable",
            {},
            io.BytesIO(b'{"error": {"message": "temporary"}}'),
        ),
        _FakeHTTPResponse(b'{"ok": true}'),
    ]

    def fake_urlopen(_request: object, *, timeout: float) -> object:
        assert timeout == 7.0
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)

    result = llm_module._post_json("https://example.test/llm", {"prompt": "x"}, timeout_seconds=7.0)

    assert result == {"ok": True}
    assert sleeps == [0.25]
    assert responses == []


def test_llm_post_json_does_not_retry_non_transient_http_errors(monkeypatch) -> None:
    monkeypatch.setenv("EVOFOREST_LLM_MAX_RETRIES", "2")
    sleeps: list[float] = []

    def fake_urlopen(_request: object, *, timeout: float) -> object:
        raise urllib.error.HTTPError(
            "https://example.test/llm",
            400,
            "bad request",
            {},
            io.BytesIO(b'{"error": {"message": "invalid"}}'),
        )

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="status 400"):
        llm_module._post_json("https://example.test/llm", {"prompt": "x"}, timeout_seconds=7.0)
    assert sleeps == []


class _FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_missing_env_llm_provider_is_an_error_when_required(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EVOFOREST_LLM_PROVIDER", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=secret\nEVOFOREST_LLM_MODEL=gpt-test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="EVOFOREST_LLM_PROVIDER"):
        llm_provider_from_env(env_file, required=True)


def test_prompt_builder_advertises_source_schema_only_when_enabled() -> None:
    dataset = make_structural_break_data(n_series=35, length=70, seed=24)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=24, max_configurations=4).evaluate(graph, dataset.inputs(), dataset.y)
    hypothesis = ScientistAgent().generate(graph, result, max_hypotheses=1)
    _system, source_user = PromptBuilder(allow_source=True).engineer_prompts(graph, result, hypothesis)
    _system, primitive_user = PromptBuilder(allow_source=False).engineer_prompts(graph, result, hypothesis)
    assert '"source": "lambda ctx, values:' in source_user
    assert 'add:\n  output:\n    - "lambda ctx, values:' in source_user
    assert '"output_contract": {"type": "feature_block"' in source_user
    assert '"torch_source": "lambda ctx, values:' in source_user
    assert "infer parents" in source_user
    assert '"source": "lambda ctx, values:' not in primitive_user
    assert 'add:\n  output:\n    - "lambda ctx, values:' not in primitive_user


def test_failed_llm_source_mutation_is_logged_and_fed_back(tmp_path) -> None:
    dataset = make_structural_break_data(n_series=45, length=70, seed=26)
    graph = build_structural_break_seed_graph()
    bad_document = MutationDocument(
        hypotheses=("Try a source alternative that fails at runtime.",),
        rationale="Exercise execution-error feedback.",
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="output",
                primitive="source",
                alternative_id="bad_source_output",
                parents=("segment_stats",),
                source="lambda ctx, values: values['missing_parent']",
                description="Intentional runtime failure.",
            ),
        ),
    )
    repair_document = MutationDocument(
        hypotheses=("Repair by using a registry-backed spectral alternative.",),
        rationale="Use a known primitive after the source failure.",
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="shape_stats",
                primitive="spectral_basic",
                alternative_id="spectral_after_error",
                parents=("series",),
                description="Valid repair after runtime failure.",
            ),
        ),
    )
    client = StaticLLMClient(
        (
            "Hypothesis: Add a failing source output.\nRationale: Exercise failures.\nExpected Improvement: none.\nRisk Mode: Risky.",
            bad_document.to_yaml(),
            "Hypothesis: Use a safer shape_stats primitive.\nRationale: Prior source failed.\nExpected Improvement: recover valid search.\nRisk Mode: Conservative.",
            repair_document.to_yaml(),
        )
    )
    result = EvolutionLoop(
        graph,
        evaluator=RidgeEvaluator(n_splits=3, seed=26, max_configurations=4),
        mutation_engine=MutationEngine(allow_source=True),
        scientist=LLMScientistAgent(client),
        engineer=LLMEngineerAgent(client, allow_source=True),
        seed=26,
    ).run(dataset.inputs(), dataset.y, steps=1, output_dir=tmp_path)
    assert result.score > 0.6
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert "failed" not in events[0] or events[0]["failed"] is False
    assert events[0]["mutation"]["add"][0]["alternative_id"] == "spectral_after_error"
    assert (tmp_path / "mutations" / "step_0001_repair_01.yaml").exists()
    memorandum = (tmp_path / "memorandum.md").read_text(encoding="utf-8")
    assert "[ERROR LOG]" in memorandum
    assert "KeyError" in memorandum
    assert "KeyError" in str(client.requests[3]["user_prompt"])


def test_llm_island_mode_uses_scientist_temperature_schedule(tmp_path) -> None:
    dataset = make_structural_break_data(n_series=42, length=70, seed=30)
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        hypotheses=("Add a spectral shape alternative from island diagnostics.",),
        rationale="Exercise island-specific LLM temperature scheduling.",
        add=(
            MutationSpec(
                kind="add_alternative",
                target_node="shape_stats",
                primitive="spectral_basic",
                alternative_id="spectral_island_temp",
                parents=("series",),
                description="Island-scheduled spectral statistic.",
            ),
        ),
    )
    client = StaticLLMClient(
        (
            "\n".join(
                [
                    "Hypothesis: Add island spectral shape.",
                    "Rationale: residual frequency evidence.",
                    "Expected Improvement: complementary coverage.",
                    "Risk Mode: Balanced.",
                ]
            ),
            document.to_yaml(),
            "\n".join(
                [
                    "Hypothesis: Add island spectral shape.",
                    "Rationale: residual frequency evidence.",
                    "Expected Improvement: complementary coverage.",
                    "Risk Mode: Balanced.",
                ]
            ),
            document.to_yaml(),
        )
    )

    result = EvolutionLoop(
        graph,
        evaluator=RidgeEvaluator(n_splits=3, seed=30, max_configurations=4),
        scientist=LLMScientistAgent(client, island_temperatures=(0.35, 0.5)),
        engineer=LLMEngineerAgent(client, temperature=0.0),
        seed=30,
    ).run_islands(
        dataset.inputs(),
        dataset.y,
        islands=2,
        steps_per_island=1,
        output_dir=tmp_path,
    )

    assert result.score > 0.6
    assert [request["temperature"] for request in client.requests] == [0.35, 0.0, 0.5, 0.0]


def test_async_islands_record_failed_candidates_without_crashing(tmp_path) -> None:
    class BadSourceEngineer(EngineerAgent):
        def synthesize(self, graph, result, hypotheses, step, island, rng):  # type: ignore[no-untyped-def]
            del graph, result, hypotheses, rng
            return MutationDocument(
                hypotheses=("Async island source failure.",),
                rationale="Ensure failed worker candidates become events.",
                add=(
                    MutationSpec(
                        kind="add_alternative",
                        target_node="output",
                        primitive="source",
                        alternative_id=f"bad_async_{island}_{step}",
                        parents=("segment_stats",),
                        source="lambda ctx, values: values['missing_async_parent']",
                        description="Intentional async runtime failure.",
                    ),
                ),
            )

    dataset = make_structural_break_data(n_series=42, length=70, seed=28)
    graph = build_structural_break_seed_graph()
    result = EvolutionLoop(
        graph,
        evaluator=RidgeEvaluator(n_splits=3, seed=28, max_configurations=4),
        mutation_engine=MutationEngine(allow_source=True),
        engineer=BadSourceEngineer(),
        seed=28,
    ).run_async_islands(
        dataset.inputs(),
        dataset.y,
        islands=2,
        steps_per_island=1,
        output_dir=tmp_path,
        max_workers=2,
    )
    assert result.score > 0.7
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert all(event["failed"] is True for event in events)
    assert all(event["score"] is None for event in events)
    assert all("KeyError" in event["error"] for event in events)
    assert "[ERROR LOG]" in (tmp_path / "memorandum.md").read_text(encoding="utf-8")


def test_island_evolution_writes_global_and_island_artifacts(tmp_path) -> None:
    dataset = make_structural_break_data(n_series=60, length=80, seed=25)
    graph = build_structural_break_seed_graph()
    result = EvolutionLoop(graph, evaluator=RidgeEvaluator(n_splits=3, seed=25, max_configurations=8), seed=25).run_islands(
        dataset.inputs(),
        dataset.y,
        islands=2,
        steps_per_island=2,
        output_dir=tmp_path,
        migration_interval=2,
    )
    assert result.score > 0.7
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 4
    assert json.loads(events[0])["island"] in {0, 1}
    archive_rows = [
        json.loads(line)
        for line in (tmp_path / "archive" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert archive_rows[0]["mode"] == "sequential_island"
    assert archive_rows[0]["path"].startswith("global_best_v0000_step_0000_")
    assert (tmp_path / "archive" / archive_rows[0]["path"]).exists()
    assert (tmp_path / "mutations" / "island_0_step_0001.yaml").exists()
    assert (tmp_path / "task_context.md").exists()
    assert (tmp_path / "island_0" / "task_context.md").exists()
    assert (tmp_path / "island_0" / "memorandum.md").exists()
    assert (tmp_path / "island_1" / "memorandum.md").exists()


def test_async_island_evolution_writes_concurrent_artifacts(tmp_path) -> None:
    dataset = make_structural_break_data(n_series=50, length=70, seed=27)
    graph = build_structural_break_seed_graph()
    result = EvolutionLoop(graph, evaluator=RidgeEvaluator(n_splits=3, seed=27, max_configurations=6), seed=27).run_async_islands(
        dataset.inputs(),
        dataset.y,
        islands=2,
        steps_per_island=2,
        output_dir=tmp_path,
        migration_interval=2,
        max_workers=2,
    )
    assert result.score > 0.7
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert len(events) == 4
    assert {event["mode"] for event in events} == {"async_island"}
    assert {event["round"] for event in events} == {1, 2}
    assert all("proposed_step" in event for event in events)
    archive_rows = [
        json.loads(line)
        for line in (tmp_path / "archive" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert archive_rows[0]["mode"] == "async_island"
    assert (tmp_path / "archive" / archive_rows[0]["path"]).exists()
    assert (tmp_path / "mutations" / "island_0_step_0001.yaml").exists()
    assert (tmp_path / "task_context.md").exists()
    assert (tmp_path / "island_0" / "task_context.md").exists()
    assert (tmp_path / "island_0" / "memorandum.md").exists()
    assert (tmp_path / "island_1" / "memorandum.md").exists()


def test_demo_predictions_are_deterministic() -> None:
    dataset = make_structural_break_data(n_series=60, length=70, seed=31)
    graph = build_structural_break_seed_graph()
    config = graph.default_config()
    x1, names1, _ = graph.evaluate_features(dataset.inputs(), config)
    x2, names2, _ = graph.evaluate_features(dataset.inputs(), config)
    assert names1 == names2
    assert np.max(np.abs(x1 - x2)) == 0.0
