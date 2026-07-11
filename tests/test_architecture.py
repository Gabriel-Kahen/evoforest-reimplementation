from __future__ import annotations

from collections import OrderedDict
import io
import json
import argparse
import time
import urllib.error

import numpy as np
import pytest

from evoforest_arch import evaluator as evaluator_module, graph as graph_module, llm as llm_module
from evoforest_arch.cli import build_llm_agents, main as cli_main
from evoforest_arch.evaluator import _SCREENING_PREPARED_FOLD_CACHE_MAX_ENTRIES, PreparedFold, RidgeEvaluator, abs_feature_correlation, effective_rank, feature_correlation_summary, max_correlation_scores, most_correlated_names, safe_corr_columns
from evoforest_arch.evaluation_cache import PersistentEvaluationCache, fingerprint_callable
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.feedback import toon_report
from evoforest_arch.graph import CallableFamily, EvalContext, EvaluationKeyFingerprints, FeatureBlock, Graph, NodeAlternative, ResidualWeightRule
from evoforest_arch.graph_io import graph_from_dict, graph_hash
from evoforest_arch.hypotheses import Hypothesis
from evoforest_arch.llm import ClaudeLLMClient, GeminiLLMClient, LLMEngineerAgent, LLMMemorandumAgent, LLMScientistAgent, OpenAILLMClient, PromptBuilder, StaticLLMClient, llm_client_from_env, llm_provider_from_env
from evoforest_arch.maintenance import GraphMaintenance
from evoforest_arch.metrics import FoldStrategy, RMSE_SCORER, TaskScorer, safe_corr
from evoforest_arch.mutations import GlobalSpec, MutationDocument, MutationEngine, MutationSpec, NodeSpec, RemoveSpec
from evoforest_arch.primitives import PrimitiveRegistry
from evoforest_arch.rank_readout import fit_rank_feature_expansion, select_rank_ensemble
from evoforest_arch.readout import fit_ridge, select_alpha, select_alpha_and_fit_ridge
from evoforest_arch.seed import build_seed_graph, build_structural_break_seed_graph
from evoforest_arch.source import SourceExecutionError
from evoforest_arch.synthetic import make_structural_break_data, make_tabular_data
from evoforest_arch.task import InputSpec, TaskSchema
from tests.paper_test_support import PaperTestClient, paper_test_agents


_PERSISTENT_CACHE_CALLS: dict[str, int] = {}


def sample_hypothesis() -> Hypothesis:
    return Hypothesis(
        kind="residual_feature_search",
        target_node="output",
        rationale="Residual structure remains unexplained.",
        expected_improvement="complementary predictive structure",
    )


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


def test_paper_graph_contract_rejects_extra_output_and_fitting_nodes() -> None:
    graph = build_structural_break_seed_graph()
    graph.validate_paper_architecture()

    with pytest.raises(ValueError, match="logical output node must be named 'output'"):
        Graph().add_node("predictions", "output")
    with pytest.raises(ValueError, match="exactly one output node"):
        graph.add_node("output", "output")
    with pytest.raises(ValueError, match="Fitting nodes are limited"):
        Graph().add_node("custom_fit", "fitting")


def test_built_in_alternatives_have_typed_output_contracts() -> None:
    graph = build_structural_break_seed_graph()

    assert graph.nodes["segment_stats"].alternatives[0].output_contract == {
        "type": "feature_block",
        "min_columns": 1,
    }
    assert graph.nodes["activation"].alternatives[0].output_contract == {"type": "callable"}
    assert graph.nodes["output"].alternatives[0].output_contract == {
        "type": "feature_block",
        "min_columns": 1,
    }
    assert graph.nodes["ridge_w"].alternatives[0].output_contract == {"type": "sample_weight"}
    assert graph.nodes["ridge_g"].alternatives[0].output_contract == {"type": "residual_weight_rule"}


def test_registry_graph_round_trip_preserves_serialized_output_contract() -> None:
    graph = build_seed_graph()
    graph.nodes["output"].alternatives[0].output_contract["n_columns"] = 123

    restored = graph_from_dict(graph.to_dict())

    assert restored.nodes["output"].alternatives[0].output_contract["n_columns"] == 123
    assert graph_hash(restored) == graph_hash(graph)


def test_custom_registry_primitive_infers_contract_from_target_node() -> None:
    registry = PrimitiveRegistry(factories={})

    def custom_factory(alternative_id: str, parents: tuple[str, ...]) -> NodeAlternative:
        return NodeAlternative(
            alternative_id,
            parents,
            lambda _ctx, _values: CallableFamily("custom", lambda values: values),
        )

    registry.register("custom_gate", custom_factory)
    graph = build_seed_graph()
    graph.nodes["activation"].add_alternative(registry.build("custom_gate", "custom", ()))

    graph.validate_paper_architecture()
    assert graph.nodes["activation"].alternatives[-1].output_contract == {"type": "callable"}


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


def test_evaluation_key_fingerprints_hash_each_callable_once(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = build_structural_break_seed_graph()
    selected = graph.selected_config(graph.default_config())
    fingerprints = EvaluationKeyFingerprints()
    calls = 0
    original = graph_module.fingerprint_callable

    def counting_fingerprint(fn: object) -> str:
        nonlocal calls
        calls += 1
        return original(fn)

    monkeypatch.setattr(graph_module, "fingerprint_callable", counting_fingerprint)
    for _ in range(4):
        graph.alternative_cache_key("output", "raw_concat", selected, key_fingerprints=fingerprints)
    first_pass_calls = calls
    assert first_pass_calls > 0
    assert calls == len(fingerprints.alternatives)

    graph.alternative_cache_key("output", "raw_concat", selected, key_fingerprints=EvaluationKeyFingerprints())
    assert calls > first_pass_calls


def test_evaluation_key_fingerprints_do_not_cross_mutation_boundary() -> None:
    graph = Graph("evaluation_fingerprint_boundary")
    graph.add_input("x")
    graph.add_node("output", "output")
    scale = [1.0]

    def output_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        return FeatureBlock(np.asarray(values["x"], dtype=np.float64)[:, :1] * scale[0], ["value"])

    graph.add_alternative("output", "scaled", ("x",), output_fn)
    inputs = {"x": np.arange(12, dtype=np.float64).reshape(6, 2)}
    shared_cache: dict[object, object] = {}
    first, _, _ = graph.evaluate_features(inputs, cache=shared_cache, key_fingerprints=EvaluationKeyFingerprints())

    scale[0] = -1.0
    second, _, ctx = graph.evaluate_features(inputs, cache=shared_cache, key_fingerprints=EvaluationKeyFingerprints())

    np.testing.assert_allclose(second, -first)
    assert ctx.cache_hits == 0


def test_assembled_feature_bundle_reuses_frozen_snapshot_within_evaluation() -> None:
    graph = Graph("assembled_feature_bundle")
    graph.add_input("x")
    graph.add_node("output", "output")
    calls = [0]

    def output_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        calls[0] += 1
        return FeatureBlock(np.asarray(values["x"], dtype=np.float64)[:, :1], ["value"])

    graph.add_alternative("output", "base", ("x",), output_fn)
    inputs = {"x": np.arange(12, dtype=np.float64).reshape(6, 2)}
    cache = PersistentEvaluationCache(max_entries=8, max_bytes=1_000_000)
    cache.begin_evaluation()
    fingerprints = EvaluationKeyFingerprints()
    first, first_names, _ = graph.evaluate_features(inputs, cache=cache, key_fingerprints=fingerprints)
    second, second_names, ctx = graph.evaluate_features(inputs, cache=cache, key_fingerprints=fingerprints)

    assert calls == [1]
    np.testing.assert_array_equal(second, first)
    assert second_names == first_names
    assert ctx.cache_hits == 1
    assert second.flags.writeable is True
    second[0, 0] = -1.0
    third, _, _ = graph.evaluate_features(inputs, cache=cache, key_fingerprints=fingerprints)
    np.testing.assert_array_equal(third, first)


def test_assembled_feature_bundle_can_be_borrowed_read_only_internally() -> None:
    graph = Graph("borrowed_assembled_feature_bundle")
    graph.add_input("x")
    graph.add_node("output", "output")

    def output_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        return FeatureBlock(np.asarray(values["x"], dtype=np.float64)[:, :1], ["value"])

    graph.add_alternative("output", "base", ("x",), output_fn)
    inputs = {"x": np.arange(12, dtype=np.float64).reshape(6, 2)}
    cache = PersistentEvaluationCache(max_entries=8, max_bytes=1_000_000)
    cache.begin_evaluation()
    fingerprints = EvaluationKeyFingerprints()
    graph.evaluate_features(inputs, cache=cache, key_fingerprints=fingerprints)
    borrowed, _, _ = graph.evaluate_features(
        inputs,
        cache=cache,
        key_fingerprints=fingerprints,
        copy_cached_bundle=False,
    )

    assert borrowed.flags.writeable is False


def test_assembled_feature_bundle_isolated_from_public_plain_dict_results() -> None:
    graph = Graph("plain_dict_assembled_feature_bundle")
    graph.add_input("x")
    graph.add_node("output", "output")

    def output_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        return FeatureBlock(np.asarray(values["x"], dtype=np.float64)[:, :1], ["value"])

    graph.add_alternative("output", "base", ("x",), output_fn)
    inputs = {"x": np.arange(12, dtype=np.float64).reshape(6, 2)}
    cache: dict[object, object] = {}
    expected, expected_names, _ = graph.evaluate_features(inputs)
    first, first_names, _ = graph.evaluate_features(inputs, cache=cache)
    first[0, 0] = -1.0
    first_names[0] = "corrupted"

    second, second_names, _ = graph.evaluate_features(inputs, cache=cache)

    np.testing.assert_array_equal(second, expected)
    assert second_names == expected_names


def test_configuration_search_uses_ancestor_conditioned_shared_cache() -> None:
    dataset = make_structural_break_data(n_series=48, length=60, seed=9)
    graph = Graph("cache_test")
    graph.add_input("series", "Synthetic series.")
    graph.add_node("a", "intermediate", "Shared ancestor.")
    graph.add_node("b", "intermediate", "Child depending on selected ancestor.")
    graph.add_node("output", "output", "Scored output.")
    keys = ["search_a0", "search_a1", "search_b0", "search_b1", "search_out"]
    _PERSISTENT_CACHE_CALLS.update({key: 0 for key in keys})

    def a_fn(label: str, offset: float):
        def fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
            _PERSISTENT_CACHE_CALLS[f"search_{label}"] += 1
            series = np.asarray(values["series"], dtype=np.float64)
            return FeatureBlock(series[:, :1] + offset, [label])

        return fn

    def b_fn(label: str, scale: float):
        def fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
            _PERSISTENT_CACHE_CALLS[f"search_{label}"] += 1
            block = values["a"]
            assert isinstance(block, FeatureBlock)
            return FeatureBlock(block.values * scale, [label])

        return fn

    def out_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        _PERSISTENT_CACHE_CALLS["search_out"] += 1
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
    assert _PERSISTENT_CACHE_CALLS["search_a0"] == 1
    assert _PERSISTENT_CACHE_CALLS["search_a1"] == 1
    assert _PERSISTENT_CACHE_CALLS["search_b0"] == 2
    assert _PERSISTENT_CACHE_CALLS["search_b1"] == 2
    assert _PERSISTENT_CACHE_CALLS["search_out"] == 4


def test_persistent_cache_reuses_identical_graph_clone_across_evaluations() -> None:
    dataset = make_tabular_data(n_samples=48, n_features=3, seed=901)
    graph = Graph("persistent_clone")
    graph.add_input("x")
    graph.add_node("output", "output")
    _PERSISTENT_CACHE_CALLS["clone_output"] = 0

    def output_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        _PERSISTENT_CACHE_CALLS["clone_output"] += 1
        return FeatureBlock(np.asarray(values["x"], dtype=np.float64), ["a", "b", "c"])

    graph.add_alternative("output", "raw", ("x",), output_fn)
    evaluator = RidgeEvaluator(n_splits=2, seed=901, refine_globals=False, diagnostics_mode="basic", feature_pool_diagnostics=False)

    first = evaluator.evaluate(graph, dataset.inputs(), dataset.y, config={})
    expected_features = first.feature_matrix.copy()
    first.feature_matrix[0, 0] = 1_000_000.0
    second = evaluator.evaluate(graph.clone(), dataset.inputs(), dataset.y, config={})

    assert _PERSISTENT_CACHE_CALLS["clone_output"] == 1
    np.testing.assert_allclose(second.predictions, first.predictions)
    np.testing.assert_allclose(second.feature_matrix, expected_features)
    persistent = second.diagnostics["configuration_search"]["cache"]["persistent"]
    assert persistent["cross_evaluation_hits"] >= 1
    assert persistent["stores"] == 0

    changed_inputs = {"x": dataset.inputs()["x"].copy()}
    changed_inputs["x"][0, 0] += 0.5
    third = evaluator.evaluate(graph, changed_inputs, dataset.y, config={})
    assert _PERSISTENT_CACHE_CALLS["clone_output"] == 2
    assert third.diagnostics["configuration_search"]["cache"]["persistent"]["cross_evaluation_hits"] == 0


def test_persistent_cache_output_mutation_reuses_unchanged_subpaths() -> None:
    dataset = make_tabular_data(n_samples=48, n_features=3, seed=902)
    graph = Graph("persistent_output_mutation")
    graph.add_input("x")
    graph.add_node("shared", "intermediate")
    graph.add_node("output", "output")
    _PERSISTENT_CACHE_CALLS.update({"mutation_shared": 0, "mutation_old": 0, "mutation_new": 0})

    def shared_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        _PERSISTENT_CACHE_CALLS["mutation_shared"] += 1
        return FeatureBlock(np.asarray(values["x"], dtype=np.float64)[:, :1], ["shared"])

    def old_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        _PERSISTENT_CACHE_CALLS["mutation_old"] += 1
        block = values["shared"]
        assert isinstance(block, FeatureBlock)
        return FeatureBlock(block.values, ["old"])

    def new_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        _PERSISTENT_CACHE_CALLS["mutation_new"] += 1
        block = values["shared"]
        assert isinstance(block, FeatureBlock)
        return FeatureBlock(np.square(block.values), ["new"])

    graph.add_alternative("shared", "base", ("x",), shared_fn)
    graph.add_alternative("output", "old", ("shared",), old_fn)
    evaluator = RidgeEvaluator(n_splits=2, seed=902, refine_globals=False, diagnostics_mode="basic", feature_pool_diagnostics=False)
    evaluator.evaluate(graph, dataset.inputs(), dataset.y, config={})

    mutated = graph.clone()
    mutated.add_alternative("output", "new", ("shared",), new_fn)
    result = evaluator.evaluate(mutated, dataset.inputs(), dataset.y, config={})

    assert {
        "shared": _PERSISTENT_CACHE_CALLS["mutation_shared"],
        "old": _PERSISTENT_CACHE_CALLS["mutation_old"],
        "new": _PERSISTENT_CACHE_CALLS["mutation_new"],
    } == {"shared": 1, "old": 1, "new": 1}
    assert result.feature_names == ["output.old.old", "output.new.new"]
    assert result.diagnostics["configuration_search"]["cache"]["persistent"]["cross_evaluation_hits"] >= 2


def test_persistent_cache_invalidates_changed_global_and_descendants() -> None:
    dataset = make_tabular_data(n_samples=48, n_features=3, seed=903)
    graph = Graph("persistent_global_mutation")
    graph.add_input("x")
    graph.add_node("scaled", "intermediate")
    graph.add_node("output", "output")
    graph.globals.add("scale", [1.0], trainable=False)
    calls = {"scaled": 0, "output": 0}

    def scaled_fn(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        calls["scaled"] += 1
        scale = float(ctx.globals.get("scale")[0])
        return FeatureBlock(np.asarray(values["x"], dtype=np.float64)[:, :1] * scale, ["scaled"])

    def output_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        calls["output"] += 1
        block = values["scaled"]
        assert isinstance(block, FeatureBlock)
        return FeatureBlock(block.values, ["output"])

    graph.add_alternative("scaled", "base", ("x",), scaled_fn, global_refs=("scale",))
    graph.add_alternative("output", "base", ("scaled",), output_fn)
    evaluator = RidgeEvaluator(n_splits=2, seed=903, refine_globals=False, diagnostics_mode="basic", feature_pool_diagnostics=False)
    first = evaluator.evaluate(graph, dataset.inputs(), dataset.y, config={})

    mutated = graph.clone()
    mutated.globals.set("scale", [2.0])
    second = evaluator.evaluate(mutated, dataset.inputs(), dataset.y, config={})

    assert calls == {"scaled": 2, "output": 2}
    assert not np.array_equal(first.feature_matrix, second.feature_matrix)
    assert second.diagnostics["configuration_search"]["cache"]["persistent"]["cross_evaluation_hits"] == 0


def test_persistent_cache_invalidates_changed_parent_semantics_and_descendants() -> None:
    dataset = make_tabular_data(n_samples=48, n_features=3, seed=904)
    graph = Graph("persistent_parent_mutation")
    graph.add_input("x")
    graph.add_node("parent", "intermediate")
    graph.add_node("output", "output")
    calls = {"parent_v1": 0, "parent_v2": 0, "output": 0}

    def parent_v1(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        calls["parent_v1"] += 1
        return FeatureBlock(np.asarray(values["x"], dtype=np.float64)[:, :1], ["parent"])

    def parent_v2(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        calls["parent_v2"] += 1
        return FeatureBlock(-np.asarray(values["x"], dtype=np.float64)[:, :1], ["parent"])

    def output_fn(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
        calls["output"] += 1
        block = values["parent"]
        assert isinstance(block, FeatureBlock)
        return FeatureBlock(block.values, ["output"])

    graph.add_alternative("parent", "base", ("x",), parent_v1)
    graph.add_alternative("output", "base", ("parent",), output_fn)
    evaluator = RidgeEvaluator(n_splits=2, seed=904, refine_globals=False, diagnostics_mode="basic", feature_pool_diagnostics=False)
    first = evaluator.evaluate(graph, dataset.inputs(), dataset.y, config={})

    mutated = graph.clone()
    mutated.nodes["parent"].alternatives[0].fn = parent_v2
    second = evaluator.evaluate(mutated, dataset.inputs(), dataset.y, config={})

    assert calls == {"parent_v1": 1, "parent_v2": 1, "output": 2}
    np.testing.assert_allclose(second.feature_matrix, -first.feature_matrix)
    assert second.diagnostics["configuration_search"]["cache"]["persistent"]["cross_evaluation_hits"] == 0


def test_persistent_cache_fingerprints_captured_callable_state() -> None:
    def make_transform(sign: float):
        def transform(values: np.ndarray) -> np.ndarray:
            return values * sign

        return transform

    def make_output(transform):
        def output(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
            return FeatureBlock(transform(np.asarray(values["x"], dtype=np.float64)[:, :1]), ["value"])

        return output

    positive = make_output(make_transform(1.0))
    negative = make_output(make_transform(-1.0))
    assert fingerprint_callable(positive) != fingerprint_callable(negative)

    dataset = make_tabular_data(n_samples=48, n_features=3, seed=905)
    graph = Graph("captured_callable")
    graph.add_input("x")
    graph.add_node("output", "output")
    graph.add_alternative("output", "signed", ("x",), positive)
    evaluator = RidgeEvaluator(n_splits=2, seed=905, refine_globals=False, diagnostics_mode="basic")
    first = evaluator.evaluate(graph, dataset.inputs(), dataset.y, config={})

    mutated = graph.clone()
    mutated.nodes["output"].alternatives[0].fn = negative
    second = evaluator.evaluate(mutated, dataset.inputs(), dataset.y, config={})

    np.testing.assert_allclose(second.feature_matrix, -first.feature_matrix)
    assert second.diagnostics["configuration_search"]["cache"]["persistent"]["cross_evaluation_hits"] == 0


def test_disabling_persistent_cache_skips_input_fingerprinting() -> None:
    dataset = make_tabular_data(n_samples=40, n_features=3, seed=906)
    inputs = {**dataset.inputs(), "unused": lambda value: value}
    result = RidgeEvaluator(
        n_splits=2,
        seed=906,
        refine_globals=False,
        diagnostics_mode="basic",
        persistent_cache=False,
    ).evaluate(build_seed_graph(), inputs, dataset.y, config={})

    assert result.diagnostics["configuration_search"]["cache"]["persistent"]["enabled"] is False


def test_persistent_cache_byte_limit_counts_python_payloads() -> None:
    cache = PersistentEvaluationCache(max_entries=4, max_bytes=1_024)
    cache.begin_evaluation()
    cache["large"] = FeatureBlock(np.ones((1, 1)), ["x" * 2_000_000])

    assert len(cache) == 0
    assert cache.stats()["evictions"] == 1


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
    assert (
        result.diagnostics["configuration_search"]["evaluated"]
        == result.diagnostics["configuration_search"]["unique_feature_matrices"]
        + cache["feature_matrix_reuse_hits"]
    )


def test_full_diagnostics_are_computed_only_for_the_winning_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = make_tabular_data(n_samples=64, n_features=5, seed=49)
    evaluator = RidgeEvaluator(
        n_splits=2,
        seed=49,
        max_configurations=8,
        diagnostics_mode="full",
        feature_pool_diagnostics=False,
    )
    full_diagnostic_calls = 0
    objective_diagnostic_calls = 0
    original_diagnostics = evaluator._diagnostics
    original_objective_diagnostics = evaluator._objective_diagnostics

    def counted_diagnostics(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal full_diagnostic_calls
        full_diagnostic_calls += 1
        return original_diagnostics(*args, **kwargs)

    def counted_objective_diagnostics(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal objective_diagnostic_calls
        objective_diagnostic_calls += 1
        return original_objective_diagnostics(*args, **kwargs)

    monkeypatch.setattr(evaluator, "_diagnostics", counted_diagnostics)
    monkeypatch.setattr(evaluator, "_objective_diagnostics", counted_objective_diagnostics)
    result = evaluator.evaluate(build_seed_graph(), dataset.inputs(), dataset.y)

    assert result.diagnostics["configuration_search"]["evaluated"] == 8
    assert full_diagnostic_calls == 1
    assert objective_diagnostic_calls == 1
    assert result.diagnostics["evaluation_passes"] == {
        "configuration_scoring": "score",
        "winner_diagnostics": "full",
        "winner_rerun": True,
    }
    assert result.diagnostics["features"]


def test_feature_pool_contains_each_unique_matrix_once() -> None:
    dataset = make_tabular_data(n_samples=64, n_features=5, seed=50)
    result = RidgeEvaluator(
        n_splits=2,
        seed=50,
        max_configurations=36,
        diagnostics_mode="basic",
    ).evaluate(build_seed_graph(), dataset.inputs(), dataset.y)

    search = result.diagnostics["configuration_search"]
    assert result.diagnostics["valid_feature_pool"]["n_configurations"] == search["unique_feature_matrices"]
    assert result.diagnostics["valid_feature_pool"]["n_configurations"] < search["evaluated"]


def test_capped_configuration_planner_is_seed_independent_and_covers_alternatives() -> None:
    graph = build_structural_break_seed_graph()
    first, total = RidgeEvaluator(seed=1, max_configurations=8)._configuration_candidates(graph)
    second, second_total = RidgeEvaluator(seed=999, max_configurations=8)._configuration_candidates(graph)

    assert first == second
    assert total == second_total
    assert len(first) == 8
    assert first[0] == graph.default_config()
    for node_name, alternatives in graph.configuration_space().items():
        assert {config[node_name] for config in first} == set(alternatives)


def test_staged_configuration_screening_uses_exact_cv_for_final_selection() -> None:
    dataset = make_structural_break_data(n_series=72, length=70, seed=45)
    evaluator = RidgeEvaluator(
        n_splits=3,
        seed=45,
        max_configurations=64,
        screening_finalists=5,
        refine_globals=False,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    result = evaluator.evaluate(build_structural_break_seed_graph(), dataset.inputs(), dataset.y)
    search = result.diagnostics["configuration_search"]

    assert search["planned"] == search["total"] == 48
    assert search["evaluated"] == 5
    assert search["capped"] is True
    assert search["planner_capped"] is False
    assert search["screening_capped"] is True
    assert search["planner"]["method"] == "deterministic_exhaustive"
    assert search["planner"]["coverage_fraction"] == 1.0
    assert search["screening"]["enabled"] is True
    assert search["screening"]["approximate"] is True
    assert search["screening"]["screened"] == 48
    assert search["screening"]["unique_work_units"] == 24
    assert search["screening"]["work_reuse_hits"] == 24
    assert search["screening"]["finalists"] == 5
    assert search["screening"]["winner_selected_by_exact_cv"] is True
    assert search["screening"]["full_plan_exact_best_recall_guaranteed"] is False
    assert search["screening"]["winner_config"] == result.config
    assert search["best_config_score"] == result.score
    assert search["top_configs"][0]["config"] == result.config

    repeated = evaluator.evaluate(build_structural_break_seed_graph(), dataset.inputs(), dataset.y)
    assert repeated.config == result.config
    assert repeated.score == result.score
    repeated_screening = dict(repeated.diagnostics["configuration_search"]["screening"])
    first_screening = dict(search["screening"])
    repeated_screening.pop("cache_entries_added")
    first_screening.pop("cache_entries_added")
    assert repeated_screening == first_screening


def test_reused_configuration_work_matches_independent_evaluation() -> None:
    dataset = make_structural_break_data(n_series=48, length=55, seed=145)
    inputs = dataset.inputs()
    graph = build_structural_break_seed_graph()
    kwargs = {
        "n_splits": 3,
        "seed": 145,
        "max_configurations": 64,
        "screening_finalists": 6,
        "refine_globals": False,
        "diagnostics_mode": "basic",
        "feature_pool_diagnostics": False,
        "persistent_cache": False,
    }
    reference = RidgeEvaluator(**kwargs)
    folds, fold_diagnostics = reference._folds(inputs, dataset.y)
    configs, _total = reference._configuration_candidates(graph)
    screening_rows = []
    for planner_index, candidate in enumerate(configs):
        row = reference._screen_single_config(graph, inputs, dataset.y, candidate, {}, folds)
        row["planner_index"] = planner_index
        screening_rows.append(row)
    finalists = reference._screening_finalists(graph, configs, screening_rows)
    independent = [
        reference._evaluate_single_config(
            graph,
            inputs,
            dataset.y,
            candidate,
            {},
            folds=folds,
            fold_diagnostics=fold_diagnostics,
            diagnostics_mode="score",
        )
        for candidate in finalists
    ]
    expected = max(independent, key=lambda result: result.score)

    reused = RidgeEvaluator(**kwargs).evaluate(graph, inputs, dataset.y)

    assert reused.config == expected.config
    assert reused.score == expected.score
    assert reused.alphas == expected.alphas
    np.testing.assert_array_equal(reused.predictions, expected.predictions)
    assert reused.diagnostics["configuration_search"]["screening"]["unique_work_units"] == 24
    search = reused.diagnostics["configuration_search"]
    cache = search["cache"]
    assert 0 < cache["prepared_fold_entries"] <= kwargs["n_splits"] * search["unique_feature_matrices"]
    assert 0 < cache["initial_ridge_entries"] <= kwargs["n_splits"] * search["evaluated"]


def test_exact_work_caches_respect_tiny_byte_budget() -> None:
    dataset = make_tabular_data(n_samples=64, n_features=5, seed=245)
    graph = build_seed_graph()
    kwargs = {
        "n_splits": 3,
        "seed": 245,
        "max_configurations": 8,
        "screening_finalists": 8,
        "refine_globals": False,
        "diagnostics_mode": "basic",
        "feature_pool_diagnostics": False,
        "persistent_cache": False,
    }
    expected = RidgeEvaluator(**kwargs).evaluate(graph, dataset.inputs(), dataset.y)
    bounded = RidgeEvaluator(**kwargs, cache_max_bytes=1).evaluate(graph, dataset.inputs(), dataset.y)

    assert bounded.score == expected.score
    assert bounded.config == expected.config
    np.testing.assert_array_equal(bounded.predictions, expected.predictions)
    cache = bounded.diagnostics["configuration_search"]["cache"]
    assert cache["prepared_fold_entries"] == 0
    assert cache["initial_ridge_entries"] == 0


def test_exact_work_reports_progress_in_planner_order() -> None:
    dataset = make_tabular_data(n_samples=64, n_features=5, seed=246)
    graph = build_seed_graph()
    kwargs = {
        "n_splits": 2,
        "seed": 246,
        "max_configurations": 8,
        "screening_finalists": 8,
        "refine_globals": False,
        "diagnostics_mode": "basic",
        "feature_pool_diagnostics": False,
        "persistent_cache": False,
    }
    reference = RidgeEvaluator(**kwargs)
    folds, fold_diagnostics = reference._folds(dataset.inputs(), dataset.y)
    configs, _ = reference._configuration_candidates(graph)
    expected_scores = [
        reference._evaluate_single_config(
            graph,
            dataset.inputs(),
            dataset.y,
            config,
            {},
            folds=folds,
            fold_diagnostics=fold_diagnostics,
            diagnostics_mode="score",
        ).score
        for config in configs
    ]
    events: list[dict[str, object]] = []
    RidgeEvaluator(**kwargs, progress_callback=events.append).evaluate(graph, dataset.inputs(), dataset.y)
    configuration_events = [event for event in events if event.get("phase") == "configuration_evaluated"]

    assert [event["config_index"] for event in configuration_events] == list(range(1, len(configs) + 1))
    assert [event["score"] for event in configuration_events] == expected_scores


def test_exact_progress_callback_can_interrupt_before_fold_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = make_tabular_data(n_samples=64, n_features=5, seed=247)
    fold_fit_calls = 0

    def stop_on_features(payload: dict[str, object]) -> None:
        if payload.get("phase") == "features_evaluated":
            raise RuntimeError("stop before folds")

    evaluator = RidgeEvaluator(
        n_splits=3,
        seed=247,
        max_configurations=8,
        screening_finalists=8,
        refine_globals=False,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
        persistent_cache=False,
        progress_callback=stop_on_features,
    )
    original_fit = evaluator._fit_reweighted_ridge

    def counted_fit(*args: object, **kwargs: object):
        nonlocal fold_fit_calls
        fold_fit_calls += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(evaluator, "_fit_reweighted_ridge", counted_fit)

    with pytest.raises(RuntimeError, match="stop before folds"):
        evaluator.evaluate(build_seed_graph(), dataset.inputs(), dataset.y)

    assert fold_fit_calls == 0


def test_screening_reuses_prepared_fold_by_feature_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = make_structural_break_data(n_series=48, length=55, seed=146)
    inputs = dataset.inputs()
    graph = build_structural_break_seed_graph()
    evaluator = RidgeEvaluator(
        n_splits=3,
        max_configurations=64,
        screening_finalists=6,
        refine_globals=False,
        persistent_cache=False,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    folds, _fold_diagnostics = evaluator._folds(inputs, dataset.y)
    configs, _total = evaluator._configuration_candidates(graph)
    signatures = [evaluator._feature_signature(graph, config) for config in configs]
    prepared_cache: OrderedDict[tuple[object, ...], PreparedFold] = OrderedDict()
    prepare_calls = 0
    original_prepare_fold = evaluator._prepare_fold

    def counted_prepare_fold(x: np.ndarray, train_idx: np.ndarray, validation_idx: np.ndarray) -> PreparedFold:
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare_fold(x, train_idx, validation_idx)

    monkeypatch.setattr(evaluator, "_prepare_fold", counted_prepare_fold)
    cached_rows = [
        evaluator._screen_single_config(
            graph,
            inputs,
            dataset.y,
            config,
            {},
            folds,
            prepared_fold_cache=prepared_cache,
            prepared_cache_key=signature,
        )
        for config, signature in zip(configs, signatures, strict=True)
    ]

    assert prepare_calls == len(set(signatures)) == 12
    assert len(prepared_cache) == 12
    uncached_rows = [
        evaluator._screen_single_config(graph, inputs, dataset.y, config, {}, folds)
        for config in configs
    ]
    assert [row["approximate_score"] for row in cached_rows] == [row["approximate_score"] for row in uncached_rows]
    assert [row["approximate_raw_score"] for row in cached_rows] == [row["approximate_raw_score"] for row in uncached_rows]


def test_screening_prepared_fold_cache_is_bounded() -> None:
    dataset = make_structural_break_data(n_series=24, length=32, seed=147)
    inputs = dataset.inputs()
    graph = build_structural_break_seed_graph()
    evaluator = RidgeEvaluator(n_splits=2, refine_globals=False, persistent_cache=False)
    folds, _fold_diagnostics = evaluator._folds(inputs, dataset.y)
    config = graph.default_config()
    prepared_cache: OrderedDict[tuple[object, ...], PreparedFold] = OrderedDict()
    byte_budget = 10_000

    for index in range(_SCREENING_PREPARED_FOLD_CACHE_MAX_ENTRIES + 5):
        evaluator._screen_single_config(
            graph,
            inputs,
            dataset.y,
            config,
            {},
            folds,
            prepared_fold_cache=prepared_cache,
            prepared_cache_key=(("synthetic_signature", index),),
            prepared_cache_max_bytes=byte_budget,
        )

    retained_bytes = sum(fold.x_train.nbytes + fold.x_validation.nbytes for fold in prepared_cache.values())
    assert 0 < len(prepared_cache) <= _SCREENING_PREPARED_FOLD_CACHE_MAX_ENTRIES
    assert retained_bytes <= byte_budget
    assert (("synthetic_signature", 0),) not in prepared_cache
    assert (("synthetic_signature", _SCREENING_PREPARED_FOLD_CACHE_MAX_ENTRIES + 4),) in prepared_cache


def test_screening_reuse_respects_ridge_g_dependencies() -> None:
    graph = Graph("screening_ridge_g_dependency")
    graph.add_input("x")
    graph.add_node("output", "output")
    graph.add_node("ridge_g", "fitting")
    graph.add_node("ridge_w", "fitting")
    graph.add_alternative(
        "output",
        "base",
        ("x",),
        lambda _ctx, values: FeatureBlock(np.asarray(values["x"], dtype=np.float64), ["x"]),
    )
    for name in ("hi", "lo"):
        graph.add_alternative(
            "ridge_g",
            name,
            (),
            lambda _ctx, _values, rule_name=name: ResidualWeightRule(
                rule_name,
                lambda residual: np.ones_like(residual),
            ),
        )

    def dependent_weight(_ctx: EvalContext, values: dict[str, object]) -> np.ndarray:
        x = np.asarray(values["x"], dtype=np.float64)[:, 0]
        rule = values["ridge_g"]
        assert isinstance(rule, ResidualWeightRule)
        positive_weight, negative_weight = (10.0, 0.1) if rule.name == "hi" else (0.1, 10.0)
        return np.where(x > 0.0, positive_weight, negative_weight)

    graph.add_alternative("ridge_w", "dependent", ("x", "ridge_g"), dependent_weight)
    rng = np.random.default_rng(10)
    x = np.linspace(-2.0, 2.0, 60)[:, None]
    y = np.where(x[:, 0] > 0.0, 5.0 * x[:, 0] + rng.normal(0.0, 0.2, 60), rng.normal(0.0, 5.0, 60))
    evaluator = RidgeEvaluator(
        n_splits=3,
        max_configurations=2,
        screening_finalists=1,
        refine_globals=False,
        persistent_cache=False,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    configs, _ = evaluator._configuration_candidates(graph)

    assert evaluator._screening_work_key(graph, configs[0]) != evaluator._screening_work_key(graph, configs[1])
    result = evaluator.evaluate(graph, {"x": x}, y)

    assert result.config["ridge_g"] == "lo"
    assert result.diagnostics["configuration_search"]["screening"]["unique_work_units"] == 2


def test_grouped_exact_evaluation_preserves_planner_winner_for_nan_scores() -> None:
    graph = Graph("nan_score_planner_order")
    graph.add_input("x")
    graph.add_node("mid", "intermediate")
    graph.add_node("output", "output")
    for name, scale in (("z", 1.0), ("a", 2.0), ("m", 3.0)):
        graph.add_alternative(
            "mid",
            name,
            ("x",),
            lambda _ctx, values, factor=scale, feature_name=name: FeatureBlock(
                np.asarray(values["x"], dtype=np.float64) * factor,
                [feature_name],
            ),
        )
    graph.add_alternative("output", "base", ("mid",), lambda _ctx, values: values["mid"])
    x = np.linspace(-1.0, 1.0, 36)[:, None]
    scorer = TaskScorer("nan", "Always returns NaN.", lambda _y, _predictions: float("nan"))

    result = RidgeEvaluator(
        n_splits=3,
        max_configurations=3,
        screening_finalists=3,
        refine_globals=False,
        persistent_cache=False,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
        scorer=scorer,
    ).evaluate(graph, {"x": x}, x[:, 0])

    assert result.config["mid"] == "z"


def test_single_screening_finalist_keeps_the_approximate_winner() -> None:
    graph = build_structural_break_seed_graph()
    evaluator = RidgeEvaluator(screening_finalists=1)
    configs, _total = evaluator._configuration_candidates(graph)
    default = graph.default_config()
    challenger = next(config for config in configs if config != default)
    rows = [
        {"config": default, "approximate_score": 0.0, "planner_index": 0},
        {"config": challenger, "approximate_score": 1.0, "planner_index": 1},
    ]

    assert evaluator._screening_finalists(graph, [default, challenger], rows) == [challenger]


def test_large_configuration_plan_avoids_quadratic_distance_rescans() -> None:
    class PlannerGraph:
        @staticmethod
        def configuration_space() -> dict[str, list[str]]:
            return {f"axis_{axis}": [f"alternative_{index}" for index in range(10)] for axis in range(8)}

        def default_config(self) -> dict[str, str]:
            return {key: values[0] for key, values in self.configuration_space().items()}

    started = time.monotonic()
    configs, total = RidgeEvaluator(max_configurations=128)._configuration_candidates(PlannerGraph())  # type: ignore[arg-type]

    assert len(configs) == 128
    assert total == 100_000_000
    assert time.monotonic() - started < 3.0


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


def test_registered_immutable_dataset_reuses_fingerprint_and_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = make_tabular_data(n_samples=64, n_features=5, seed=146)
    inputs = dataset.inputs()
    graph = build_seed_graph()
    fingerprint_calls = 0
    fold_calls = 0
    original_fingerprint = evaluator_module.fingerprint_inputs

    def counting_fingerprint(values: dict[str, object]) -> str:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return original_fingerprint(values)

    monkeypatch.setattr(evaluator_module, "fingerprint_inputs", counting_fingerprint)
    evaluator = RidgeEvaluator(
        n_splits=2,
        seed=146,
        max_configurations=1,
        refine_globals=False,
        fold_strategy=FoldStrategy(),
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    original_folds = evaluator._folds

    def counting_folds(values: dict[str, object], target: np.ndarray):
        nonlocal fold_calls
        fold_calls += 1
        return original_folds(values, target)

    monkeypatch.setattr(evaluator, "_folds", counting_folds)
    token = evaluator.register_immutable_dataset(inputs, dataset.y)
    first = evaluator.evaluate(graph, inputs, dataset.y, config=graph.default_config())
    second = evaluator.evaluate(graph, inputs, dataset.y, config=graph.default_config())

    assert token > 0
    assert fingerprint_calls == 1
    assert fold_calls == 1
    assert evaluator.evaluation_cache_diagnostics()["immutable_dataset_hits"] == 2
    assert first.config == second.config
    assert first.score == second.score
    np.testing.assert_array_equal(first.predictions, second.predictions)


def test_immutable_dataset_registration_is_identity_scoped_and_revocable(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = make_tabular_data(n_samples=48, n_features=4, seed=147)
    inputs = dataset.inputs()
    copied_inputs = {key: np.array(value, copy=True) for key, value in inputs.items()}
    copied_y = np.array(dataset.y, copy=True)
    graph = build_seed_graph()
    fingerprint_calls = 0
    original_fingerprint = evaluator_module.fingerprint_inputs

    def counting_fingerprint(values: dict[str, object]) -> str:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return original_fingerprint(values)

    monkeypatch.setattr(evaluator_module, "fingerprint_inputs", counting_fingerprint)
    evaluator = RidgeEvaluator(
        n_splits=2,
        seed=147,
        max_configurations=1,
        refine_globals=False,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    token = evaluator.register_immutable_dataset(inputs, dataset.y)
    evaluator.evaluate(graph, copied_inputs, copied_y, config=graph.default_config())
    assert fingerprint_calls == 2
    assert evaluator.evaluation_cache_diagnostics()["immutable_dataset_hits"] == 0

    assert evaluator.unregister_immutable_dataset(token) is True
    assert evaluator.unregister_immutable_dataset(token) is False
    inputs["x"][0, 0] += 10.0
    evaluator.evaluate(graph, inputs, dataset.y, config=graph.default_config())
    assert fingerprint_calls == 3
    assert evaluator.evaluation_cache_diagnostics()["immutable_dataset_registrations"] == 0


def test_registered_dataset_recomputes_folds_after_fold_configuration_change() -> None:
    class CountingFoldStrategy(FoldStrategy):
        def __init__(self) -> None:
            self.calls = 0

        def split(self, inputs: dict[str, object], y: np.ndarray, n_splits: int, seed: int):
            self.calls += 1
            return super().split(inputs, y, n_splits, seed)

    dataset = make_tabular_data(n_samples=48, n_features=4, seed=148)
    inputs = dataset.inputs()
    strategy = CountingFoldStrategy()
    evaluator = RidgeEvaluator(
        n_splits=2,
        seed=148,
        max_configurations=1,
        refine_globals=False,
        fold_strategy=strategy,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    evaluator.register_immutable_dataset(inputs, dataset.y)
    evaluator.n_splits = 3
    graph = build_seed_graph()
    result = evaluator.evaluate(graph, inputs, dataset.y, config=graph.default_config())

    assert strategy.calls == 2
    assert len(result.diagnostics["folds"]["score"]) == 3


def test_registered_dataset_recomputes_folds_after_custom_strategy_state_change() -> None:
    class OffsetFoldStrategy(FoldStrategy):
        def __init__(self) -> None:
            object.__setattr__(self, "offset", 0)
            object.__setattr__(self, "calls", 0)

        def split(self, inputs: dict[str, object], y: np.ndarray, n_splits: int, seed: int):
            del inputs, seed
            object.__setattr__(self, "calls", self.calls + 1)
            indices = np.roll(np.arange(y.shape[0]), self.offset)
            validation_blocks = np.array_split(indices, n_splits)
            folds = []
            for validation in validation_blocks:
                train = np.setdiff1d(indices, validation, assume_unique=True)
                folds.append((train, validation))
            return folds, {"method": "offset", "offset": self.offset}

    dataset = make_tabular_data(n_samples=48, n_features=4, seed=149)
    inputs = dataset.inputs()
    strategy = OffsetFoldStrategy()
    evaluator = RidgeEvaluator(
        n_splits=3,
        seed=149,
        max_configurations=1,
        refine_globals=False,
        fold_strategy=strategy,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    evaluator.register_immutable_dataset(inputs, dataset.y)
    object.__setattr__(strategy, "offset", 1)

    result = evaluator.evaluate(build_seed_graph(), inputs, dataset.y, config=build_seed_graph().default_config())
    evaluator.evaluate(build_seed_graph(), inputs, dataset.y, config=build_seed_graph().default_config())

    assert strategy.calls == 3
    assert result.diagnostics["folds"]["offset"] == 1


def test_registered_dataset_recomputes_folds_after_custom_strategy_closure_change() -> None:
    def make_split(mode: int):
        def split(self: FoldStrategy, inputs: dict[str, object], y: np.ndarray, n_splits: int, seed: int):
            del self, inputs, seed
            indices = np.roll(np.arange(y.shape[0]), mode)
            validation_blocks = np.array_split(indices, n_splits)
            folds = []
            for validation in validation_blocks:
                train = np.setdiff1d(indices, validation, assume_unique=True)
                folds.append((train, validation))
            return folds, {"method": "closure", "mode": mode}

        return split

    class ClosureFoldStrategy(FoldStrategy):
        pass

    ClosureFoldStrategy.split = make_split(0)  # type: ignore[method-assign]
    dataset = make_tabular_data(n_samples=48, n_features=4, seed=150)
    inputs = dataset.inputs()
    strategy = ClosureFoldStrategy()
    evaluator = RidgeEvaluator(
        n_splits=3,
        seed=150,
        max_configurations=1,
        refine_globals=False,
        fold_strategy=strategy,
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    evaluator.register_immutable_dataset(inputs, dataset.y)
    ClosureFoldStrategy.split = make_split(1)  # type: ignore[method-assign]

    result = evaluator.evaluate(build_seed_graph(), inputs, dataset.y, config=build_seed_graph().default_config())

    assert result.diagnostics["folds"]["mode"] == 1


def test_registered_dataset_recomputes_custom_folds_that_depend_on_module_state() -> None:
    class GlobalStateFoldStrategy(FoldStrategy):
        def split(self, inputs: dict[str, object], y: np.ndarray, n_splits: int, seed: int):
            del self, inputs, seed
            mode = _PERSISTENT_CACHE_CALLS["fold_mode"]
            indices = np.roll(np.arange(y.shape[0]), mode)
            validation_blocks = np.array_split(indices, n_splits)
            folds = []
            for validation in validation_blocks:
                train = np.setdiff1d(indices, validation, assume_unique=True)
                folds.append((train, validation))
            return folds, {"method": "global_state", "mode": mode}

    dataset = make_tabular_data(n_samples=48, n_features=4, seed=151)
    inputs = dataset.inputs()
    _PERSISTENT_CACHE_CALLS["fold_mode"] = 0
    evaluator = RidgeEvaluator(
        n_splits=3,
        seed=151,
        max_configurations=1,
        refine_globals=False,
        fold_strategy=GlobalStateFoldStrategy(),
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    evaluator.register_immutable_dataset(inputs, dataset.y)
    _PERSISTENT_CACHE_CALLS["fold_mode"] = 1

    result = evaluator.evaluate(build_seed_graph(), inputs, dataset.y, config=build_seed_graph().default_config())

    assert result.diagnostics["folds"]["mode"] == 1


def test_registered_fold_diagnostics_are_snapshot_isolated() -> None:
    dataset = make_tabular_data(n_samples=48, n_features=4, seed=152)
    inputs = {**dataset.inputs(), "group": np.repeat(np.arange(12), 4)}
    graph = build_seed_graph()
    evaluator = RidgeEvaluator(
        n_splits=3,
        seed=152,
        max_configurations=1,
        refine_globals=False,
        fold_strategy=FoldStrategy(name="group_random", group_key="group"),
        diagnostics_mode="basic",
        feature_pool_diagnostics=False,
    )
    evaluator.register_immutable_dataset(inputs, dataset.y)
    first = evaluator.evaluate(graph, inputs, dataset.y, config=graph.default_config())
    expected_counts = list(first.diagnostics["folds"]["validation_group_counts"])
    first.diagnostics["folds"]["validation_group_counts"][0] = 999

    second = evaluator.evaluate(graph, inputs, dataset.y, config=graph.default_config())

    assert second.diagnostics["folds"]["validation_group_counts"] == expected_counts


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


def test_ridge_g_skips_only_exact_normalized_weight_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1616)
    x = rng.normal(size=(72, 9))
    y = rng.normal(size=x.shape[0])
    evaluator = RidgeEvaluator(irls_steps=3)
    apply_calls = 0
    fit_calls = 0
    original_fit = evaluator_module.select_alpha_and_fit_ridge

    def constant_weights(residual: np.ndarray) -> np.ndarray:
        nonlocal apply_calls
        apply_calls += 1
        return np.full_like(residual, 7.0)

    def counted_fit(*args: object, **kwargs: object) -> tuple[float, object]:
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(evaluator_module, "select_alpha_and_fit_ridge", counted_fit)
    result = evaluator._fit_reweighted_ridge(x, y, None, ResidualWeightRule("constant", constant_weights))

    assert apply_calls == 3
    assert fit_calls == 1
    assert len(result.irls_iterations) == 3
    assert [row["step"] for row in result.irls_iterations] == [1, 2, 3]
    assert all(row["alpha"] == result.initial_alpha for row in result.irls_iterations)
    assert all(row["weight_min"] == row["weight_max"] == row["weight_mean"] == 1.0 for row in result.irls_iterations)
    assert all(row["weight_std"] == 0.0 for row in result.irls_iterations)
    np.testing.assert_array_equal(result.model.coef, result.initial_model.coef)


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


@pytest.mark.parametrize(
    ("node", "message"),
    [
        (NodeSpec("second_output", "output"), "add behavior as an alternative"),
        (NodeSpec("custom_fitter", "fitting"), "add behavior as an alternative"),
    ],
)
def test_mutation_documents_reject_new_output_and_fitting_nodes_early(node: NodeSpec, message: str) -> None:
    graph = build_structural_break_seed_graph()
    before = graph.to_dict()

    with pytest.raises(ValueError, match=message):
        MutationEngine().apply_document(graph, MutationDocument(nodes=(node,)))

    assert graph.to_dict() == before


def test_mutation_documents_reject_alternatives_on_input_nodes() -> None:
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        add=(MutationSpec(kind="add_alternative", target_node="series", alternative_id="ignored", primitive="identity_callable", parents=()),)
    )

    with pytest.raises(ValueError, match="cannot target input node"):
        MutationEngine().apply_document(graph, document)

    assert graph.nodes["series"].alternatives == []


def test_llm_document_validation_rejects_extra_output_node() -> None:
    graph = build_structural_break_seed_graph()
    document = MutationDocument(nodes=(NodeSpec("predictions", "output"),))
    engineer = LLMEngineerAgent(StaticLLMClient([]))

    with pytest.raises(ValueError, match="add behavior as an alternative"):
        engineer._validate_supported_document(graph, document)


def test_llm_document_validation_rejects_primitive_incompatible_with_target() -> None:
    graph = build_structural_break_seed_graph()
    document = MutationDocument(
        add=(MutationSpec(kind="add_alternative", target_node="output", alternative_id="weights", primitive="uniform_sample_weight", parents=()),)
    )
    engineer = LLMEngineerAgent(StaticLLMClient([]))

    with pytest.raises(ValueError, match="incompatible with output node"):
        engineer._validate_supported_document(graph, document)


def test_mutation_cannot_remove_last_paper_node_alternative() -> None:
    graph = build_structural_break_seed_graph()
    removals = tuple(RemoveSpec("output", alternative.id) for alternative in graph.nodes["output"].alternatives)

    with pytest.raises(ValueError, match="without alternatives"):
        MutationEngine().apply_document(graph, MutationDocument(remove=removals))


def test_registry_primitive_reintroduces_pruned_required_global() -> None:
    graph = build_structural_break_seed_graph()
    pruned = MutationEngine().apply_document(
        graph,
        MutationDocument(remove=(RemoveSpec("output", "projection", "test pruning"),)),
    ).graph
    assert "projection_vector" not in pruned.globals.names()

    restored = MutationEngine().apply(
        pruned,
        MutationSpec(
            kind="add_alternative",
            target_node="output",
            primitive="projection_outputs",
            alternative_id="projection_restored",
            parents=("segment_stats", "trend_stats", "shape_stats"),
            description="Reintroduce a projection after maintenance pruned its global.",
        ),
    )
    assert "projection_vector" in restored.globals.names()
    assert any(alternative.id == "projection_restored" for alternative in restored.nodes["output"].alternatives)


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


def test_block_style_machine_mutation_yaml_is_supported() -> None:
    document = MutationDocument.from_yaml(
        "\n".join(
            [
                'rationale: "block-style YAML from an LLM"',
                "hypotheses:",
                '  - "Temporal context should improve RUL features."',
                "globals:",
                '  - name: "residual_huber_scale_retry"',
                "    value: [1.35]",
                "    trainable: true",
                '    description: "Trainable Huber scale."',
                "nodes:",
                '  - name: "temporal_features_retry"',
                '    kind: "intermediate"',
                '    description: "Cycle-scaled features."',
                "remove:",
                "  []",
                "add:",
                '  - kind: "add_alternative"',
                '    target_node: "temporal_features_retry"',
                '    alternative_id: "cycle_scaling"',
                '    parents: ["base_features"]',
                '    primitive: "source"',
                '    source: "lambda ctx, values: FeatureBlock(values[\'base_features\'].data, values[\'base_features\'].feature_names)"',
                '    node_kind: "intermediate"',
                '    output_contract: {"type": "feature_block", "min_columns": 1, "differentiable": true}',
                '  - kind: "add_alternative"',
                '    target_node: "ridge_g"',
                '    alternative_id: "huber_trainable"',
                "    parents: []",
                '    primitive: "huber_residual_weight"',
                '    global_refs: ["residual_huber_scale_retry"]',
                '    description: "Huber weighting with a trainable scale parameter."',
            ]
        )
    )

    assert document.rationale == "block-style YAML from an LLM"
    assert document.hypotheses == ("Temporal context should improve RUL features.",)
    assert document.globals[0].name == "residual_huber_scale_retry"
    assert document.globals[0].value == [1.35]
    assert document.globals[0].trainable is True
    assert document.nodes[0].name == "temporal_features_retry"
    assert document.add[0].parents == ("base_features",)
    assert document.add[0].output_contract == {"type": "feature_block", "min_columns": 1, "differentiable": True}
    assert document.add[1].global_refs == ("residual_huber_scale_retry",)


def test_bare_yaml_language_tag_is_ignored_in_mutation_documents() -> None:
    document = MutationDocument.from_yaml(
        "\n".join(
            [
                "yaml",
                'rationale: "language tag without a markdown fence"',
                "hypotheses:",
                '  - "Use a compact primitive mutation."',
                "nodes:",
                "  []",
                "remove:",
                "  []",
                "globals:",
                "  []",
                "add:",
                '  - {"kind": "add_alternative", "target_node": "shape_stats", "primitive": "spectral_basic", "alternative_id": "spectral_yaml_tag", "parents": ["series"], "description": "frequency features"}',
            ]
        )
    )

    assert document.rationale == "language tag without a markdown fence"
    assert document.add[0].alternative_id == "spectral_yaml_tag"


def test_machine_mutation_yaml_accepts_stringified_parent_lists() -> None:
    document = MutationDocument.from_yaml(
        "\n".join(
            [
                'rationale: "stringified list fields from an LLM"',
                "hypotheses:",
                "  []",
                "nodes:",
                "  []",
                "remove:",
                "  []",
                "globals:",
                '  - name: "stringified_scale"',
                '    value: "[0.25]"',
                '    trainable: "false"',
                "add:",
                '  - kind: "add_alternative"',
                '    target_node: "shape_stats"',
                '    alternative_id: "spectral_stringified_parents"',
                '    primitive: "spectral_basic"',
                '    parents: "[\'series\']"',
            ]
        )
    )

    assert document.add[0].parents == ("series",)
    assert document.globals[0].value == [0.25]
    assert document.globals[0].trainable is False


def test_machine_mutation_yaml_reports_non_mapping_rows() -> None:
    with pytest.raises(ValueError, match="section 'add' item 0 must be a mapping"):
        MutationDocument.from_yaml(
            "\n".join(
                [
                    'rationale: "bad rows"',
                    "hypotheses:",
                    "  []",
                    "nodes:",
                    "  []",
                    "remove:",
                    "  []",
                    "globals:",
                    "  []",
                    "add:",
                    "  - not-a-mapping",
                ]
            )
        )


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


def test_paper_style_source_object_accepts_stringified_parent_lists() -> None:
    document = MutationDocument.from_yaml(
        "\n".join(
            [
                "add:",
                "  output:",
                '    - {"source": "lambda ctx, values: FeatureBlock(values[\'segment_stats\'].values[:, :1], [\'first_value\'])", "parents": "[\'segment_stats\']", "global_refs": "[\'gate_scale\']", "node_kind": "output"}',
            ]
        )
    )

    assert document.add[0].parents == ("segment_stats",)
    assert document.add[0].global_refs == ("gate_scale",)


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


def test_runtime_contract_rejects_wrong_callable_return_type() -> None:
    dataset = make_structural_break_data(n_series=20, length=50, seed=8)
    graph = build_structural_break_seed_graph()
    graph.nodes["activation"].alternatives[0].fn = lambda _ctx, _values: np.ones(20)

    with pytest.raises(TypeError, match="must return CallableFamily"):
        graph.evaluate_features(dataset.inputs())


def test_torch_contract_is_scoped_to_trainable_global_paths() -> None:
    graph = build_structural_break_seed_graph()
    graph.add_alternative(
        "output",
        "numpy_only_constant",
        ("series",),
        lambda _ctx, values: np.asarray(values["series"])[:, :1],
    )

    graph.validate_torch_trainable_paths()
    projection = next(alternative for alternative in graph.nodes["output"].alternatives if alternative.id == "projection")
    projection.torch_fn = None

    with pytest.raises(ValueError, match="trainable globals.*projection_vector.*output.projection"):
        graph.validate_torch_trainable_paths()


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
    projection = next(alternative for alternative in graph.nodes["output"].alternatives if alternative.id == "projection")
    projection.torch_fn = None
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
    agents = paper_test_agents()
    result = EvolutionLoop(
        graph,
        evaluator=RidgeEvaluator(n_splits=3, seed=21, max_configurations=12),
        scientist=agents.scientist,
        engineer=agents.engineer,
        memorandum_agent=agents.memorandum,
        seed=21,
    ).run(
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


def test_evolution_loop_requires_complete_llm_pipeline() -> None:
    with pytest.raises(TypeError, match="scientist.*engineer.*memorandum_agent"):
        EvolutionLoop(build_structural_break_seed_graph())  # type: ignore[call-arg]


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
        memorandum_agent=LLMMemorandumAgent(PaperTestClient()),
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
    hypotheses = (sample_hypothesis(),)
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
            memorandum_agent=LLMMemorandumAgent(PaperTestClient()),
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
    hypotheses = (sample_hypothesis(),)
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


def test_gemini_uses_api_key_header_not_url(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, *, timeout: float) -> object:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeHTTPResponse(b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}')

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    response = GeminiLLMClient("secret-key", "gemini-test", timeout_seconds=9.0).complete("system", "user")
    request = captured["request"]

    assert response == "ok"
    assert captured["timeout"] == 9.0
    assert "secret-key" not in request.full_url  # type: ignore[attr-defined]
    assert dict(request.header_items())["X-goog-api-key"] == "secret-key"  # type: ignore[attr-defined]


def test_llm_check_is_redacted_and_does_not_call_api(tmp_path, monkeypatch, capsys) -> None:
    for key in ("EVOFOREST_LLM_PROVIDER", "EVOFOREST_LLM_MODEL", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EVOFOREST_LLM_PROVIDER=gemini\nGEMINI_API_KEY=top-secret\nEVOFOREST_LLM_MODEL=gemini-test\n",
        encoding="utf-8",
    )

    assert cli_main(["llm-check", "--env-file", str(env_file)]) == 0
    output = capsys.readouterr().out
    assert '"ready": true' in output
    assert '"model": "gemini-test"' in output
    assert "top-secret" not in output


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
    hypothesis = (sample_hypothesis(),)
    source_system, source_user = PromptBuilder(allow_source=True).engineer_prompts(graph, result, hypothesis)
    _system, primitive_user = PromptBuilder(allow_source=False).engineer_prompts(graph, result, hypothesis)
    assert '"source": "lambda ctx, values:' in source_user
    assert 'add:\n  output:\n    - "lambda ctx, values:' in source_user
    assert '"output_contract": {"type": "feature_block"' in source_user
    assert '"torch_source": "lambda ctx, values:' in source_user
    assert "infer parents" in source_user
    assert "Prefer registry-backed primitives" in source_user
    assert "Avoid repeating rejected mutation signatures" in source_system
    assert "Never emit multiline code" in source_user
    assert "usually one add/remove" in source_user
    assert '"source": "lambda ctx, values:' not in primitive_user
    assert 'add:\n  output:\n    - "lambda ctx, values:' not in primitive_user


def test_llm_agents_can_disable_source_mutation_schema(tmp_path, monkeypatch) -> None:
    for key in ("EVOFOREST_LLM_PROVIDER", "EVOFOREST_LLM_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test-key\nEVOFOREST_LLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        llm_provider="openai",
        env_file=env_file,
        allow_source_mutations=False,
        disable_llm_source_mutations=True,
        llm_scientist_temperature=0.35,
        llm_island_temperatures=(),
        llm_engineer_temperature=0.0,
    )

    _scientist, engineer, _memorandum = build_llm_agents(
        args,
        registry=PrimitiveRegistry.for_task(TaskSchema.structural_break()),
    )

    assert engineer is not None
    assert engineer.allow_source is False
    dataset = make_structural_break_data(n_series=35, length=70, seed=25)
    graph = build_structural_break_seed_graph()
    result = RidgeEvaluator(n_splits=3, seed=25, max_configurations=4).evaluate(graph, dataset.inputs(), dataset.y)
    hypothesis = (sample_hypothesis(),)
    _system, user = engineer.prompt_builder.engineer_prompts(graph, result, hypothesis)
    assert '"source": "lambda ctx, values:' not in user
    assert "Source-backed lambda edits are disabled" in user


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
        memorandum_agent=LLMMemorandumAgent(PaperTestClient()),
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
        memorandum_agent=LLMMemorandumAgent(PaperTestClient()),
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
    class BadSourceEngineer:
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
    agents = paper_test_agents()
    result = EvolutionLoop(
        graph,
        evaluator=RidgeEvaluator(n_splits=3, seed=28, max_configurations=4),
        mutation_engine=MutationEngine(allow_source=True),
        scientist=agents.scientist,
        engineer=BadSourceEngineer(),  # type: ignore[arg-type]
        memorandum_agent=agents.memorandum,
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
    agents = paper_test_agents()
    result = EvolutionLoop(
        graph,
        evaluator=RidgeEvaluator(n_splits=3, seed=25, max_configurations=8),
        scientist=agents.scientist,
        engineer=agents.engineer,
        memorandum_agent=agents.memorandum,
        seed=25,
    ).run_islands(
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
    agents = paper_test_agents()
    result = EvolutionLoop(
        graph,
        evaluator=RidgeEvaluator(n_splits=3, seed=27, max_configurations=6),
        scientist=agents.scientist,
        engineer=agents.engineer,
        memorandum_agent=agents.memorandum,
        seed=27,
    ).run_async_islands(
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
