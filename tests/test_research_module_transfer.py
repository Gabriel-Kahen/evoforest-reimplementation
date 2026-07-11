from __future__ import annotations

import numpy as np

from benchmarks.research_suite.module_transfer import transfer_alternatives
from evoforest_arch.graph import EvalContext, FeatureBlock, Graph


def _identity(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
    return FeatureBlock(np.asarray(values["x"]), ["x"])


def _square(_ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
    values_array = np.asarray(values["x"])
    return FeatureBlock(values_array**2, ["x2"])


def _global_scale(ctx: EvalContext, values: dict[str, object]) -> FeatureBlock:
    return FeatureBlock(np.asarray(values["x"]) * ctx.globals.get("scale"), ["scaled"])


def _graph(*, include_feature: bool = True) -> Graph:
    graph = Graph()
    graph.add_input("x")
    if include_feature:
        graph.add_node("feature", "intermediate")
    return graph


def test_transfers_compatible_alternative_and_resets_search_statistics() -> None:
    source = _graph()
    source.add_alternative("feature", "square", ("x",), _square, primitive="square")
    source.nodes["feature"].alternatives[0].age = 12
    source.nodes["feature"].alternatives[0].stats = {"mean_importance": 0.8}
    target = _graph()
    target.add_alternative("feature", "identity", ("x",), _identity, primitive="identity")

    report = transfer_alternatives(source, target)

    assert report.transferred == ["feature.square"]
    assert report.skipped == []
    assert target.nodes["feature"].alternatives[-1] is not source.nodes["feature"].alternatives[0]
    assert target.nodes["feature"].alternatives[-1].age == 0
    assert target.nodes["feature"].alternatives[-1].stats == {}


def test_reports_missing_nodes_parents_and_parent_kind_mismatches() -> None:
    source = _graph()
    source.add_alternative("feature", "square", ("x",), _square)

    missing_node = _graph(include_feature=False)
    assert transfer_alternatives(source, missing_node).reasons == {"target_node_missing": 1}

    missing_parent = Graph()
    missing_parent.add_node("feature", "intermediate")
    assert transfer_alternatives(source, missing_parent).reasons == {"target_parent_missing": 1}

    wrong_parent_kind = Graph()
    wrong_parent_kind.add_node("x", "intermediate")
    wrong_parent_kind.add_alternative("x", "base", (), _identity)
    wrong_parent_kind.add_node("feature", "intermediate")
    assert transfer_alternatives(source, wrong_parent_kind).reasons == {"parent_kind_mismatch": 1}


def test_checks_globals_and_can_copy_missing_global_parameters() -> None:
    source = _graph()
    source.globals.add("scale", [2.0], trainable=True, description="Reusable scale")
    source.add_alternative("feature", "scaled", ("x",), _global_scale, global_refs=("scale",))

    target = _graph()
    blocked = transfer_alternatives(source, target)
    copied = transfer_alternatives(source, target, copy_missing_globals=True)

    assert blocked.reasons == {"target_global_missing": 1}
    assert copied.transferred == ["feature.scaled"]
    assert copied.copied_globals == ["scale"]
    np.testing.assert_array_equal(target.globals.get("scale"), np.array([2.0]))


def test_rejects_incompatible_global_shape_and_trainability() -> None:
    source = _graph()
    source.globals.add("scale", [2.0], trainable=True)
    source.add_alternative("feature", "scaled", ("x",), _global_scale, global_refs=("scale",))

    wrong_shape = _graph()
    wrong_shape.globals.add("scale", [1.0, 2.0], trainable=True)
    assert transfer_alternatives(source, wrong_shape).reasons == {"global_shape_mismatch": 1}

    wrong_trainability = _graph()
    wrong_trainability.globals.add("scale", [1.0], trainable=False)
    assert transfer_alternatives(source, wrong_trainability).reasons == {"global_trainability_mismatch": 1}


def test_copying_one_missing_global_does_not_hide_another_global_mismatch() -> None:
    source = _graph()
    source.globals.add("new_scale", [2.0])
    source.globals.add("scale", [2.0])
    source.add_alternative(
        "feature",
        "scaled",
        ("x",),
        _global_scale,
        global_refs=("new_scale", "scale"),
    )
    target = _graph()
    target.globals.add("scale", [1.0, 2.0])

    report = transfer_alternatives(source, target, copy_missing_globals=True)

    assert report.reasons == {"global_shape_mismatch": 1}
    assert "new_scale" not in target.globals.names()


def test_avoids_duplicate_ids_and_duplicate_semantics() -> None:
    source = _graph()
    source.add_alternative("feature", "square", ("x",), _square, primitive="square")

    same_id = _graph()
    same_id.add_alternative("feature", "square", ("x",), _identity, primitive="identity")
    assert transfer_alternatives(source, same_id).reasons == {"duplicate_id": 1}

    same_semantics = _graph()
    same_semantics.add_alternative("feature", "renamed", ("x",), _square, primitive="square")
    assert transfer_alternatives(source, same_semantics).reasons == {"duplicate_semantics": 1}


def test_rejects_an_alternative_that_would_create_a_target_cycle() -> None:
    source = Graph()
    source.add_input("x")
    source.add_node("a", "intermediate")
    source.add_node("b", "intermediate")
    source.add_alternative("b", "source_base", ("x",), _identity)
    source.add_alternative("a", "from_b", ("b",), _identity)

    target = Graph()
    target.add_input("x")
    target.add_node("a", "intermediate")
    target.add_node("b", "intermediate")
    target.add_alternative("a", "target_base", ("x",), _identity)
    target.add_alternative("b", "from_a", ("a",), _identity)

    report = transfer_alternatives(source, target, node_names={"a"})

    assert report.reasons["would_create_cycle"] == 1
    assert [alternative.id for alternative in target.nodes["a"].alternatives] == ["target_base"]


def test_node_and_alternative_filters_limit_candidates() -> None:
    source = _graph()
    source.add_alternative("feature", "identity", ("x",), _identity, tags=("base",))
    source.add_alternative("feature", "square", ("x",), _square, tags=("transfer",))
    target = _graph()

    report = transfer_alternatives(
        source,
        target,
        node_filter=lambda name, node: name == "feature" and node.kind == "intermediate",
        alternative_filter=lambda _name, alternative: "transfer" in alternative.tags,
    )

    assert report.transferred == ["feature.square"]
    assert report.reasons == {"alternative_filtered": 1}
    assert report.to_dict()["transferred"] == ["feature.square"]
