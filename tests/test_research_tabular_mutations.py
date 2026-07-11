from __future__ import annotations

import numpy as np

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.mutations import MutationEngine, built_in_mutations
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.synthetic import make_tabular_data


def test_research_tabular_mutations_expand_multiple_search_axes() -> None:
    graph = build_seed_graph()
    engine = MutationEngine()
    specs = [
        spec
        for spec in built_in_mutations()
        if spec.primitive in {"tabular_signed_log", "tabular_sine_interactions", "tabular_quantile_summary", "tabular_row_norm_weight"}
    ]

    for spec in specs:
        graph = engine.apply(graph, spec)

    data = make_tabular_data(n_samples=48, n_features=6, seed=91)
    result = RidgeEvaluator(n_splits=2, max_configurations=4, refine_globals=False).evaluate(graph, data.inputs(), data.y)

    assert len(specs) == 4
    assert all(any(alt.primitive == spec.primitive for alt in graph.nodes[spec.target_node].alternatives) for spec in specs)
    assert np.isfinite(result.score)
