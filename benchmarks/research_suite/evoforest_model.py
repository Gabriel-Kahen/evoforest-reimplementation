from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.graph import Graph
from evoforest_arch.metrics import TaskScorer, coerce_scorer
from evoforest_arch.readout import RidgeModel, Standardizer


@dataclass(frozen=True)
class FrozenEvoForestRegressor:
    """A train-fitted EvoForest readout for genuinely held-out prediction."""

    graph: Graph
    config: dict[str, str]
    feature_names: tuple[str, ...]
    standardizer: Standardizer
    readout: RidgeModel

    def predict(self, inputs: dict[str, object]) -> np.ndarray:
        features, names, _ = self.graph.evaluate_features(inputs, config=self.config)
        if tuple(names) != self.feature_names:
            raise ValueError("Held-out feature schema does not match the fitted graph schema.")
        return self.readout.predict(self.standardizer.transform(features))

    def score(
        self,
        inputs: dict[str, object],
        y: np.ndarray,
        scorer: TaskScorer | str | None = "rmse",
    ) -> float:
        return coerce_scorer(scorer).raw_score(np.asarray(y, dtype=np.float64), self.predict(inputs))


def fit_frozen_evoforest_regressor(
    graph: Graph,
    config: dict[str, str],
    inputs: dict[str, object],
    y: np.ndarray,
    *,
    evaluator: RidgeEvaluator | None = None,
) -> FrozenEvoForestRegressor:
    """Fit the final readout on training data without touching future holdouts.

    The graph and its persistent globals are cloned and treated as frozen. Any global
    refinement or graph selection must therefore happen before this function is called.
    Sample- and residual-weighting alternatives are fitted on training rows only.
    """

    frozen_graph = graph.clone()
    selected = frozen_graph.selected_config(config)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    features, names, ctx = frozen_graph.evaluate_features(inputs, config=selected)
    if features.shape[0] != target.shape[0]:
        raise ValueError("Training features and target must have the same row count.")

    ridge_evaluator = evaluator or RidgeEvaluator(refine_globals=False, diagnostics_mode="basic")
    sample_weight, residual_rule, _ = ridge_evaluator._evaluate_fitting_rules(
        frozen_graph,
        inputs,
        target,
        selected,
        ctx,
    )
    standardizer = Standardizer.fit(features)
    prepared = standardizer.transform(features)
    fitted = ridge_evaluator._fit_reweighted_ridge(prepared, target, sample_weight, residual_rule)
    return FrozenEvoForestRegressor(
        graph=frozen_graph,
        config=selected,
        feature_names=tuple(names),
        standardizer=standardizer,
        readout=fitted.model,
    )
