from __future__ import annotations

import numpy as np

from benchmarks.research_suite.evoforest_model import fit_frozen_evoforest_regressor
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.synthetic import make_tabular_data


def test_frozen_evoforest_regressor_predicts_unseen_rows() -> None:
    dataset = make_tabular_data(n_samples=100, n_features=6, seed=801)
    train = np.arange(70)
    test = np.arange(70, 100)
    model = fit_frozen_evoforest_regressor(
        build_seed_graph(),
        {},
        {"x": dataset.x[train]},
        dataset.y[train],
    )

    predictions = model.predict({"x": dataset.x[test]})

    assert predictions.shape == (30,)
    assert np.all(np.isfinite(predictions))
    assert model.score({"x": dataset.x[test]}, dataset.y[test], "rmse") >= 0.0


def test_frozen_evoforest_regressor_does_not_reuse_test_targets() -> None:
    dataset = make_tabular_data(n_samples=90, n_features=5, seed=802)
    model = fit_frozen_evoforest_regressor(
        build_seed_graph(),
        {},
        {"x": dataset.x[:60]},
        dataset.y[:60],
    )

    first = model.predict({"x": dataset.x[60:]})
    second = model.predict({"x": dataset.x[60:]})

    np.testing.assert_allclose(first, second)
