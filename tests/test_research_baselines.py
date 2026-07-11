from __future__ import annotations

import numpy as np
import pytest

from benchmarks.research_suite.baselines import RandomFeatureRidge, RawRidge


def _linear_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(120, 4))
    y = 2.0 * x[:, 0] - 0.75 * x[:, 1] + 0.1 * rng.normal(size=x.shape[0])
    return x, y


def test_raw_ridge_fit_predict_and_evaluate() -> None:
    x, y = _linear_data()
    model = RawRidge(alphas=np.array([1e-4, 1e-2, 1.0])).fit(x[:80], y[:80])

    predictions = model.predict(x[80:])
    evaluation = model.evaluate(x[80:], y[80:], scorer="rmse")

    assert predictions.shape == (40,)
    np.testing.assert_array_equal(evaluation.predictions, predictions)
    assert evaluation.metric == "rmse"
    assert evaluation.score == pytest.approx(-evaluation.raw_score)
    assert evaluation.raw_score < 0.2
    assert model.selected_alpha in {1e-4, 1e-2, 1.0}


def test_preprocessing_uses_train_statistics_and_handles_nonfinite_inputs() -> None:
    x, y = _linear_data()
    x[0, 0] = np.nan
    x[1, 1] = np.inf
    model = RawRidge().fit(x[:80], y[:80])

    test_x = x[80:].copy()
    test_x[0, 0] = np.nan
    test_x[1, 1] = -np.inf
    predictions = model.predict(test_x)

    assert np.all(np.isfinite(predictions))
    assert model.input_transform_ is not None
    assert model.input_transform_.medians[0] == pytest.approx(np.nanmedian(x[:80, 0]))


def test_random_feature_ridge_is_deterministic_for_a_seed() -> None:
    x, y = _linear_data()
    left = RandomFeatureRidge(n_random_features=31, seed=42).fit(x[:90], y[:90])
    right = RandomFeatureRidge(n_random_features=31, seed=42).fit(x[:90], y[:90])

    np.testing.assert_array_equal(left.projection_, right.projection_)
    np.testing.assert_array_equal(left.bias_, right.bias_)
    np.testing.assert_array_equal(left.predict(x[90:]), right.predict(x[90:]))


def test_random_feature_seed_changes_the_map() -> None:
    x, y = _linear_data()
    left = RandomFeatureRidge(n_random_features=16, seed=1).fit(x, y)
    right = RandomFeatureRidge(n_random_features=16, seed=2).fit(x, y)

    assert not np.array_equal(left.projection_, right.projection_)


@pytest.mark.parametrize("model", [RawRidge(), RandomFeatureRidge(n_random_features=8)])
def test_predict_before_fit_is_rejected(model: object) -> None:
    with pytest.raises(RuntimeError, match="Fit the baseline"):
        model.predict(np.ones((3, 2)))


def test_invalid_training_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="2-D"):
        RawRidge().fit(np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="1-D"):
        RawRidge().fit(np.ones((5, 2)), np.ones((5, 1)))
    with pytest.raises(ValueError, match="positive"):
        RandomFeatureRidge(n_random_features=0).fit(np.ones((5, 2)), np.ones(5))
