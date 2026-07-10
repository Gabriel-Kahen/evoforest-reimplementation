from __future__ import annotations

import numpy as np
import pytest

from evoforest_arch import readout as readout_module
from evoforest_arch.readout import EPS, _select_alpha_from_svd, _weighted_svd, fit_ridge, fit_ridge_screening, select_alpha_and_fit_ridge


def _reference_select_alpha(decomp: object, y: np.ndarray, alphas: np.ndarray) -> float:
    best_alpha = float(alphas[0])
    best_mse = float("inf")
    spectral_power = decomp.s * decomp.s
    u_squared = decomp.u * decomp.u
    for alpha in alphas:
        shrinkage = spectral_power / (spectral_power + alpha)
        centered_pred = decomp.u @ (shrinkage * decomp.uy)
        pred = decomp.y_mean + centered_pred / np.maximum(decomp.sqrt_w, EPS)
        leverage = u_squared @ shrinkage
        residual = (y - pred) / np.maximum(1.0 - leverage, EPS)
        mse = float(np.average(residual**2, weights=decomp.weights))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha


@pytest.mark.parametrize("shape", [(96, 12), (24, 80)])
@pytest.mark.parametrize("weighted", [False, True])
def test_vectorized_alpha_selection_matches_reference(shape: tuple[int, int], weighted: bool) -> None:
    rng = np.random.default_rng(7000 + shape[0] + shape[1] + weighted)
    x = rng.normal(size=shape)
    y = rng.normal(size=shape[0])
    weights = rng.lognormal(sigma=1.5, size=shape[0]) if weighted else None
    alphas = np.array([1e-12, 1e-8, 1e-4, 1.0, 1e4, 1e12, 1e100])
    decomp = _weighted_svd(x, y, weights)

    assert _select_alpha_from_svd(decomp, y, alphas) == _reference_select_alpha(decomp, y, alphas)


@pytest.mark.parametrize("weighted", [False, True])
def test_vectorized_alpha_selection_matches_reference_for_rank_deficient_data(weighted: bool) -> None:
    rng = np.random.default_rng(7100 + weighted)
    base = rng.normal(size=(72, 5))
    x = np.column_stack((base, base[:, 0], base[:, 1] * 2.0, np.ones(base.shape[0])))
    y = rng.normal(size=x.shape[0])
    weights = np.linspace(0.0, 3.0, x.shape[0]) if weighted else None
    alphas = np.array([np.finfo(np.float64).tiny, 1e-16, 1e-8, 1.0, 1e16, 1e200])
    decomp = _weighted_svd(x, y, weights)

    assert _select_alpha_from_svd(decomp, y, alphas) == _reference_select_alpha(decomp, y, alphas)


def test_vectorized_alpha_selection_preserves_first_alpha_tie_break() -> None:
    rng = np.random.default_rng(7200)
    x = rng.normal(size=(40, 8))
    y = np.full(x.shape[0], 3.0)
    alphas = np.array([0.25, 1.0, 4.0])
    decomp = _weighted_svd(x, y, None)

    selected = _select_alpha_from_svd(decomp, y, alphas)

    assert selected == alphas[0]


def test_vectorized_selection_keeps_fitted_model_equivalent_to_selected_svd_model() -> None:
    rng = np.random.default_rng(7300)
    x = rng.normal(size=(84, 18))
    y = rng.normal(size=x.shape[0])
    weights = rng.uniform(0.1, 2.5, size=x.shape[0])
    alphas = np.array([1e-12, 1e-4, 1.0, 1e4, 1e100])
    decomp = _weighted_svd(x, y, weights)
    expected_alpha = _reference_select_alpha(decomp, y, alphas)

    alpha, model = select_alpha_and_fit_ridge(x, y, alphas, weights)

    assert alpha == expected_alpha
    expected_coef = decomp.vt.T @ ((decomp.s / (decomp.s * decomp.s + alpha)) * decomp.uy)
    expected_intercept = decomp.y_mean - float(decomp.x_mean @ expected_coef)
    np.testing.assert_allclose(model.coef, expected_coef, rtol=1e-13, atol=1e-13)
    assert model.intercept == pytest.approx(expected_intercept, rel=1e-13, abs=1e-13)


def test_vectorized_alpha_selection_chunks_large_working_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7400)
    x = rng.normal(size=(96, 12))
    y = rng.normal(size=x.shape[0])
    alphas = np.logspace(-8, 8, 37)
    decomp = _weighted_svd(x, y, None)
    monkeypatch.setattr(readout_module, "_ALPHA_SELECTION_MAX_ELEMENTS", 400)

    selected = _select_alpha_from_svd(decomp, y, alphas)

    assert selected == _reference_select_alpha(decomp, y, alphas)


@pytest.mark.parametrize("explicit_weights", [False, True])
def test_uniform_weight_svd_fast_path_is_bitwise_equivalent(explicit_weights: bool) -> None:
    rng = np.random.default_rng(7500 + explicit_weights)
    x = rng.normal(size=(91, 17))
    y = rng.normal(size=x.shape[0])
    sample_weight = np.full(x.shape[0], 3.5) if explicit_weights else None
    weights = np.ones(x.shape[0], dtype=np.float64)
    x_mean = np.average(x, axis=0, weights=weights)
    y_mean = float(np.average(y, weights=weights))
    xc = x - x_mean
    yc = y - y_mean
    sqrt_w = np.sqrt(weights)
    u, s, vt = np.linalg.svd(xc * sqrt_w[:, None], full_matrices=False)

    actual = _weighted_svd(x, y, sample_weight)

    np.testing.assert_array_equal(actual.u, u)
    np.testing.assert_array_equal(actual.s, s)
    np.testing.assert_array_equal(actual.vt, vt)
    np.testing.assert_array_equal(actual.uy, u.T @ (yc * sqrt_w))
    np.testing.assert_array_equal(actual.x_mean, x_mean)
    assert actual.y_mean == y_mean


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("rank_deficient", [False, True])
def test_screening_normal_equation_matches_svd(weighted: bool, rank_deficient: bool) -> None:
    rng = np.random.default_rng(7600 + 10 * weighted + rank_deficient)
    x = rng.normal(size=(120, 18))
    if rank_deficient:
        x[:, -3:] = x[:, :3]
    y = rng.normal(size=x.shape[0])
    weights = rng.lognormal(sigma=0.7, size=x.shape[0]) if weighted else None

    expected = fit_ridge(x, y, 1.0, weights)
    actual = fit_ridge_screening(x, y, 1.0, weights)

    np.testing.assert_allclose(actual.coef, expected.coef, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual.predict(x), expected.predict(x), rtol=2e-12, atol=2e-12)
    assert actual.intercept == pytest.approx(expected.intercept, rel=2e-12, abs=2e-12)


def test_screening_normal_equation_falls_back_for_unsafe_alpha() -> None:
    rng = np.random.default_rng(7700)
    x = rng.normal(size=(80, 14))
    y = rng.normal(size=x.shape[0])
    alpha = 1e-14

    expected = fit_ridge(x, y, alpha)
    actual = fit_ridge_screening(x, y, alpha)

    np.testing.assert_array_equal(actual.coef, expected.coef)
    assert actual.intercept == expected.intercept


def test_screening_normal_equation_falls_back_when_solve_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7800)
    x = rng.normal(size=(70, 11))
    y = rng.normal(size=x.shape[0])
    expected = fit_ridge(x, y, 1.0)

    def fail_solve(_matrix: np.ndarray, _rhs: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("synthetic failure")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    actual = fit_ridge_screening(x, y, 1.0)

    np.testing.assert_array_equal(actual.coef, expected.coef)
    assert actual.intercept == expected.intercept


def test_screening_normal_equation_falls_back_for_nonfinite_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7850)
    x = rng.normal(size=(70, 11))
    y = rng.normal(size=x.shape[0])
    expected = fit_ridge(x, y, 1.0)

    monkeypatch.setattr(np.linalg, "solve", lambda matrix, _rhs: np.full(matrix.shape[0], np.nan))
    actual = fit_ridge_screening(x, y, 1.0)

    np.testing.assert_array_equal(actual.coef, expected.coef)
    assert actual.intercept == expected.intercept


def test_screening_normal_equation_falls_back_for_very_wide_data(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7875)
    x = rng.normal(size=(20, 70))
    y = rng.normal(size=x.shape[0])
    expected = fit_ridge(x, y, 1.0)

    monkeypatch.setattr(np.linalg, "solve", lambda _matrix, _rhs: pytest.fail("normal equations should not run"))
    actual = fit_ridge_screening(x, y, 1.0)

    np.testing.assert_array_equal(actual.coef, expected.coef)
    assert actual.intercept == expected.intercept


def test_screening_normal_equation_preserves_a_near_tie_ordering() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 12))
    y = x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=0.2, size=x.shape[0])
    validation_x = rng.normal(size=(50, 12))
    validation_y = validation_x[:, 0] - 0.3 * validation_x[:, 1] + rng.normal(scale=0.2, size=validation_x.shape[0])
    perturbed_x = x.copy()
    perturbed_validation_x = validation_x.copy()
    perturbed_x[:, 2] += 1e-8 * rng.normal(size=x.shape[0])
    perturbed_validation_x[:, 2] += 1e-8 * rng.normal(size=validation_x.shape[0])
    candidates = [(x, validation_x), (perturbed_x, perturbed_validation_x)]

    svd_losses = [np.mean((validation_y - fit_ridge(train, y, 1.0).predict(valid)) ** 2) for train, valid in candidates]
    fast_losses = [np.mean((validation_y - fit_ridge_screening(train, y, 1.0).predict(valid)) ** 2) for train, valid in candidates]

    assert abs(svd_losses[0] - svd_losses[1]) < 1e-10
    assert int(np.argmin(fast_losses)) == int(np.argmin(svd_losses))
