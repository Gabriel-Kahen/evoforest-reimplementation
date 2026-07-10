from __future__ import annotations

import numpy as np
import pytest

from evoforest_arch import readout as readout_module
from evoforest_arch.readout import EPS, _select_alpha_from_svd, _weighted_svd, select_alpha_and_fit_ridge


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
