"""Small, deterministic regression baselines for controlled comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evoforest_arch.metrics import TaskScorer, coerce_scorer
from evoforest_arch.readout import DEFAULT_ALPHAS, RidgeModel, Standardizer, select_alpha_and_fit_ridge


@dataclass(frozen=True)
class BaselineEvaluation:
    """Predictions and both metric conventions for a fitted baseline."""

    predictions: np.ndarray
    score: float
    raw_score: float
    metric: str
    higher_is_better: bool


@dataclass
class _InputTransform:
    medians: np.ndarray
    standardizer: Standardizer

    @classmethod
    def fit(cls, x: np.ndarray) -> "_InputTransform":
        finite = np.where(np.isfinite(x), x, np.nan)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(finite, axis=0)
        medians = np.nan_to_num(medians, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.where(np.isfinite(x), x, medians)
        return cls(medians=medians, standardizer=Standardizer.fit(clean))

    def transform(self, x: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(x), x, self.medians)
        return self.standardizer.transform(clean)


def _as_features(x: np.ndarray, *, expected_columns: int | None = None) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"x must be a 2-D array, got shape {values.shape}.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("x must contain at least one row and one column.")
    if expected_columns is not None and values.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} input columns, got {values.shape[1]}.")
    return values


def _as_target(y: np.ndarray, n_rows: int) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != n_rows:
        raise ValueError(f"y must be a 1-D array with {n_rows} rows, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("y must contain only finite values.")
    return values


def _alphas(values: np.ndarray) -> np.ndarray:
    alphas = np.asarray(values, dtype=np.float64).reshape(-1)
    if alphas.size == 0 or not np.all(np.isfinite(alphas)) or np.any(alphas <= 0.0):
        raise ValueError("alphas must contain at least one finite positive value.")
    return alphas


@dataclass
class RawRidge:
    """Median-imputed, standardized Ridge regression on the raw inputs."""

    alphas: np.ndarray = field(default_factory=lambda: DEFAULT_ALPHAS.copy())
    input_transform_: _InputTransform | None = field(default=None, init=False, repr=False)
    model_: RidgeModel | None = field(default=None, init=False, repr=False)

    @property
    def selected_alpha(self) -> float:
        self._require_fitted()
        return float(self.model_.alpha)  # type: ignore[union-attr]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RawRidge":
        features = _as_features(x)
        target = _as_target(y, features.shape[0])
        transform = _InputTransform.fit(features)
        standardized = transform.transform(features)
        _, model = select_alpha_and_fit_ridge(standardized, target, _alphas(self.alphas))
        self.input_transform_ = transform
        self.model_ = model
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._require_fitted()
        transform = self.input_transform_
        model = self.model_
        assert transform is not None and model is not None
        features = _as_features(x, expected_columns=transform.medians.shape[0])
        return model.predict(transform.transform(features))

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        scorer: TaskScorer | str | None = None,
    ) -> BaselineEvaluation:
        predictions = self.predict(x)
        target = _as_target(y, predictions.shape[0])
        metric = coerce_scorer(scorer)
        return BaselineEvaluation(
            predictions=predictions,
            score=metric.score(target, predictions),
            raw_score=metric.raw_score(target, predictions),
            metric=metric.raw_name or metric.name,
            higher_is_better=metric.higher_is_better,
        )

    def _require_fitted(self) -> None:
        if self.input_transform_ is None or self.model_ is None:
            raise RuntimeError("Fit the baseline before predicting or evaluating.")


@dataclass
class RandomFeatureRidge(RawRidge):
    """Ridge on a deterministic random mixture of sine and tanh features.

    The projection is sampled once per fit from ``seed``. Raw standardized inputs
    are included by default so this baseline cannot lose the linear signal merely
    because of an unlucky random projection.
    """

    n_random_features: int = 256
    seed: int = 0
    include_linear: bool = True
    projection_: np.ndarray | None = field(default=None, init=False, repr=False)
    bias_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomFeatureRidge":
        features = _as_features(x)
        target = _as_target(y, features.shape[0])
        if self.n_random_features <= 0:
            raise ValueError("n_random_features must be positive.")

        transform = _InputTransform.fit(features)
        standardized = transform.transform(features)
        rng = np.random.default_rng(self.seed)
        projection = rng.normal(
            scale=1.0 / np.sqrt(standardized.shape[1]),
            size=(standardized.shape[1], self.n_random_features),
        )
        bias = rng.uniform(-np.pi, np.pi, size=self.n_random_features)
        expanded = self._expand(standardized, projection, bias)
        _, model = select_alpha_and_fit_ridge(expanded, target, _alphas(self.alphas))

        self.input_transform_ = transform
        self.projection_ = projection
        self.bias_ = bias
        self.model_ = model
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._require_fitted()
        transform = self.input_transform_
        projection = self.projection_
        bias = self.bias_
        model = self.model_
        assert transform is not None and projection is not None and bias is not None and model is not None
        features = _as_features(x, expected_columns=projection.shape[0])
        expanded = self._expand(transform.transform(features), projection, bias)
        return model.predict(expanded)

    def _require_fitted(self) -> None:
        super()._require_fitted()
        if self.projection_ is None or self.bias_ is None:
            raise RuntimeError("Fit the baseline before predicting or evaluating.")

    def _expand(self, x: np.ndarray, projection: np.ndarray, bias: np.ndarray) -> np.ndarray:
        latent = x @ projection + bias
        nonlinear = np.empty_like(latent)
        nonlinear[:, 0::2] = np.sin(latent[:, 0::2])
        nonlinear[:, 1::2] = np.tanh(latent[:, 1::2])
        if self.include_linear:
            return np.column_stack((x, nonlinear))
        return nonlinear
