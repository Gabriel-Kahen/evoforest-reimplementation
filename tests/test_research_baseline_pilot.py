from __future__ import annotations

from dataclasses import asdict

import numpy as np

from benchmarks.research_suite import baseline_pilot
from benchmarks.research_suite.optional_baselines import CapabilityStatus


class _MeanEstimator:
    def __init__(self, **_kwargs):
        self.mean = 0.0

    def fit(self, _x, y):
        self.mean = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean)


def test_baseline_pilot_method_rows_and_failures_are_explicit(monkeypatch) -> None:
    original = baseline_pilot.HistGradientBoostingAdapter
    monkeypatch.setattr(baseline_pilot, "HistGradientBoostingAdapter", lambda **kwargs: original(estimator_factory=_MeanEstimator, **kwargs))
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = np.arange(8, dtype=float)

    baseline_pilot._run_methods("task", "test", X, y, X, y, 1, rows, failures)

    assert rows
    assert all("wall_time_seconds" in row for row in rows)
    assert isinstance(asdict(CapabilityStatus("x", True, "y", "z")), dict)
