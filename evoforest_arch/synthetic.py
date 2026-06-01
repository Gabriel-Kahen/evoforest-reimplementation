from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeSeriesDataset:
    values: np.ndarray
    y: np.ndarray
    boundary: int

    def inputs(self) -> dict[str, object]:
        return {"series": self.values, "boundary": self.boundary}


def make_structural_break_data(
    n_series: int = 240,
    length: int = 160,
    boundary: int | None = None,
    seed: int = 0,
) -> TimeSeriesDataset:
    rng = np.random.default_rng(seed)
    boundary = int(boundary if boundary is not None else length // 2)
    y = np.zeros(n_series, dtype=np.float64)
    y[: n_series // 2] = 1.0
    rng.shuffle(y)
    values = np.empty((n_series, length), dtype=np.float64)
    time = np.linspace(-1.0, 1.0, length)
    for idx, label in enumerate(y):
        noise = rng.normal(0.0, 0.35, size=length)
        drift = rng.normal(0.0, 0.05) * time
        series = noise + drift
        if label == 1.0:
            kind = idx % 4
            if kind == 0:
                series[boundary:] += rng.normal(0.7, 0.08)
            elif kind == 1:
                series[boundary:] *= rng.uniform(1.8, 2.3)
            elif kind == 2:
                series[boundary:] += np.linspace(0.0, rng.uniform(0.8, 1.2), length - boundary)
            else:
                series[boundary:] += 0.45 * np.sin(np.linspace(0.0, 7.0, length - boundary))
        values[idx] = series
    return TimeSeriesDataset(values=values, y=y, boundary=boundary)
