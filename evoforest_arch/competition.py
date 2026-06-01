from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Any

import numpy as np


DEFAULT_COMPETITION_DATA_DIR = pathlib.Path("/Users/gabrielkahen/Downloads/data")
COMPETITION_DATASET_NAME = "competition-parquet-event"


@dataclass(frozen=True)
class CompetitionEventDataset:
    inputs: dict[str, object]
    y: np.ndarray
    metadata: dict[str, Any]


def load_competition_event_dataset(
    data_dir: str | pathlib.Path = DEFAULT_COMPETITION_DATA_DIR,
    *,
    split: str = "train",
    series_length: int = 160,
    max_samples: int | None = None,
) -> CompetitionEventDataset:
    """Load the Crunch-style parquet bundle as one event-detection sample per id.

    The current graph consumes one fixed-length sequence per sample. This loader
    maps each id into a fixed-length sequence by resampling period 1 into the
    pre-boundary half and period 2 into the post-boundary half. The label is
    whether the corresponding index file has an event (`tau_index >= 0`).
    """

    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    data_path = pathlib.Path(data_dir)
    if split == "train":
        x_path = data_path / "X_train.parquet"
        index_path = data_path / "y_train_index.parquet"
    elif split in {"reduced_test", "test_reduced"}:
        x_path = data_path / "X_test.reduced.parquet"
        index_path = data_path / "y_test_index.reduced.parquet"
    else:
        raise ValueError("split must be 'train' or 'reduced_test'.")
    _require_files(x_path, index_path)

    index_table = pq.read_table(index_path, columns=["tau_index", "tau", "id"])
    index_payload = index_table.to_pydict()
    ids = np.asarray(index_payload["id"], dtype=np.int64)
    tau_index = np.asarray(index_payload["tau_index"], dtype=np.int64)
    tau = np.asarray(index_payload["tau"], dtype=np.int64)
    order = np.argsort(ids)
    ids = ids[order]
    tau_index = tau_index[order]
    tau = tau[order]
    if max_samples is not None:
        max_count = max(1, int(max_samples))
        ids = ids[:max_count]
        tau_index = tau_index[:max_count]
        tau = tau[:max_count]

    x_dataset = ds.dataset(x_path, format="parquet")
    x_table = x_dataset.to_table(
        columns=["value", "period", "id", "time"],
        filter=pc.is_in(pc.field("id"), value_set=pa.array(ids)),
    )
    x_payload = x_table.to_pydict()
    x_ids = np.asarray(x_payload["id"], dtype=np.int64)
    periods = np.asarray(x_payload["period"], dtype=np.int64)
    times = np.asarray(x_payload["time"], dtype=np.int64)
    values = np.asarray(x_payload["value"], dtype=np.float64)
    series, missing_ids = _build_fixed_length_series(
        requested_ids=ids,
        x_ids=x_ids,
        periods=periods,
        times=times,
        values=values,
        series_length=series_length,
    )
    if missing_ids:
        raise ValueError(f"X parquet is missing rows for ids: {missing_ids[:5]}.")

    labels = (tau_index >= 0).astype(np.float64)
    boundary = int(series.shape[1] // 2)
    inputs: dict[str, object] = {
        "series": series,
        "boundary": boundary,
        "sample_id": ids,
    }
    metadata: dict[str, Any] = {
        "name": COMPETITION_DATASET_NAME,
        "split": split,
        "data_dir": str(data_path),
        "series_length": int(series_length),
        "n_samples": int(labels.shape[0]),
        "positive_count": int(np.sum(labels)),
        "negative_count": int(labels.shape[0] - np.sum(labels)),
        "ids_min": int(ids.min()) if ids.size else None,
        "ids_max": int(ids.max()) if ids.size else None,
        "max_samples": max_samples,
        "target_source": "tau_index >= 0 from y_train_index.parquet" if split == "train" else "tau_index >= 0 from y_test_index.reduced.parquet",
        "mapping": "period_1_resampled_to_pre_boundary_and_period_2_resampled_to_post_boundary",
        "official_metric_note": "This is an id-level event-detection surrogate for the graph interface, not the official row-level competition metric.",
        "tau_min": int(tau[tau >= 0].min()) if np.any(tau >= 0) else None,
        "tau_max": int(tau[tau >= 0].max()) if np.any(tau >= 0) else None,
    }
    return CompetitionEventDataset(inputs=inputs, y=labels, metadata=metadata)


def competition_data_summary(
    data_dir: str | pathlib.Path = DEFAULT_COMPETITION_DATA_DIR,
    *,
    max_samples: int | None = None,
    series_length: int = 160,
    include_reduced_test: bool = False,
) -> dict[str, Any]:
    train = load_competition_event_dataset(
        data_dir,
        split="train",
        max_samples=max_samples,
        series_length=series_length,
    )
    summary = {"train": train.metadata}
    data_path = pathlib.Path(data_dir)
    reduced_paths = [data_path / "X_test.reduced.parquet", data_path / "y_test_index.reduced.parquet"]
    if include_reduced_test and all(path.exists() for path in reduced_paths):
        reduced = load_competition_event_dataset(
            data_dir,
            split="reduced_test",
            max_samples=max_samples,
            series_length=series_length,
        )
        summary["reduced_test"] = reduced.metadata
    return summary


def _build_fixed_length_series(
    *,
    requested_ids: np.ndarray,
    x_ids: np.ndarray,
    periods: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    series_length: int,
) -> tuple[np.ndarray, list[int]]:
    if series_length < 4:
        raise ValueError("series_length must be at least 4.")
    boundary = int(series_length // 2)
    output = np.zeros((requested_ids.shape[0], series_length), dtype=np.float64)
    missing: list[int] = []
    if x_ids.size == 0:
        return output, [int(id_) for id_ in requested_ids]
    sort_order = np.lexsort((times, x_ids))
    sorted_ids = x_ids[sort_order]
    sorted_periods = periods[sort_order]
    sorted_times = times[sort_order]
    sorted_values = values[sort_order]
    unique_ids, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    ranges = {int(id_): (int(start), int(start + count)) for id_, start, count in zip(unique_ids, starts, counts)}
    for row_index, id_ in enumerate(requested_ids):
        bounds = ranges.get(int(id_))
        if bounds is None:
            missing.append(int(id_))
            continue
        start, stop = bounds
        id_periods = sorted_periods[start:stop]
        id_times = sorted_times[start:stop]
        id_values = sorted_values[start:stop]
        pre_values = id_values[id_periods == 1]
        pre_times = id_times[id_periods == 1]
        post_values = id_values[id_periods == 2]
        post_times = id_times[id_periods == 2]
        if pre_values.size == 0 and post_values.size == 0:
            missing.append(int(id_))
            continue
        if pre_values.size == 0:
            pre_values = post_values[:1]
            pre_times = post_times[:1]
        if post_values.size == 0:
            post_values = pre_values[-1:]
            post_times = pre_times[-1:]
        output[row_index, :boundary] = _resample_values(pre_values, pre_times, boundary)
        output[row_index, boundary:] = _resample_values(post_values, post_times, series_length - boundary)
    return output, missing


def _resample_values(values: np.ndarray, times: np.ndarray, length: int) -> np.ndarray:
    clean_values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if clean_values.size == 0:
        return np.zeros(length, dtype=np.float64)
    if clean_values.size == 1:
        return np.full(length, float(clean_values[0]), dtype=np.float64)
    clean_times = np.asarray(times, dtype=np.float64)
    order = np.argsort(clean_times)
    clean_times = clean_times[order]
    clean_values = clean_values[order]
    if float(clean_times[-1]) == float(clean_times[0]):
        return np.full(length, float(np.mean(clean_values)), dtype=np.float64)
    target = np.linspace(float(clean_times[0]), float(clean_times[-1]), length)
    return np.interp(target, clean_times, clean_values).astype(np.float64)


def _require_files(*paths: pathlib.Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing competition parquet files: {missing}")


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError("The competition parquet loader requires pyarrow. Install with `python -m pip install -e '.[parquet]'`.") from exc
