from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Any

import numpy as np


DEFAULT_COMPETITION_DATA_DIR = pathlib.Path("/Users/gabrielkahen/Downloads/data")
COMPETITION_DATASET_NAME = "competition-parquet-event"
COMPETITION_ROW_DATASET_NAME = "competition-parquet-row"


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
        "read_paths": [str(x_path), str(index_path)],
        "mapping": "period_1_resampled_to_pre_boundary_and_period_2_resampled_to_post_boundary",
        "official_metric_note": "This is an id-level event-detection surrogate for the graph interface, not the official row-level competition metric.",
        "tau_min": int(tau[tau >= 0].min()) if np.any(tau >= 0) else None,
        "tau_max": int(tau[tau >= 0].max()) if np.any(tau >= 0) else None,
    }
    return CompetitionEventDataset(inputs=inputs, y=labels, metadata=metadata)


def load_competition_row_dataset(
    data_dir: str | pathlib.Path = DEFAULT_COMPETITION_DATA_DIR,
    *,
    split: str = "train",
    series_length: int = 160,
    max_samples: int | None = None,
    max_ids: int | None = None,
    max_rows_per_id: int | None = None,
    row_stride: int = 1,
) -> CompetitionEventDataset:
    """Load the parquet bundle at the official `(id, time) -> target` granularity.

    Each labeled target row becomes one graph sample. The graph input sequence
    preserves the architecture's boundary semantics: period 1 is resampled into
    the pre-boundary half, and the period-2 prefix available at the target time
    is resampled into the post-boundary half. No target labels, tau indices,
    future X rows, or reduced-test files are used for the train split.
    """

    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    data_path = pathlib.Path(data_dir)
    if split == "train":
        x_path = data_path / "X_train.parquet"
        y_path = data_path / "y_train.parquet"
        target_source = "target from y_train.parquet"
    elif split in {"reduced_test", "test_reduced"}:
        x_path = data_path / "X_test.reduced.parquet"
        y_path = data_path / "y_test.reduced.parquet"
        target_source = "target from y_test.reduced.parquet"
    else:
        raise ValueError("split must be 'train' or 'reduced_test'.")
    _require_files(x_path, y_path)

    row_stride = max(1, int(row_stride))
    if series_length < 4:
        raise ValueError("series_length must be at least 4.")

    y_table = pq.read_table(y_path, columns=["target", "id", "time"])
    y_ids = y_table.column("id").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    y_times = y_table.column("time").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    y_targets = y_table.column("target").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    if y_ids.size == 0:
        raise ValueError(f"No target rows found in {y_path}.")

    y_order = np.lexsort((y_times, y_ids))
    y_ids = y_ids[y_order]
    y_times = y_times[y_order]
    y_targets = y_targets[y_order]
    selected_row_indices = _select_target_rows(
        y_ids,
        row_stride=row_stride,
        max_ids=max_ids,
        max_rows_per_id=max_rows_per_id,
        max_samples=max_samples,
    )
    target_ids = y_ids[selected_row_indices]
    target_times = y_times[selected_row_indices]
    labels = y_targets[selected_row_indices].astype(np.float64, copy=False)
    requested_ids = np.unique(target_ids)

    x_dataset = ds.dataset(x_path, format="parquet")
    x_table = x_dataset.to_table(
        columns=["value", "period", "id", "time"],
        filter=pc.is_in(pc.field("id"), value_set=pa.array(requested_ids)),
    )
    x_ids = x_table.column("id").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    periods = x_table.column("period").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    times = x_table.column("time").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    values = x_table.column("value").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    series, observed_lengths, sample_periods, missing_ids = _build_row_lookback_series(
        target_ids=target_ids,
        target_times=target_times,
        x_ids=x_ids,
        periods=periods,
        times=times,
        values=values,
        series_length=series_length,
    )
    if missing_ids:
        raise ValueError(f"X parquet is missing rows for ids: {missing_ids[:5]}.")

    boundary = int(series.shape[1] // 2)
    inputs: dict[str, object] = {
        "series": series,
        "boundary": boundary,
        "sample_id": target_ids,
        "sample_time": target_times,
        "sample_period": sample_periods,
        "lookback_observed": observed_lengths,
        "sample_time_scale": np.full(labels.shape[0], max(float(np.max(target_times)), 1.0), dtype=np.float64),
    }
    positive_count = int(np.sum(labels >= 0.5))
    unique_ids = np.unique(target_ids)
    metadata: dict[str, Any] = {
        "name": COMPETITION_ROW_DATASET_NAME,
        "split": split,
        "data_dir": str(data_path),
        "series_length": int(series_length),
        "n_samples": int(labels.shape[0]),
        "n_ids": int(unique_ids.shape[0]),
        "positive_count": positive_count,
        "negative_count": int(labels.shape[0] - positive_count),
        "ids_min": int(unique_ids.min()) if unique_ids.size else None,
        "ids_max": int(unique_ids.max()) if unique_ids.size else None,
        "max_samples": max_samples,
        "max_ids": max_ids,
        "max_rows_per_id": max_rows_per_id,
        "row_stride": int(row_stride),
        "target_source": target_source,
        "read_paths": [str(x_path), str(y_path)],
        "mapping": "period_1_reference_resampled_to_pre_boundary_and_causal_period_2_prefix_resampled_to_post_boundary",
        "split_policy": "Use grouped split manifests keyed by sample_id; rows from one id must stay in one split.",
        "reduced_test_policy": "Reduced labeled test parquet files are read only when split='reduced_test' is explicitly requested.",
        "official_metric_note": "This is row/time-level target evaluation on y_*.parquet rows; held-out validation should be grouped by id.",
        "time_min": int(target_times.min()) if target_times.size else None,
        "time_max": int(target_times.max()) if target_times.size else None,
    }
    return CompetitionEventDataset(inputs=inputs, y=labels, metadata=metadata)


def competition_data_summary(
    data_dir: str | pathlib.Path = DEFAULT_COMPETITION_DATA_DIR,
    *,
    max_samples: int | None = None,
    series_length: int = 160,
    include_reduced_test: bool = False,
    row_level: bool = False,
    max_ids: int | None = None,
    max_rows_per_id: int | None = None,
    row_stride: int = 1,
) -> dict[str, Any]:
    loader = load_competition_row_dataset if row_level else load_competition_event_dataset
    loader_kwargs: dict[str, Any] = {"max_samples": max_samples, "series_length": series_length}
    if row_level:
        loader_kwargs.update({"max_ids": max_ids, "max_rows_per_id": max_rows_per_id, "row_stride": row_stride})
    train = loader(data_dir, split="train", **loader_kwargs)
    summary = {"train": train.metadata}
    data_path = pathlib.Path(data_dir)
    reduced_paths = (
        [data_path / "X_test.reduced.parquet", data_path / "y_test.reduced.parquet"]
        if row_level
        else [data_path / "X_test.reduced.parquet", data_path / "y_test_index.reduced.parquet"]
    )
    if include_reduced_test and all(path.exists() for path in reduced_paths):
        reduced = loader(data_dir, split="reduced_test", **loader_kwargs)
        summary["reduced_test"] = reduced.metadata
    return summary


def _select_target_rows(
    sorted_ids: np.ndarray,
    *,
    row_stride: int,
    max_ids: int | None,
    max_rows_per_id: int | None,
    max_samples: int | None,
) -> np.ndarray:
    unique_ids, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    if max_ids is not None:
        unique_ids = unique_ids[: max(1, int(max_ids))]
        starts = starts[: unique_ids.shape[0]]
        counts = counts[: unique_ids.shape[0]]
    selected_parts: list[np.ndarray] = []
    for start, count in zip(starts, counts):
        indices = np.arange(int(start), int(start + count), int(row_stride), dtype=np.int64)
        if max_rows_per_id is not None and indices.shape[0] > int(max_rows_per_id):
            take = np.linspace(0, indices.shape[0] - 1, max(1, int(max_rows_per_id)), dtype=np.int64)
            indices = indices[take]
        selected_parts.append(indices)
    if not selected_parts:
        return np.asarray([], dtype=np.int64)
    selected = np.concatenate(selected_parts)
    if max_samples is not None and selected.shape[0] > int(max_samples):
        take = np.linspace(0, selected.shape[0] - 1, max(1, int(max_samples)), dtype=np.int64)
        selected = selected[take]
    return np.asarray(selected, dtype=np.int64)


def _build_row_lookback_series(
    *,
    target_ids: np.ndarray,
    target_times: np.ndarray,
    x_ids: np.ndarray,
    periods: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    series_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    n_rows = int(target_ids.shape[0])
    boundary = int(series_length // 2)
    output = np.zeros((n_rows, int(series_length)), dtype=np.float32)
    observed_lengths = np.zeros(n_rows, dtype=np.float64)
    sample_periods = np.zeros(n_rows, dtype=np.float64)
    missing: list[int] = []
    if n_rows == 0:
        return output, observed_lengths, sample_periods, missing
    if x_ids.size == 0:
        return output, observed_lengths, sample_periods, [int(id_) for id_ in np.unique(target_ids)]

    x_order = np.lexsort((times, x_ids))
    sorted_x_ids = x_ids[x_order]
    sorted_periods = periods[x_order]
    sorted_times = times[x_order]
    sorted_values = np.nan_to_num(values[x_order], nan=0.0, posinf=0.0, neginf=0.0)
    unique_x_ids, x_starts, x_counts = np.unique(sorted_x_ids, return_index=True, return_counts=True)
    x_ranges = {int(id_): (int(start), int(start + count)) for id_, start, count in zip(unique_x_ids, x_starts, x_counts)}

    unique_target_ids, target_starts, target_counts = np.unique(target_ids, return_index=True, return_counts=True)
    for id_, target_start, target_count in zip(unique_target_ids, target_starts, target_counts):
        bounds = x_ranges.get(int(id_))
        if bounds is None:
            missing.append(int(id_))
            continue
        x_start, x_stop = bounds
        id_times = sorted_times[x_start:x_stop]
        id_values = sorted_values[x_start:x_stop]
        id_periods = sorted_periods[x_start:x_stop]
        row_start = int(target_start)
        row_stop = int(target_start + target_count)
        target_slice = target_times[row_start:row_stop]
        positions = np.searchsorted(id_times, target_slice, side="right")
        period1_mask = id_periods == 1
        period2_mask = id_periods == 2
        period1_times = id_times[period1_mask]
        period1_values = id_values[period1_mask]
        period2_times = id_times[period2_mask]
        period2_values = id_values[period2_mask]
        for offset, position in enumerate(positions):
            row_index = row_start + offset
            stop = int(position)
            if stop <= 0:
                fill_value = float(id_values[0]) if id_values.size else 0.0
                output[row_index, :] = fill_value
                sample_periods[row_index] = float(id_periods[0]) if id_periods.size else 0.0
                continue
            target_time = target_slice[offset]
            sample_periods[row_index] = float(id_periods[stop - 1])
            pre_mask = period1_times <= target_time
            post_mask = period2_times <= target_time
            pre_values = period1_values[pre_mask]
            pre_times = period1_times[pre_mask]
            post_values = period2_values[post_mask]
            post_times = period2_times[post_mask]
            if pre_values.size == 0 and stop > 0:
                pre_values = id_values[:1]
                pre_times = id_times[:1]
            if post_values.size == 0:
                post_values = pre_values[-1:] if pre_values.size else id_values[max(0, stop - 1) : stop]
                post_times = np.asarray([target_time], dtype=np.int64)
            observed_lengths[row_index] = float(pre_values.shape[0] + post_values.shape[0])
            output[row_index, :boundary] = _resample_values(pre_values, pre_times, boundary)
            output[row_index, boundary:] = _resample_values(post_values, post_times, int(series_length) - boundary)
    return output, observed_lengths, sample_periods, missing


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
