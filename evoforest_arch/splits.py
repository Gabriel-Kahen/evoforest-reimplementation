from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SplitManifest:
    dataset_fingerprint: str
    n_samples: int
    seed: int
    method: str
    validation_fraction: float
    test_fraction: float
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    group_key: str | None = None
    train_groups: tuple[int, ...] = ()
    validation_groups: tuple[int, ...] = ()
    test_groups: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "dataset_fingerprint": self.dataset_fingerprint,
            "n_samples": int(self.n_samples),
            "seed": int(self.seed),
            "method": self.method,
            "validation_fraction": float(self.validation_fraction),
            "test_fraction": float(self.test_fraction),
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "test_indices": list(self.test_indices),
        }
        if self.group_key is not None:
            payload.update(
                {
                    "group_key": self.group_key,
                    "train_groups": list(self.train_groups),
                    "validation_groups": list(self.validation_groups),
                    "test_groups": list(self.test_groups),
                }
            )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitManifest":
        return cls(
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            n_samples=int(payload["n_samples"]),
            seed=int(payload["seed"]),
            method=str(payload["method"]),
            validation_fraction=float(payload["validation_fraction"]),
            test_fraction=float(payload["test_fraction"]),
            train_indices=tuple(int(index) for index in payload["train_indices"]),
            validation_indices=tuple(int(index) for index in payload["validation_indices"]),
            test_indices=tuple(int(index) for index in payload["test_indices"]),
            group_key=str(payload["group_key"]) if payload.get("group_key") is not None else None,
            train_groups=tuple(int(group) for group in payload.get("train_groups", [])),
            validation_groups=tuple(int(group) for group in payload.get("validation_groups", [])),
            test_groups=tuple(int(group) for group in payload.get("test_groups", [])),
        )


def dataset_fingerprint(inputs: dict[str, object], y: np.ndarray) -> str:
    hasher = hashlib.sha256()
    _hash_value(hasher, "y", np.asarray(y))
    for key in sorted(inputs):
        _hash_value(hasher, key, inputs[key])
    return hasher.hexdigest()


def make_split_manifest(
    inputs: dict[str, object],
    y: np.ndarray,
    *,
    seed: int,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    method: str = "stratified_random",
) -> SplitManifest:
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError("Split manifests require a 1-D target array.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")
    n_samples = int(y_array.shape[0])
    if n_samples < 6:
        raise ValueError("At least 6 samples are required for train/validation/test splits.")
    rng = np.random.default_rng(seed)
    if method == "stratified_random":
        train, validation, test = _stratified_indices(y_array, rng, validation_fraction, test_fraction)
    elif method == "random":
        shuffled = np.arange(n_samples)
        rng.shuffle(shuffled)
        test_count, validation_count = _split_counts(n_samples, validation_fraction, test_fraction)
        test = shuffled[:test_count]
        validation = shuffled[test_count : test_count + validation_count]
        train = shuffled[test_count + validation_count :]
    else:
        raise ValueError("method must be 'stratified_random' or 'random'.")

    for indices in (train, validation, test):
        rng.shuffle(indices)
    manifest = SplitManifest(
        dataset_fingerprint=dataset_fingerprint(inputs, y_array),
        n_samples=n_samples,
        seed=int(seed),
        method=method,
        validation_fraction=float(validation_fraction),
        test_fraction=float(test_fraction),
        train_indices=tuple(int(index) for index in sorted(train.tolist())),
        validation_indices=tuple(int(index) for index in sorted(validation.tolist())),
        test_indices=tuple(int(index) for index in sorted(test.tolist())),
    )
    validate_split_manifest(manifest)
    return manifest


def make_grouped_split_manifest(
    inputs: dict[str, object],
    y: np.ndarray,
    *,
    groups: np.ndarray,
    group_key: str = "sample_id",
    seed: int,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    method: str = "group_stratified_random",
) -> SplitManifest:
    y_array = np.asarray(y)
    group_array = np.asarray(groups)
    if y_array.ndim != 1:
        raise ValueError("Split manifests require a 1-D target array.")
    if group_array.ndim != 1 or group_array.shape[0] != y_array.shape[0]:
        raise ValueError("groups must be a 1-D array with the same length as y.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")
    n_samples = int(y_array.shape[0])
    if n_samples < 6:
        raise ValueError("At least 6 samples are required for train/validation/test splits.")
    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    if unique_groups.shape[0] < 6:
        raise ValueError("At least 6 groups are required for grouped train/validation/test splits.")

    group_labels = np.zeros(unique_groups.shape[0], dtype=np.float64)
    for group_index in range(unique_groups.shape[0]):
        group_y = y_array[inverse == group_index]
        group_labels[group_index] = float(np.max(group_y))

    rng = np.random.default_rng(seed)
    if method == "group_stratified_random":
        train_group_idx, validation_group_idx, test_group_idx = _group_split_indices(
            group_labels,
            rng,
            validation_fraction,
            test_fraction,
        )
    elif method == "group_random":
        shuffled = np.arange(unique_groups.shape[0])
        rng.shuffle(shuffled)
        test_count, validation_count = _split_counts(unique_groups.shape[0], validation_fraction, test_fraction)
        test_group_idx = shuffled[:test_count]
        validation_group_idx = shuffled[test_count : test_count + validation_count]
        train_group_idx = shuffled[test_count + validation_count :]
    else:
        raise ValueError("method must be 'group_stratified_random' or 'group_random'.")

    train_groups = unique_groups[train_group_idx]
    validation_groups = unique_groups[validation_group_idx]
    test_groups = unique_groups[test_group_idx]
    train = np.flatnonzero(np.isin(group_array, train_groups))
    validation = np.flatnonzero(np.isin(group_array, validation_groups))
    test = np.flatnonzero(np.isin(group_array, test_groups))
    manifest = SplitManifest(
        dataset_fingerprint=dataset_fingerprint(inputs, y_array),
        n_samples=n_samples,
        seed=int(seed),
        method=method,
        validation_fraction=float(validation_fraction),
        test_fraction=float(test_fraction),
        train_indices=tuple(int(index) for index in sorted(train.tolist())),
        validation_indices=tuple(int(index) for index in sorted(validation.tolist())),
        test_indices=tuple(int(index) for index in sorted(test.tolist())),
        group_key=group_key,
        train_groups=tuple(int(group) for group in sorted(train_groups.tolist())),
        validation_groups=tuple(int(group) for group in sorted(validation_groups.tolist())),
        test_groups=tuple(int(group) for group in sorted(test_groups.tolist())),
    )
    validate_split_manifest(manifest)
    validate_grouped_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: SplitManifest) -> None:
    train = set(manifest.train_indices)
    validation = set(manifest.validation_indices)
    test = set(manifest.test_indices)
    if not train or not validation or not test:
        raise ValueError("Train, validation, and test splits must all be non-empty.")
    if train & validation or train & test or validation & test:
        raise ValueError("Split indices must be disjoint.")
    combined = train | validation | test
    expected = set(range(manifest.n_samples))
    if combined != expected:
        missing = sorted(expected - combined)[:5]
        extra = sorted(combined - expected)[:5]
        raise ValueError(f"Split indices must cover every sample exactly once; missing={missing}, extra={extra}.")


def validate_grouped_split_manifest(manifest: SplitManifest) -> None:
    if manifest.group_key is None:
        return
    train = set(manifest.train_groups)
    validation = set(manifest.validation_groups)
    test = set(manifest.test_groups)
    if not train or not validation or not test:
        raise ValueError("Grouped split manifests must have non-empty train, validation, and test groups.")
    if train & validation or train & test or validation & test:
        raise ValueError("Grouped split group values must be disjoint.")


def subset_inputs(inputs: dict[str, object], indices: tuple[int, ...] | list[int] | np.ndarray, *, n_samples: int) -> dict[str, object]:
    index_array = np.asarray(indices, dtype=np.int64)
    subset: dict[str, object] = {}
    for key, value in inputs.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == n_samples:
            subset[key] = value[index_array]
        elif isinstance(value, (list, tuple)) and len(value) == n_samples:
            subset[key] = np.asarray(value)[index_array]
        else:
            subset[key] = value
    return subset


def split_dataset(
    inputs: dict[str, object],
    y: np.ndarray,
    manifest: SplitManifest,
) -> dict[str, tuple[dict[str, object], np.ndarray]]:
    y_array = np.asarray(y)
    if manifest.dataset_fingerprint != dataset_fingerprint(inputs, y_array):
        raise ValueError("Dataset fingerprint does not match split manifest.")
    return {
        "train": (
            subset_inputs(inputs, manifest.train_indices, n_samples=manifest.n_samples),
            y_array[np.asarray(manifest.train_indices, dtype=np.int64)],
        ),
        "validation": (
            subset_inputs(inputs, manifest.validation_indices, n_samples=manifest.n_samples),
            y_array[np.asarray(manifest.validation_indices, dtype=np.int64)],
        ),
        "test": (
            subset_inputs(inputs, manifest.test_indices, n_samples=manifest.n_samples),
            y_array[np.asarray(manifest.test_indices, dtype=np.int64)],
        ),
    }


def write_split_manifest(path: str | pathlib.Path, manifest: SplitManifest) -> pathlib.Path:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return output_path


def read_split_manifest(path: str | pathlib.Path) -> SplitManifest:
    return SplitManifest.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def _stratified_indices(
    y: np.ndarray,
    rng: np.random.Generator,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        test_count, validation_count = _split_counts(len(indices), validation_fraction, test_fraction)
        test_parts.append(indices[:test_count])
        validation_parts.append(indices[test_count : test_count + validation_count])
        train_parts.append(indices[test_count + validation_count :])
    return np.concatenate(train_parts), np.concatenate(validation_parts), np.concatenate(test_parts)


def _group_split_indices(
    group_labels: np.ndarray,
    rng: np.random.Generator,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, counts = np.unique(group_labels, return_counts=True)
    if labels.shape[0] > 1 and np.all(counts >= 3):
        return _stratified_indices(group_labels, rng, validation_fraction, test_fraction)
    shuffled = np.arange(group_labels.shape[0])
    rng.shuffle(shuffled)
    test_count, validation_count = _split_counts(group_labels.shape[0], validation_fraction, test_fraction)
    test = shuffled[:test_count]
    validation = shuffled[test_count : test_count + validation_count]
    train = shuffled[test_count + validation_count :]
    return train, validation, test


def _split_counts(n_samples: int, validation_fraction: float, test_fraction: float) -> tuple[int, int]:
    if n_samples < 3:
        raise ValueError("Each stratification group must contain at least 3 samples.")
    test_count = max(1, int(round(n_samples * test_fraction))) if test_fraction > 0.0 else 0
    validation_count = max(1, int(round(n_samples * validation_fraction))) if validation_fraction > 0.0 else 0
    while n_samples - test_count - validation_count < 1:
        if test_count >= validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break
    return test_count, validation_count


def _hash_value(hasher: "hashlib._Hash", key: str, value: object) -> None:
    hasher.update(key.encode("utf-8"))
    hasher.update(b"\0")
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.tobytes())
        return
    if isinstance(value, np.generic):
        _hash_value(hasher, key, value.item())
        return
    try:
        encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    except TypeError:
        encoded = repr(value).encode("utf-8")
    hasher.update(encoded)
