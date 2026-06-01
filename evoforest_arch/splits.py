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

    def to_dict(self) -> dict[str, Any]:
        return {
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
