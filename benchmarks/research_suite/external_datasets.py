"""Manifest-driven loaders for frozen, local numeric regression datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np


_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ExternalDatasetManifest:
    dataset_id: str
    manifest_path: Path
    data_path: Path
    feature_key: str
    target_key: str
    split_indices: Mapping[str, tuple[int, ...]]
    feature_names: tuple[str, ...] | None
    sha256: str | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenPartition:
    name: str
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray


@dataclass(frozen=True)
class FrozenRegressionDataset:
    manifest: ExternalDatasetManifest
    feature_names: tuple[str, ...]
    train: FrozenPartition
    validation: FrozenPartition
    test: FrozenPartition


def load_manifest(path: str | Path) -> ExternalDatasetManifest:
    """Read a versioned JSON manifest without accessing its dataset arrays."""

    manifest_path = Path(path).expanduser().resolve(strict=True)
    payload = _read_json_object(manifest_path)
    if payload.get("version") != 1:
        raise ValueError("manifest version must be 1")
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")

    data_path = _resolve_reference(manifest_path.parent, payload.get("data"), field="data")
    if data_path.suffix.lower() != ".npz":
        raise ValueError("data must reference an .npz file")

    has_inline = "splits" in payload
    has_file = "split_file" in payload
    if has_inline == has_file:
        raise ValueError("provide exactly one of splits or split_file")
    if has_file:
        split_path = _resolve_reference(manifest_path.parent, payload["split_file"], field="split_file")
        if split_path.suffix.lower() != ".json":
            raise ValueError("split_file must reference a JSON file")
        split_payload = _read_json_object(split_path)
        splits = split_payload.get("splits", split_payload)
    else:
        splits = payload["splits"]

    feature_key = payload.get("feature_key", "X")
    target_key = payload.get("target_key", "y")
    if not isinstance(feature_key, str) or not feature_key:
        raise ValueError("feature_key must be a non-empty string")
    if not isinstance(target_key, str) or not target_key:
        raise ValueError("target_key must be a non-empty string")

    feature_names_payload = payload.get("feature_names")
    feature_names = None
    if feature_names_payload is not None:
        if not isinstance(feature_names_payload, list) or not all(
            isinstance(value, str) and value for value in feature_names_payload
        ):
            raise ValueError("feature_names must be a list of non-empty strings")
        feature_names = tuple(feature_names_payload)

    checksum = payload.get("sha256")
    if checksum is not None and (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in checksum)
    ):
        raise ValueError("sha256 must contain 64 hexadecimal characters")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")

    return ExternalDatasetManifest(
        dataset_id=dataset_id,
        manifest_path=manifest_path,
        data_path=data_path,
        feature_key=feature_key,
        target_key=target_key,
        split_indices=_parse_splits(splits),
        feature_names=feature_names,
        sha256=None if checksum is None else checksum.lower(),
        metadata=metadata,
    )


def load_regression_dataset(path: str | Path) -> FrozenRegressionDataset:
    """Load and validate a manifest's NPZ arrays and immutable partitions."""

    manifest = load_manifest(path)
    if manifest.sha256 is not None and _file_sha256(manifest.data_path) != manifest.sha256:
        raise ValueError("dataset sha256 does not match manifest")
    try:
        archive = np.load(manifest.data_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not load numeric NPZ dataset: {error}") from error
    with archive:
        if manifest.feature_key not in archive or manifest.target_key not in archive:
            raise ValueError("NPZ does not contain the configured feature and target keys")
        X = np.asarray(archive[manifest.feature_key])
        y = np.asarray(archive[manifest.target_key])

    if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        raise ValueError("features and targets must be numeric")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("feature array must be two-dimensional")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("target must be one-dimensional with one value per row")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must contain only finite values")

    _validate_split_indices(manifest.split_indices, len(X))
    names = manifest.feature_names or tuple(f"x{i}" for i in range(X.shape[1]))
    if len(names) != X.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and match the feature count")

    partitions = {
        name: _partition(name, X, y, manifest.split_indices[name])
        for name in _SPLIT_NAMES
    }
    return FrozenRegressionDataset(
        manifest=manifest,
        feature_names=names,
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
    )


def _partition(name: str, X: np.ndarray, y: np.ndarray, indices: tuple[int, ...]) -> FrozenPartition:
    index_array = np.asarray(indices, dtype=np.int64)
    part_X = np.array(X[index_array], copy=True)
    part_y = np.array(y[index_array], copy=True)
    index_array.setflags(write=False)
    part_X.setflags(write=False)
    part_y.setflags(write=False)
    return FrozenPartition(name=name, X=part_X, y=part_y, indices=index_array)


def _parse_splits(payload: object) -> dict[str, tuple[int, ...]]:
    if not isinstance(payload, dict) or set(payload) != set(_SPLIT_NAMES):
        raise ValueError("splits must contain exactly train, validation, and test")
    result: dict[str, tuple[int, ...]] = {}
    for name in _SPLIT_NAMES:
        values = payload[name]
        if (
            not isinstance(values, list)
            or not values
            or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        ):
            raise ValueError(f"split {name} must be a non-empty list of integer indices")
        result[name] = tuple(values)
    return result


def _validate_split_indices(splits: Mapping[str, tuple[int, ...]], n_rows: int) -> None:
    seen: set[int] = set()
    for name in _SPLIT_NAMES:
        indices = splits[name]
        if len(set(indices)) != len(indices):
            raise ValueError(f"split {name} contains duplicate indices")
        if any(index < 0 or index >= n_rows for index in indices):
            raise ValueError(f"split {name} contains an out-of-range index")
        overlap = seen.intersection(indices)
        if overlap:
            raise ValueError(f"frozen splits overlap at indices {sorted(overlap)}")
        seen.update(indices)


def _resolve_reference(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty local path")
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        raise ValueError(f"{field} must be local; URLs are not supported")
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve(strict=True)
    resolved_base = base.resolve(strict=True)
    resolved = (resolved_base / raw_path).resolve(strict=True)
    try:
        resolved.relative_to(resolved_base)
    except ValueError as error:
        raise ValueError(f"{field} relative path escapes the manifest directory") from error
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file {path} must contain an object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
