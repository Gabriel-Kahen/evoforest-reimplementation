from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from benchmarks.research_suite.external_datasets import load_manifest, load_regression_dataset


def _write_dataset(tmp_path, *, split_file: bool = False, checksum: bool = False):
    X = np.arange(36, dtype=float).reshape(12, 3)
    y = np.linspace(-1.0, 1.0, 12)
    data_path = tmp_path / "data.npz"
    np.savez(data_path, features=X, response=y)
    splits = {"train": list(range(6)), "validation": [6, 7, 8], "test": [9, 10, 11]}
    payload = {
        "version": 1,
        "dataset_id": "fixture-regression",
        "data": "data.npz",
        "feature_key": "features",
        "target_key": "response",
        "feature_names": ["a", "b", "c"],
        "metadata": {"source": "unit test"},
    }
    if split_file:
        (tmp_path / "splits.json").write_text(json.dumps({"splits": splits}), encoding="utf-8")
        payload["split_file"] = "splits.json"
    else:
        payload["splits"] = splits
    if checksum:
        payload["sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, X, y


def test_loads_inline_frozen_splits_and_copies_read_only_arrays(tmp_path) -> None:
    manifest_path, X, y = _write_dataset(tmp_path, checksum=True)

    dataset = load_regression_dataset(manifest_path)

    assert dataset.manifest.dataset_id == "fixture-regression"
    assert dataset.feature_names == ("a", "b", "c")
    np.testing.assert_array_equal(dataset.train.X, X[:6])
    np.testing.assert_array_equal(dataset.validation.y, y[6:9])
    np.testing.assert_array_equal(dataset.test.indices, [9, 10, 11])
    with pytest.raises(ValueError):
        dataset.test.y[0] = 100
    with pytest.raises(ValueError):
        dataset.train.indices[0] = 4


def test_loads_splits_from_referenced_json(tmp_path) -> None:
    manifest_path, _, _ = _write_dataset(tmp_path, split_file=True)

    manifest = load_manifest(manifest_path)
    dataset = load_regression_dataset(manifest_path)

    assert manifest.split_indices["validation"] == (6, 7, 8)
    assert dataset.test.X.shape == (3, 3)


def test_rejects_overlapping_or_out_of_range_splits(tmp_path) -> None:
    manifest_path, _, _ = _write_dataset(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["splits"]["test"] = [8, 10, 99]
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="out-of-range"):
        load_regression_dataset(manifest_path)

    payload["splits"]["test"] = [8, 10, 11]
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="overlap"):
        load_regression_dataset(manifest_path)


def test_rejects_remote_and_escaping_relative_references(tmp_path) -> None:
    manifest_path, _, _ = _write_dataset(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["data"] = "https://example.test/data.npz"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="URLs"):
        load_manifest(manifest_path)

    outside = tmp_path.parent / "outside.npz"
    np.savez(outside, X=np.ones((3, 2)), y=np.ones(3))
    payload["data"] = "../outside.npz"
    manifest_path.write_text(json.dumps(payload))
    try:
        with pytest.raises(ValueError, match="escapes"):
            load_manifest(manifest_path)
    finally:
        outside.unlink()


def test_rejects_checksum_mismatch_and_non_numeric_data(tmp_path) -> None:
    manifest_path, _, _ = _write_dataset(tmp_path, checksum=True)
    payload = json.loads(manifest_path.read_text())
    payload["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha256"):
        load_regression_dataset(manifest_path)

    payload.pop("sha256")
    np.savez(tmp_path / "data.npz", features=np.array([["bad"]] * 12), response=np.ones(12))
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="numeric"):
        load_regression_dataset(manifest_path)


def test_rejects_bad_feature_names_and_nonfinite_values(tmp_path) -> None:
    manifest_path, X, y = _write_dataset(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["feature_names"] = ["a", "a", "c"]
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="feature_names"):
        load_regression_dataset(manifest_path)

    payload["feature_names"] = ["a", "b", "c"]
    X[0, 0] = np.nan
    np.savez(tmp_path / "data.npz", features=X, response=y)
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite"):
        load_regression_dataset(manifest_path)
