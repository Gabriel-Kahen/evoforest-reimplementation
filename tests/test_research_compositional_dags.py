from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.research_suite.compositional_dags import (
    NodeSpec,
    TaskSpec,
    evaluate_ground_truth,
    generate_benchmark,
    task_catalog,
)


def test_catalog_is_valid_and_exposes_ground_truth_metadata() -> None:
    catalog = task_catalog()

    assert set(catalog) == {
        "shared_wave_gate",
        "piecewise_rational",
        "heteroscedastic_reuse",
        "missing_sensor_composition",
    }
    for spec in catalog.values():
        metadata = spec.metadata()
        assert metadata["active_variables"]
        assert metadata["motifs"]
        assert any(motif["reuse_count"] > 1 for motif in metadata["motifs"])
        json.dumps(metadata)


def test_generation_is_deterministic_and_split_specific() -> None:
    first = generate_benchmark("shared_wave_gate", seed=42, n_train=40, n_validation=20, n_test=30)
    second = generate_benchmark("shared_wave_gate", seed=42, n_train=40, n_validation=20, n_test=30)

    for split_name in ("train", "validation", "test_interpolation", "test_extrapolation"):
        left = getattr(first, split_name)
        right = getattr(second, split_name)
        np.testing.assert_array_equal(left.X, right.X)
        np.testing.assert_array_equal(left.y, right.y)
    assert not np.array_equal(first.train.latent_X[:20], first.validation.latent_X)


def test_extrapolation_split_is_outside_active_variable_support() -> None:
    dataset = generate_benchmark("piecewise_rational", seed=7, n_train=50, n_validation=20, n_test=80)
    active = np.asarray(dataset.spec.active_variables)

    assert np.all(np.abs(dataset.train.latent_X[:, active]) <= 1.0)
    assert np.all(np.abs(dataset.test_interpolation.latent_X[:, active]) <= 1.0)
    assert np.all(np.abs(dataset.test_extrapolation.latent_X[:, active]) >= 1.25)
    assert np.all(np.abs(dataset.test_extrapolation.latent_X[:, active]) <= 2.25)
    assert dataset.test_extrapolation.regime == "extrapolation"


def test_ground_truth_nodes_match_known_formula() -> None:
    spec = task_catalog()["shared_wave_gate"]
    X = np.zeros((2, spec.n_features))
    X[0, :5] = (1.0, 0.5, 3.0, 1.0, 2.0)
    X[1, :5] = (-1.0, 0.25, -2.0, -1.0, 4.0)

    values = evaluate_ground_truth(spec, X)
    wave = np.sin(1.7 * X[:, 0] * X[:, 1])
    expected = wave + (X[:, 3] > 0) * np.log1p(np.abs(X[:, 2])) + 0.2 * wave * X[:, 4]

    np.testing.assert_allclose(values["shared_wave"], wave)
    np.testing.assert_allclose(values["target"], expected)


def test_missingness_changes_observations_not_latent_targets() -> None:
    dataset = generate_benchmark("missing_sensor_composition", seed=11, n_train=1000, n_validation=10, n_test=10)

    assert dataset.train.missing_mask.mean() == pytest.approx(0.12, abs=0.025)
    assert np.isnan(dataset.train.X)[dataset.train.missing_mask].all()
    assert np.isfinite(dataset.train.latent_X).all()
    reconstructed = evaluate_ground_truth(dataset.spec, dataset.train.latent_X)[dataset.spec.output_node]
    np.testing.assert_allclose(dataset.train.y_clean, reconstructed)


def test_task_spec_rejects_forward_references() -> None:
    with pytest.raises(ValueError, match="forward/unknown"):
        TaskSpec(
            name="invalid",
            n_features=2,
            active_variables=(0,),
            nodes=(NodeSpec("first", "square", ("later",)), NodeSpec("later", "square", ("x0",))),
            output_node="first",
            motifs=(),
            description="invalid graph",
        )


def test_dataset_metadata_records_split_sizes_and_support() -> None:
    dataset = generate_benchmark("heteroscedastic_reuse", seed=3, n_train=13, n_validation=7, n_test=11)
    metadata = dataset.metadata()

    assert metadata["splits"] == {
        "train": 13,
        "validation": 7,
        "test_interpolation": 11,
        "test_extrapolation": 11,
    }
    assert metadata["task"]["heteroscedastic_node"] == "compressed"
    json.dumps(metadata)
