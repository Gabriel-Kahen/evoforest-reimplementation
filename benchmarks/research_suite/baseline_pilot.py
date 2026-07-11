from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from benchmarks.common import write_report
from benchmarks.research_suite.compositional_dags import generate_benchmark
from benchmarks.research_suite.external_datasets import load_regression_dataset
from benchmarks.research_suite.metrics import nrmse, r2_score, rmse
from benchmarks.research_suite.optional_baselines import (
    AutoFeatAdapter,
    ExtraTreesAdapter,
    FEATCommandAdapter,
    HistGradientBoostingAdapter,
    PySRAdapter,
    capability_report,
)


def run_baseline_pilot(root: Path, *, seed: int = 101) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for task_name in ("shared_wave_gate", "piecewise_rational", "heteroscedastic_reuse"):
        dataset = generate_benchmark(task_name, seed=seed, n_train=128, n_validation=64, n_test=128)
        train_x = np.vstack((dataset.train.X, dataset.validation.X))
        train_y = np.concatenate((dataset.train.y, dataset.validation.y))
        _run_methods_multi(
            task_name,
            train_x,
            train_y,
            {
                "interpolation": (dataset.test_interpolation.X, dataset.test_interpolation.y),
                "extrapolation": (dataset.test_extrapolation.X, dataset.test_extrapolation.y),
            },
            seed,
            rows,
            failures,
        )

    manifest_dir = Path(__file__).with_name("manifests") / "pilot_v1"
    for manifest_path in sorted(manifest_dir.glob("*.manifest.json")):
        dataset = load_regression_dataset(manifest_path)
        train_x = np.vstack((dataset.train.X, dataset.validation.X))
        train_y = np.concatenate((dataset.train.y, dataset.validation.y))
        _run_methods_multi(
            dataset.manifest.dataset_id,
            train_x,
            train_y,
            {"sealed_test": (dataset.test.X, dataset.test.y)},
            seed,
            rows,
            failures,
        )
    _run_pysr_sentinel(seed, rows, failures)
    _run_feat_sentinel(seed, rows, failures)
    capabilities = {name: asdict(status) for name, status in capability_report().items()}
    return {"seed": seed, "capabilities": capabilities, "results": rows, "failures": failures}


def _run_pysr_sentinel(seed: int, rows: list[dict[str, object]], failures: list[dict[str, str]]) -> None:
    status = capability_report()["pysr"]
    if not status.available:
        failures.append({"task": "shared_wave_gate", "split": "sentinel", "method": "pysr", "error": status.detail})
        return
    dataset = generate_benchmark("shared_wave_gate", seed=seed, n_train=64, n_validation=32, n_test=64)
    train_x = np.vstack((dataset.train.X, dataset.validation.X))
    train_y = np.concatenate((dataset.train.y, dataset.validation.y))
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="evoforest-pysr-pilot-") as output_directory:
            model = PySRAdapter(
                niterations=3,
                populations=2,
                population_size=20,
                binary_operators=["+", "-", "*"],
                unary_operators=["sin", "square"],
                maxsize=12,
                verbosity=0,
                progress=False,
                parallelism="serial",
                random_state=seed,
                deterministic=True,
                output_directory=output_directory,
            ).fit(train_x, train_y)
            fit_seconds = time.monotonic() - started
            for split_name, split in (
                ("interpolation", dataset.test_interpolation),
                ("extrapolation", dataset.test_extrapolation),
            ):
                predict_started = time.monotonic()
                prediction = model.predict(split.X)
                predict_seconds = time.monotonic() - predict_started
                rows.append(
                    {
                        "task": dataset.spec.name,
                        "split": split_name,
                        "method": "pysr",
                        "rmse": rmse(split.y, prediction),
                        "nrmse": nrmse(split.y, prediction),
                        "r2": r2_score(split.y, prediction),
                        "fit_wall_time_seconds": fit_seconds,
                        "predict_wall_time_seconds": predict_seconds,
                        "wall_time_seconds": fit_seconds + predict_seconds,
                    }
                )
    except Exception as error:
        failures.append({"task": dataset.spec.name, "split": "sentinel", "method": "pysr", "error": f"{type(error).__name__}: {error}"})


def _run_feat_sentinel(seed: int, rows: list[dict[str, object]], failures: list[dict[str, str]]) -> None:
    executable = os.environ.get("EVOFOREST_FEAT_PYTHON", "")
    if not executable:
        failures.append({"task": "shared_wave_gate", "split": "sentinel", "method": "feat", "error": "EVOFOREST_FEAT_PYTHON is not configured."})
        return
    runner = Path(__file__).parent / "environments" / "feat_runner.py"
    dataset = generate_benchmark("shared_wave_gate", seed=seed, n_train=64, n_validation=32, n_test=64)
    train_x = np.vstack((dataset.train.X, dataset.validation.X))
    train_y = np.concatenate((dataset.train.y, dataset.validation.y))
    model = FEATCommandAdapter(
        executable=executable,
        arguments=(
            str(runner),
            "--train-csv", "{train_csv}",
            "--test-csv", "{test_csv}",
            "--predictions-csv", "{predictions_csv}",
            "--seed", str(seed),
            "--generations", "1",
            "--population", "12",
        ),
        timeout_seconds=300,
    )
    started = time.monotonic()
    try:
        prediction = model.fit_predict(train_x, train_y, dataset.test_interpolation.X)
        elapsed = time.monotonic() - started
        rows.append(
            {
                "task": dataset.spec.name,
                "split": "interpolation",
                "method": "feat",
                "rmse": rmse(dataset.test_interpolation.y, prediction),
                "nrmse": nrmse(dataset.test_interpolation.y, prediction),
                "r2": r2_score(dataset.test_interpolation.y, prediction),
                "fit_wall_time_seconds": elapsed,
                "predict_wall_time_seconds": 0.0,
                "wall_time_seconds": elapsed,
            }
        )
    except Exception as error:
        failures.append({"task": dataset.spec.name, "split": "sentinel", "method": "feat", "error": f"{type(error).__name__}: {error}"})


def _run_methods(
    task: str,
    split: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    rows: list[dict[str, object]],
    failures: list[dict[str, str]],
) -> None:
    _run_methods_multi(task, train_x, train_y, {split: (test_x, test_y)}, seed, rows, failures)


def _run_methods_multi(
    task: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    tests: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
    rows: list[dict[str, object]],
    failures: list[dict[str, str]],
) -> None:
    methods = (
        ("hist_gradient_boosting", HistGradientBoostingAdapter(random_state=seed, max_iter=200)),
        ("extra_trees", ExtraTreesAdapter(random_state=seed, n_estimators=200, n_jobs=1)),
        (
            "autofeat",
            AutoFeatAdapter(
                feateng_steps=1,
                featsel_runs=1,
                max_gb=0.25,
                transformations=("1/", "log", "abs", "sqrt", "^2"),
                n_jobs=1,
                verbose=0,
            ),
        ),
    )
    for name, model in methods:
        started = time.monotonic()
        try:
            model.fit(train_x, train_y)
            fit_seconds = time.monotonic() - started
            for split, (test_x, test_y) in tests.items():
                predict_started = time.monotonic()
                prediction = model.predict(test_x)
                predict_seconds = time.monotonic() - predict_started
                rows.append(
                    {
                        "task": task,
                        "split": split,
                        "method": name,
                        "rmse": rmse(test_y, prediction),
                        "nrmse": nrmse(test_y, prediction),
                        "r2": r2_score(test_y, prediction),
                        "fit_wall_time_seconds": fit_seconds,
                        "predict_wall_time_seconds": predict_seconds,
                        "wall_time_seconds": fit_seconds + predict_seconds,
                    }
                )
        except Exception as error:
            failures.append({"task": task, "split": ",".join(tests), "method": name, "error": f"{type(error).__name__}: {error}"})


def markdown_report(payload: dict[str, object]) -> str:
    lines = ["# Optional Baseline Pilot", ""]
    for name, status in payload["capabilities"].items():  # type: ignore[union-attr]
        lines.append(f"- {name}: available={status['available']} ({status['detail']})")
    lines.extend(["", f"Completed rows: `{len(payload['results'])}`", f"Failures: `{len(payload['failures'])}`"])
    return "\n".join(lines)


def run(output_dir: Path) -> tuple[Path, Path]:
    payload = run_baseline_pilot(output_dir)
    return write_report(output_dir, "optional_baseline_pilot", payload, markdown_report(payload))
