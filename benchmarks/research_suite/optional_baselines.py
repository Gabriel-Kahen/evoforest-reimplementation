"""Explicit adapters for optional third-party research baselines.

Imports are deliberately lazy: importing the research suite does not require any
of these packages.  An unavailable backend raises instead of falling back to a
different estimator, which keeps benchmark labels honest.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Sequence

import numpy as np


class OptionalBaselineUnavailable(RuntimeError):
    """Raised when a requested baseline's real backend is not installed."""


@dataclass(frozen=True)
class CapabilityStatus:
    """Machine-readable availability report for an optional baseline."""

    name: str
    available: bool
    backend: str
    detail: str

    def require(self) -> None:
        if not self.available:
            raise OptionalBaselineUnavailable(f"{self.name} is unavailable: {self.detail}")


def _module_status(name: str, module: str) -> CapabilityStatus:
    available = importlib.util.find_spec(module) is not None
    detail = f"Python module {module!r} is installed." if available else f"Install Python module {module!r}."
    return CapabilityStatus(name=name, available=available, backend=module, detail=detail)


def _command_status(name: str, executable: str) -> CapabilityStatus:
    path = shutil.which(executable)
    detail = f"Executable found at {path}." if path else f"Executable {executable!r} was not found on PATH."
    return CapabilityStatus(name=name, available=path is not None, backend=executable, detail=detail)


def capability_report() -> dict[str, CapabilityStatus]:
    """Return the current environment's optional-baseline capabilities."""

    return {
        "hist_gradient_boosting": _module_status("HistGradientBoosting", "sklearn"),
        "extra_trees": _module_status("ExtraTrees", "sklearn"),
        "autofeat": _module_status("AutoFeat", "autofeat"),
        "pysr": _module_status("PySR", "pysr"),
        "feat_command": _command_status("FEAT command", "feat"),
        "operon_command": _command_status("Operon command", "operon"),
    }


def _arrays(x: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    features = np.asarray(x, dtype=np.float64)
    if features.ndim != 2 or min(features.shape) == 0:
        raise ValueError(f"x must be a non-empty 2-D array, got {features.shape}.")
    if y is None:
        return features, None
    target = np.asarray(y, dtype=np.float64)
    if target.ndim != 1 or target.shape[0] != features.shape[0] or not np.all(np.isfinite(target)):
        raise ValueError(f"y must be a finite 1-D array with {features.shape[0]} rows.")
    return features, target


class _PythonRegressorAdapter:
    name = "optional baseline"
    module = ""

    def __init__(self, *, estimator_factory: Callable[..., Any] | None = None, **parameters: Any) -> None:
        self._factory = estimator_factory
        self.parameters = parameters
        self.estimator_: Any | None = None
        self.n_features_: int | None = None

    @property
    def capability(self) -> CapabilityStatus:
        if self._factory is not None:
            return CapabilityStatus(self.name, True, "injected", "An explicit estimator factory was supplied.")
        return _module_status(self.name, self.module)

    def _estimator_factory(self) -> Callable[..., Any]:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_PythonRegressorAdapter":
        features, target = _arrays(x, y)
        self.capability.require()
        factory = self._factory or self._estimator_factory()
        estimator = factory(**self.parameters)
        estimator.fit(features, target)
        self.estimator_ = estimator
        self.n_features_ = features.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.estimator_ is None or self.n_features_ is None:
            raise RuntimeError("Fit the baseline before predicting.")
        features, _ = _arrays(x)
        if features.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} input columns, got {features.shape[1]}.")
        predictions = np.asarray(self.estimator_.predict(features), dtype=np.float64).reshape(-1)
        if predictions.shape[0] != features.shape[0] or not np.all(np.isfinite(predictions)):
            raise RuntimeError(f"{self.name} returned invalid predictions.")
        return predictions


class HistGradientBoostingAdapter(_PythonRegressorAdapter):
    name = "HistGradientBoosting"
    module = "sklearn"

    def _estimator_factory(self) -> Callable[..., Any]:
        return importlib.import_module("sklearn.ensemble").HistGradientBoostingRegressor


class ExtraTreesAdapter(_PythonRegressorAdapter):
    name = "ExtraTrees"
    module = "sklearn"

    def _estimator_factory(self) -> Callable[..., Any]:
        return importlib.import_module("sklearn.ensemble").ExtraTreesRegressor


class AutoFeatAdapter(_PythonRegressorAdapter):
    name = "AutoFeat"
    module = "autofeat"

    def _estimator_factory(self) -> Callable[..., Any]:
        return importlib.import_module("autofeat").AutoFeatRegressor


class PySRAdapter(_PythonRegressorAdapter):
    name = "PySR"
    module = "pysr"

    def _estimator_factory(self) -> Callable[..., Any]:
        return importlib.import_module("pysr").PySRRegressor


class CommandRegressorAdapter:
    """One-shot adapter for FEAT/Operon CLI wrappers.

    ``arguments`` is an argv template supplied by the experiment environment.
    Supported placeholders are ``{train_csv}``, ``{test_csv}``,
    ``{predictions_csv}``, and ``{model_path}``.  The command must write one
    numeric prediction per test row.  Requiring an explicit template avoids
    assuming a CLI dialect or silently invoking an incompatible binary.
    """

    name = "command baseline"
    default_executable = ""

    def __init__(
        self,
        arguments: Sequence[str],
        *,
        executable: str | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.executable = executable or self.default_executable
        self.arguments = tuple(arguments)
        self.timeout_seconds = timeout_seconds

    @property
    def capability(self) -> CapabilityStatus:
        return _command_status(self.name, self.executable)

    def fit_predict(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        train, target = _arrays(x_train, y_train)
        test, _ = _arrays(x_test)
        if test.shape[1] != train.shape[1]:
            raise ValueError(f"Expected {train.shape[1]} test columns, got {test.shape[1]}.")
        self.capability.require()
        if not self.arguments:
            raise ValueError(f"{self.name} requires an explicit command argument template.")

        with tempfile.TemporaryDirectory(prefix="evoforest-baseline-") as directory:
            root = Path(directory)
            paths = {
                "train_csv": root / "train.csv",
                "test_csv": root / "test.csv",
                "predictions_csv": root / "predictions.csv",
                "model_path": root / "model",
            }
            np.savetxt(paths["train_csv"], np.column_stack((train, target)), delimiter=",")
            np.savetxt(paths["test_csv"], test, delimiter=",")
            argv = [self.executable, *(item.format_map(paths) for item in self.arguments)]
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
            if completed.returncode != 0:
                message = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
                raise RuntimeError(f"{self.name} failed with exit code {completed.returncode}: {message}")
            output = paths["predictions_csv"]
            if not output.is_file():
                raise RuntimeError(f"{self.name} did not create {output.name}.")
            predictions = np.asarray(np.loadtxt(output, delimiter=","), dtype=np.float64).reshape(-1)
            if predictions.shape[0] != test.shape[0] or not np.all(np.isfinite(predictions)):
                raise RuntimeError(f"{self.name} returned invalid predictions.")
            return predictions


class FEATCommandAdapter(CommandRegressorAdapter):
    name = "FEAT command"
    default_executable = "feat"


class OperonCommandAdapter(CommandRegressorAdapter):
    name = "Operon command"
    default_executable = "operon"
