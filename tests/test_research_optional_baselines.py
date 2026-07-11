from __future__ import annotations

import sys

import numpy as np
import pytest

from benchmarks.research_suite.optional_baselines import (
    AutoFeatAdapter,
    CapabilityStatus,
    CommandRegressorAdapter,
    ExtraTreesAdapter,
    HistGradientBoostingAdapter,
    OptionalBaselineUnavailable,
    PySRAdapter,
    capability_report,
)


class _FakeRegressor:
    def __init__(self, **parameters: object) -> None:
        self.parameters = parameters

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_FakeRegressor":
        self.mean = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.mean)


@pytest.mark.parametrize(
    "adapter_type",
    [HistGradientBoostingAdapter, ExtraTreesAdapter, AutoFeatAdapter, PySRAdapter],
)
def test_python_adapters_accept_an_explicit_fake_backend(adapter_type: type) -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    adapter = adapter_type(estimator_factory=_FakeRegressor, sentinel=3).fit(x, y)

    assert adapter.capability.available
    assert adapter.estimator_.parameters == {"sentinel": 3}
    np.testing.assert_allclose(adapter.predict(x[:3]), np.mean(y))


def test_unavailable_python_backend_never_substitutes(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.research_suite.optional_baselines as optional

    real_find_spec = optional.importlib.util.find_spec
    monkeypatch.setattr(
        optional.importlib.util,
        "find_spec",
        lambda name: None if name == "autofeat" else real_find_spec(name),
    )
    adapter = AutoFeatAdapter()

    assert not adapter.capability.available
    with pytest.raises(OptionalBaselineUnavailable, match="AutoFeat"):
        adapter.fit(np.ones((4, 2)), np.arange(4.0))


def test_capability_report_has_every_optional_backend() -> None:
    report = capability_report()
    assert set(report) == {
        "hist_gradient_boosting",
        "extra_trees",
        "autofeat",
        "pysr",
        "feat_command",
        "operon_command",
    }
    assert all(isinstance(status, CapabilityStatus) for status in report.values())


def test_command_adapter_with_a_fake_backend() -> None:
    program = (
        "import numpy as n,sys; "
        "x=n.loadtxt(sys.argv[1],delimiter=',',ndmin=2); "
        "n.savetxt(sys.argv[2],n.sum(x,axis=1),delimiter=',')"
    )
    adapter = CommandRegressorAdapter(
        ["-c", program, "{test_csv}", "{predictions_csv}"],
        executable=sys.executable,
    )
    train = np.arange(12, dtype=float).reshape(6, 2)
    predictions = adapter.fit_predict(train, np.arange(6.0), np.array([[1.0, 2.0], [3.0, 5.0]]))

    np.testing.assert_array_equal(predictions, [3.0, 8.0])


def test_missing_command_backend_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.research_suite.optional_baselines as optional

    monkeypatch.setattr(optional.shutil, "which", lambda executable: None)
    adapter = CommandRegressorAdapter(["{test_csv}"], executable="definitely-not-installed")

    assert not adapter.capability.available
    with pytest.raises(OptionalBaselineUnavailable, match="command baseline"):
        adapter.fit_predict(np.ones((3, 2)), np.ones(3), np.ones((1, 2)))
