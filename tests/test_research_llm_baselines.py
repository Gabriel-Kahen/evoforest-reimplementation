from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from benchmarks.research_suite.llm_baselines import (
    AIDECommandAdapter,
    BaselineCapabilityError,
    BaselineCredentialError,
    CompletionEnvelope,
    FeatureResponseError,
    IterativeScalarLLMFeatureRegressor,
    OneShotLLMFeatureRegressor,
)
from evoforest_arch.source import SourceSandboxPolicy


SOURCE_X0 = 'lambda ctx, values: FeatureBlock(np.column_stack([values["x"][:, 0] ** 2]), ["x0_sq"])'
SOURCE_X1 = 'lambda ctx, values: FeatureBlock(np.column_stack([values["x"][:, 1]]), ["x1"])'
NO_SANDBOX = SourceSandboxPolicy(enabled=False)


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, str, float]] = []

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> object:
        self.requests.append((system_prompt, user_prompt, temperature))
        return self.responses.pop(0)


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 3))
    return x, x[:, 0] ** 2 + 0.01 * rng.normal(size=100)


def test_one_shot_generates_sandbox_contract_features_and_accounts_usage() -> None:
    x, y = _data()
    events = []
    client = FakeClient([CompletionEnvelope(json.dumps({"source": SOURCE_X0}), 11, 7, 0.025)])
    model = OneShotLLMFeatureRegressor(client, sandbox_policy=NO_SANDBOX, usage_hook=events.append).fit(x[:70], y[:70])

    predictions = model.predict(x[70:])
    assert predictions.shape == (30,)
    assert np.sqrt(np.mean((predictions - y[70:]) ** 2)) < 0.1
    assert model.usage.calls == 1
    assert model.usage.total_tokens == 18
    assert model.usage.cost_usd == pytest.approx(0.025)
    assert events == [model.usage]


def test_iterative_baseline_returns_best_candidate_and_only_scalar_feedback() -> None:
    x, y = _data()
    client = FakeClient([SOURCE_X1, SOURCE_X0])
    model = IterativeScalarLLMFeatureRegressor(client, rounds=2, sandbox_policy=NO_SANDBOX).fit(
        x[:60], y[:60], x[60:80], y[60:80]
    )

    assert model.source_ == SOURCE_X0
    assert model.validation_scores_[1] < model.validation_scores_[0]
    assert "scalar feedback only" in client.requests[1][1]
    assert "residual" not in client.requests[1][1].lower()
    assert model.usage.calls == 2
    assert model.predict(x[80:]).shape == (20,)


def test_llm_baselines_fail_explicitly_without_client_or_valid_contract() -> None:
    x, y = _data()
    with pytest.raises(BaselineCredentialError, match="explicit configured LLM client"):
        OneShotLLMFeatureRegressor(None, sandbox_policy=NO_SANDBOX).fit(x, y)
    with pytest.raises(FeatureResponseError, match="must be a lambda"):
        OneShotLLMFeatureRegressor(FakeClient(["ordinary prose"]), sandbox_policy=NO_SANDBOX).fit(x, y)


def test_aide_adapter_uses_explicit_command_contract(tmp_path, monkeypatch) -> None:
    x, y = _data()
    calls: list[tuple[str, ...]] = []

    def runner(command, *, env, timeout):
        calls.append(tuple(command))
        if command[1] == "predict":
            input_x = np.load(command[2], allow_pickle=False)
            np.save(command[3], input_x[:, 0] ** 2)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("AIDE_TEST_KEY", "present")
    adapter = AIDECommandAdapter(
        fit_command=("python", "fit", "{dataset}", "{artifact_dir}"),
        predict_command=("python", "predict", "{input}", "{predictions}", "{artifact_dir}"),
        required_env=("AIDE_TEST_KEY",),
        runner=runner,
        artifact_dir=tmp_path / "aide",
    ).fit(x[:60], y[:60], x[60:80], y[60:80])

    np.testing.assert_allclose(adapter.predict(x[80:]), x[80:, 0] ** 2)
    assert adapter.command_calls == 2
    assert len(calls) == 2
    with np.load(tmp_path / "aide" / "dataset.npz") as dataset:
        assert dataset["train_x"].shape == (60, 3)


def test_aide_adapter_reports_missing_capability_and_credentials(tmp_path, monkeypatch) -> None:
    x, y = _data()
    unavailable = AIDECommandAdapter(("definitely-not-an-aide-command",), ("x",), artifact_dir=tmp_path)
    with pytest.raises(BaselineCapabilityError, match="unavailable"):
        unavailable.fit(x[:60], y[:60], x[60:], y[60:])

    monkeypatch.delenv("MISSING_AIDE_TOKEN", raising=False)
    missing_key = AIDECommandAdapter(
        ("python", "fit"), ("python", "predict"), required_env=("MISSING_AIDE_TOKEN",), artifact_dir=tmp_path
    )
    with pytest.raises(BaselineCredentialError, match="MISSING_AIDE_TOKEN"):
        missing_key.fit(x[:60], y[:60], x[60:], y[60:])
