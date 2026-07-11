from __future__ import annotations

from pathlib import Path
import sys

from benchmarks.research_suite.confirmatory import verify_confirmatory_readiness
from benchmarks.research_suite.execution_config import AIDEExecutionConfig, ExecutionConfig, LLMExecutionConfig


def test_confirmatory_readiness_verifies_lock_and_names_missing_external_requirements() -> None:
    status = verify_confirmatory_readiness(external_manifests=())

    assert status["lock_valid"] is True
    assert status["ready"] is False
    assert "12 frozen confirmatory external dataset manifests" in status["missing_requirements"]
    assert status["execution_config_fingerprint"] is None
    assert len(status["execution_schema_fingerprint"]) == 64
    assert status["execution_schema_lock_valid"] is True


def test_confirmatory_can_be_ready_with_injected_capabilities(monkeypatch, tmp_path) -> None:
    from benchmarks.research_suite import confirmatory
    from benchmarks.research_suite.optional_baselines import CapabilityStatus

    monkeypatch.setattr(confirmatory, "capability_report", lambda: {
        "feat_command": CapabilityStatus("x", True, "x", "x"),
        "pysr": CapabilityStatus("x", True, "x", "x"),
    })
    manifests = tuple(tmp_path / f"{index}.json" for index in range(12))

    status = verify_confirmatory_readiness(llm_configured=True, aide_configured=True, external_manifests=manifests)

    assert status["ready"] is True


def test_confirmatory_uses_frozen_execution_capabilities(monkeypatch, tmp_path) -> None:
    from benchmarks.research_suite import confirmatory
    from benchmarks.research_suite.optional_baselines import CapabilityStatus

    monkeypatch.setattr(confirmatory, "capability_report", lambda: {
        "feat_command": CapabilityStatus("x", True, "x", "x"),
    })
    monkeypatch.setenv("LLM_TEST_KEY", "secret-not-reported")
    monkeypatch.setenv("AIDE_TEST_KEY", "secret-not-reported")
    config = ExecutionConfig(
        name="test",
        version=1,
        llm=LLMExecutionConfig("provider", "model", 0.0, 10, 1000, 1.0, 2.0, ("LLM_TEST_KEY",)),
        aide=AIDEExecutionConfig(
            (sys.executable, "fit"), (sys.executable, "predict"), 30.0, ("AIDE_TEST_KEY",)
        ),
    )
    manifests = tuple(tmp_path / f"{index}.json" for index in range(12))

    status = verify_confirmatory_readiness(execution_config=config, external_manifests=manifests)

    assert status["ready"] is True
    assert status["execution_capabilities"]["llm"]["credential_presence"] == {"LLM_TEST_KEY": True}
    assert "secret-not-reported" not in repr(status)
