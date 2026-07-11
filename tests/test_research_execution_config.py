from __future__ import annotations

import json
import sys

import pytest

from benchmarks.research_suite.execution_config import (
    AIDEExecutionConfig,
    ExecutionConfig,
    LLMExecutionConfig,
    execution_capability_report,
    load_execution_config,
    schema_fingerprint,
    schema_lock_valid,
    write_execution_config,
)


def _config() -> ExecutionConfig:
    return ExecutionConfig(
        name="confirmatory-runtime",
        version=1,
        llm=LLMExecutionConfig(
            provider="example-provider",
            model_identifier="example-model-2026-01",
            temperature=0.2,
            call_budget=100,
            token_budget=200_000,
            input_price_usd_per_million_tokens=1.25,
            output_price_usd_per_million_tokens=5.0,
            credential_env_names=("EXAMPLE_API_KEY",),
        ),
        aide=AIDEExecutionConfig(
            fit_command=(sys.executable, "aide_fit.py", "{dataset}", "{artifact_dir}"),
            predict_command=(sys.executable, "aide_predict.py", "{input}", "{predictions}"),
            timeout_seconds=600,
            credential_env_names=("AIDE_API_KEY",),
        ),
    )


def test_execution_config_round_trip_is_fingerprinted_and_secret_free(tmp_path) -> None:
    config = _config()
    path = write_execution_config(tmp_path / "execution.json", config)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert load_execution_config(path) == config
    assert payload["fingerprint"] == config.fingerprint()
    assert payload["llm"]["credential_env_names"] == ["EXAMPLE_API_KEY"]
    assert "credential_values" not in path.read_text(encoding="utf-8")
    assert len(schema_fingerprint()) == 64
    assert schema_lock_valid()


def test_capability_report_checks_presence_without_exposing_values() -> None:
    secret = "do-not-leak-this-value"
    report = execution_capability_report(
        _config(),
        {"EXAMPLE_API_KEY": secret, "AIDE_API_KEY": secret},
    )

    assert report["llm"].available
    assert report["aide"].available
    assert report["llm"].credential_presence == {"EXAMPLE_API_KEY": True}
    assert secret not in repr(report)


def test_missing_credentials_are_named_but_not_read() -> None:
    report = execution_capability_report(_config(), {})

    assert not report["llm"].available
    assert not report["aide"].available
    assert report["llm"].credential_presence == {"EXAMPLE_API_KEY": False}


def test_embedded_secrets_and_unknown_fields_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot embed credential"):
        AIDEExecutionConfig(
            fit_command=("aide", "--api-key=literal"),
            predict_command=("aide", "predict"),
            timeout_seconds=10,
            credential_env_names=("AIDE_KEY",),
        )

    path = write_execution_config(tmp_path / "execution.json", _config())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["api_key"] = "literal"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown execution-config fields"):
        load_execution_config(path)
