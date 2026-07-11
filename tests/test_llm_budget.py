from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

import pytest

from evoforest_arch.llm import GeminiLLMClient
from evoforest_arch.llm_budget import LLMBudget, LLMBudgetExceededError


def budget(path: Path, *, hard: float = 20.0) -> LLMBudget:
    return LLMBudget(path, 5.0, hard, 0.25, 1.50)


def test_budget_reserves_then_reconciles_provider_usage(tmp_path) -> None:
    guard = budget(tmp_path / "ledger.json")
    reservation = guard.reserve(input_token_upper_bound=10_000, max_output_tokens=2_000)
    during = guard.status()
    assert during["spent_usd"] == 0.0
    assert during["reserved_usd"] == pytest.approx(0.0055)

    status = guard.commit(reservation, input_tokens=1_000, output_tokens=500)

    assert status["spent_usd"] == pytest.approx(0.001)
    assert status["reserved_usd"] == 0.0
    assert status["calls"] == 1
    assert status["input_tokens"] == 1_000
    assert status["output_tokens"] == 500
    assert (tmp_path / "ledger.json").stat().st_mode & 0o777 == 0o600


def test_budget_blocks_before_crossing_hard_ceiling(tmp_path) -> None:
    guard = budget(tmp_path / "ledger.json", hard=0.01)
    first = guard.reserve(input_token_upper_bound=0, max_output_tokens=5_000)
    guard.commit(first, input_tokens=0, output_tokens=5_000)

    with pytest.raises(LLMBudgetExceededError, match="hard LLM budget"):
        guard.reserve(input_token_upper_bound=0, max_output_tokens=2_000)


def test_concurrent_reservations_cannot_oversubscribe_ceiling(tmp_path) -> None:
    guard = budget(tmp_path / "ledger.json", hard=0.005)

    def attempt() -> bool:
        try:
            guard.reserve(input_token_upper_bound=0, max_output_tokens=1_000)
            return True
        except LLMBudgetExceededError:
            return False

    with ThreadPoolExecutor(max_workers=12) as executor:
        accepted = list(executor.map(lambda _: attempt(), range(20)))

    assert sum(accepted) == 3
    status = guard.status()
    assert status["reserved_usd"] == pytest.approx(0.0045)
    assert float(status["spent_usd"]) + float(status["reserved_usd"]) <= 0.005


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_gemini_client_commits_reported_thinking_usage(tmp_path, monkeypatch) -> None:
    payload = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {
            "promptTokenCount": 1_000,
            "candidatesTokenCount": 200,
            "thoughtsTokenCount": 300,
            "totalTokenCount": 1_500,
        },
    }
    monkeypatch.setenv("EVOFOREST_LLM_MAX_TOKENS", "1000")
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _Response(payload))
    guard = budget(tmp_path / "ledger.json")
    client = GeminiLLMClient("secret", "gemini-test", budget=guard)

    assert client.complete("system", "user") == "ok"
    status = guard.status()
    assert status["input_tokens"] == 1_000
    assert status["output_tokens"] == 500
    assert status["spent_usd"] == pytest.approx(0.001)


def test_missing_usage_metadata_charges_reserved_worst_case(tmp_path, monkeypatch) -> None:
    payload = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    monkeypatch.setenv("EVOFOREST_LLM_MAX_TOKENS", "1000")
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _Response(payload))
    guard = budget(tmp_path / "ledger.json")
    client = GeminiLLMClient("secret", "gemini-test", budget=guard)

    client.complete("a", "b")
    status = guard.status()
    assert status["spent_usd"] == pytest.approx(guard.cost(1026, 1000))
    assert status["reserved_usd"] == 0.0
