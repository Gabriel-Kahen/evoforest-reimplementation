from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid


class LLMBudgetExceededError(RuntimeError):
    """Raised before a request that could exceed the configured hard ceiling."""


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    maximum_cost_usd: float


@dataclass(frozen=True)
class LLMBudget:
    ledger_path: Path
    target_usd: float
    hard_limit_usd: float
    input_price_usd_per_million_tokens: float
    output_price_usd_per_million_tokens: float

    @classmethod
    def from_env(cls) -> "LLMBudget | None":
        hard_text = os.getenv("EVOFOREST_LLM_HARD_BUDGET_USD", "").strip()
        if not hard_text:
            return None
        budget = cls(
            ledger_path=Path(os.getenv("EVOFOREST_LLM_BUDGET_LEDGER", ".evoforest_llm_budget.json")),
            target_usd=_positive_env_float("EVOFOREST_LLM_TARGET_BUDGET_USD", 5.0),
            hard_limit_usd=_positive_env_float("EVOFOREST_LLM_HARD_BUDGET_USD", 20.0),
            input_price_usd_per_million_tokens=_positive_env_float(
                "EVOFOREST_LLM_INPUT_PRICE_USD_PER_MILLION_TOKENS", 0.25
            ),
            output_price_usd_per_million_tokens=_positive_env_float(
                "EVOFOREST_LLM_OUTPUT_PRICE_USD_PER_MILLION_TOKENS", 1.50
            ),
        )
        if budget.target_usd > budget.hard_limit_usd:
            raise ValueError("EVOFOREST_LLM_TARGET_BUDGET_USD cannot exceed the hard budget.")
        return budget

    def reserve(self, *, input_token_upper_bound: int, max_output_tokens: int) -> BudgetReservation:
        maximum_cost = self.cost(input_token_upper_bound, max_output_tokens)
        reservation_id = uuid.uuid4().hex
        with self._locked_ledger() as ledger:
            committed = float(ledger["spent_usd"])
            reserved = sum(float(value) for value in ledger["reservations"].values())
            if committed + reserved + maximum_cost > self.hard_limit_usd + 1e-12:
                raise LLMBudgetExceededError(
                    "Gemini request blocked by the hard LLM budget: "
                    f"spent=${committed:.4f}, reserved=${reserved:.4f}, "
                    f"next-call maximum=${maximum_cost:.4f}, limit=${self.hard_limit_usd:.2f}."
                )
            ledger["reservations"][reservation_id] = maximum_cost
        return BudgetReservation(reservation_id, maximum_cost)

    def commit(
        self,
        reservation: BudgetReservation,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> dict[str, object]:
        usage_known = input_tokens is not None and output_tokens is not None
        charged = (
            self.cost(max(0, int(input_tokens)), max(0, int(output_tokens)))
            if usage_known
            else reservation.maximum_cost_usd
        )
        with self._locked_ledger() as ledger:
            reserved = float(ledger["reservations"].pop(reservation.reservation_id, reservation.maximum_cost_usd))
            # Never charge less than zero or more than the pre-reserved amount. The
            # reservation is deliberately conservative and is the fail-closed value
            # when provider usage metadata is absent or internally inconsistent.
            charged = min(max(0.0, charged), reserved)
            ledger["spent_usd"] = float(ledger["spent_usd"]) + charged
            ledger["calls"] = int(ledger["calls"]) + 1
            if usage_known:
                ledger["input_tokens"] = int(ledger["input_tokens"]) + max(0, int(input_tokens))
                ledger["output_tokens"] = int(ledger["output_tokens"]) + max(0, int(output_tokens))
            status = self._public_status(ledger)
        if float(status["spent_usd"]) >= self.target_usd:
            print(
                f"LLM target budget reached: ${float(status['spent_usd']):.4f} spent "
                f"of ${self.hard_limit_usd:.2f} hard limit.",
                file=sys.stderr,
                flush=True,
            )
        return status

    def release(self, reservation: BudgetReservation) -> None:
        with self._locked_ledger() as ledger:
            ledger["reservations"].pop(reservation.reservation_id, None)

    def status(self) -> dict[str, object]:
        with self._locked_ledger() as ledger:
            return self._public_status(ledger)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            max(0, int(input_tokens)) * self.input_price_usd_per_million_tokens
            + max(0, int(output_tokens)) * self.output_price_usd_per_million_tokens
        ) / 1_000_000.0

    def _public_status(self, ledger: dict[str, object]) -> dict[str, object]:
        reserved = sum(float(value) for value in ledger["reservations"].values())  # type: ignore[union-attr]
        return {
            "spent_usd": float(ledger["spent_usd"]),
            "reserved_usd": reserved,
            "target_usd": self.target_usd,
            "hard_limit_usd": self.hard_limit_usd,
            "remaining_unreserved_usd": max(0.0, self.hard_limit_usd - float(ledger["spent_usd"]) - reserved),
            "calls": int(ledger["calls"]),
            "input_tokens": int(ledger["input_tokens"]),
            "output_tokens": int(ledger["output_tokens"]),
            "ledger_path": str(self.ledger_path),
        }

    class _LedgerLock:
        def __init__(self, outer: "LLMBudget") -> None:
            self.outer = outer
            self.handle = None
            self.ledger: dict[str, object] = {}

        def __enter__(self) -> dict[str, object]:
            path = self.outer.ledger_path
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            os.chmod(path, 0o600)
            self.handle = os.fdopen(descriptor, "r+", encoding="utf-8")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            raw = self.handle.read().strip()
            self.ledger = json.loads(raw) if raw else _empty_ledger()
            _validate_ledger(self.ledger)
            return self.ledger

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            assert self.handle is not None
            if exc_type is None:
                self.handle.seek(0)
                json.dump(self.ledger, self.handle, indent=2, sort_keys=True)
                self.handle.write("\n")
                self.handle.truncate()
                self.handle.flush()
                os.fsync(self.handle.fileno())
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def _locked_ledger(self) -> "LLMBudget._LedgerLock":
        return LLMBudget._LedgerLock(self)


def _empty_ledger() -> dict[str, object]:
    return {
        "version": 1,
        "spent_usd": 0.0,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reservations": {},
    }


def _validate_ledger(ledger: dict[str, object]) -> None:
    required = {"version", "spent_usd", "calls", "input_tokens", "output_tokens", "reservations"}
    if set(ledger) != required or ledger.get("version") != 1 or not isinstance(ledger.get("reservations"), dict):
        raise ValueError("Invalid EvoForest LLM budget ledger.")


def _positive_env_float(name: str, default: float) -> float:
    text = os.getenv(name, "").strip()
    try:
        value = float(text) if text else float(default)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive number.")
    return value
