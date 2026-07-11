"""Isolated LLM and external-agent baselines for the research suite.

No provider is selected implicitly.  Callers must inject an LLM client or an
external AIDE command, which keeps tests and ordinary benchmark runs offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from benchmarks.research_suite.baselines import RawRidge, _as_features, _as_target
from benchmarks.research_suite.metrics import rmse
from evoforest_arch.globals import GlobalStore
from evoforest_arch.graph import EvalContext, FeatureBlock
from evoforest_arch.source import SourceSandboxPolicy, build_source_alternative


class BaselineCapabilityError(RuntimeError):
    """Raised when an optional baseline executable or API is unavailable."""


class BaselineCredentialError(RuntimeError):
    """Raised when a requested baseline is missing explicit credentials."""


class FeatureResponseError(ValueError):
    """Raised when an LLM response does not satisfy the feature contract."""


class CompletionClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> object: ...


@dataclass(frozen=True)
class CompletionEnvelope:
    """Optional rich completion result for exact provider-side accounting."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float = 0.0


@dataclass(frozen=True)
class LLMCallUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    tokens_estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "LLMCallUsage") -> "LLMCallUsage":
        return LLMCallUsage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            tokens_estimated=self.tokens_estimated or other.tokens_estimated,
        )


UsageHook = Callable[[LLMCallUsage], None]
TokenCounter = Callable[[str], int]


def _estimated_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _require_client(client: CompletionClient | None) -> CompletionClient:
    if client is None:
        raise BaselineCredentialError(
            "An explicit configured LLM client is required; no provider or paid API is selected automatically."
        )
    if not callable(getattr(client, "complete", None)):
        raise BaselineCapabilityError("The LLM client must provide complete(system_prompt, user_prompt, temperature=...).")
    return client


def _complete(
    client: CompletionClient | None,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    token_counter: TokenCounter,
    usage_hook: UsageHook | None,
) -> tuple[str, LLMCallUsage]:
    result = _require_client(client).complete(system_prompt, user_prompt, temperature=temperature)
    estimated = not isinstance(result, CompletionEnvelope)
    envelope = result if isinstance(result, CompletionEnvelope) else CompletionEnvelope(str(result))
    if not envelope.text.strip():
        raise FeatureResponseError("The LLM returned an empty feature response.")
    prompt_text = system_prompt + "\n" + user_prompt
    prompt_tokens = envelope.prompt_tokens
    completion_tokens = envelope.completion_tokens
    usage = LLMCallUsage(
        calls=1,
        prompt_tokens=token_counter(prompt_text) if prompt_tokens is None else int(prompt_tokens),
        completion_tokens=token_counter(envelope.text) if completion_tokens is None else int(completion_tokens),
        cost_usd=float(envelope.cost_usd),
        tokens_estimated=estimated or prompt_tokens is None or completion_tokens is None,
    )
    if usage.prompt_tokens < 0 or usage.completion_tokens < 0 or usage.cost_usd < 0:
        raise ValueError("Completion usage values must be non-negative.")
    if usage_hook is not None:
        usage_hook(usage)
    return envelope.text, usage


def _feature_source(response: str) -> str:
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FeatureResponseError(f"Feature response is not valid JSON: {exc.msg}.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("source"), str):
            raise FeatureResponseError("Feature JSON must contain a string field named 'source'.")
        text = payload["source"].strip()
    if not text.startswith("lambda "):
        raise FeatureResponseError(
            "Feature response must be a lambda or JSON with source='lambda ctx, values: FeatureBlock(...)'."
        )
    return text


@dataclass
class _GeneratedFeatureMap:
    source: str
    sandbox_policy: SourceSandboxPolicy

    def transform(self, x: np.ndarray) -> np.ndarray:
        features = _as_features(x)
        alternative = build_source_alternative(
            "llm_baseline_features",
            (),
            self.source,
            node_kind="intermediate",
            output_contract={"type": "feature_block", "min_columns": 1},
            sandbox_policy=self.sandbox_policy,
        )
        result = alternative.fn(EvalContext(inputs={"x": features}, globals=GlobalStore()), {"x": features})
        values = result.values if isinstance(result, FeatureBlock) else np.asarray(result, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[0] != features.shape[0] or not np.all(np.isfinite(values)):
            raise FeatureResponseError("Generated features must be a finite 2-D matrix with one row per sample.")
        return values


@dataclass
class _FittedGeneratedRegressor:
    feature_map: _GeneratedFeatureMap
    ridge: RawRidge
    n_input_features: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        raw = _as_features(x, expected_columns=self.n_input_features)
        return self.ridge.predict(np.column_stack((raw, self.feature_map.transform(raw))))


_SYSTEM_PROMPT = """You are a feature-engineering baseline for numeric regression.
Return only JSON with one key, \"source\". Its value must be a deterministic Python lambda expression:
lambda ctx, values: FeatureBlock(<finite numpy matrix>, [<one name per column>])
The input matrix is values[\"x\"]. Use only numpy as np and FeatureBlock. Do not fit a predictor or access files."""


def _dataset_prompt(x: np.ndarray, y: np.ndarray, max_features: int) -> str:
    means = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
    stds = np.nanstd(np.where(np.isfinite(x), x, np.nan), axis=0)
    return (
        f"Training rows: {x.shape[0]}; input columns: {x.shape[1]}; maximum generated columns: {max_features}.\n"
        f"Input means: {np.nan_to_num(means).round(6).tolist()}\n"
        f"Input standard deviations: {np.nan_to_num(stds).round(6).tolist()}\n"
        f"Target mean: {float(np.mean(y)):.6g}; target standard deviation: {float(np.std(y)):.6g}.\n"
        "Propose a compact, general-purpose nonlinear feature block."
    )


@dataclass
class OneShotLLMFeatureRegressor:
    client: CompletionClient | None
    max_features: int = 16
    temperature: float = 0.0
    sandbox_policy: SourceSandboxPolicy = field(default_factory=SourceSandboxPolicy)
    token_counter: TokenCounter = _estimated_tokens
    usage_hook: UsageHook | None = None
    usage: LLMCallUsage = field(default_factory=LLMCallUsage, init=False)
    source_: str | None = field(default=None, init=False)
    model_: _FittedGeneratedRegressor | None = field(default=None, init=False, repr=False)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OneShotLLMFeatureRegressor":
        features = _as_features(x)
        target = _as_target(y, features.shape[0])
        if self.max_features <= 0:
            raise ValueError("max_features must be positive.")
        response, call_usage = _complete(
            self.client,
            _SYSTEM_PROMPT,
            _dataset_prompt(features, target, self.max_features),
            temperature=self.temperature,
            token_counter=self.token_counter,
            usage_hook=self.usage_hook,
        )
        source = _feature_source(response)
        fmap = _GeneratedFeatureMap(source, self.sandbox_policy)
        generated = fmap.transform(features)
        if generated.shape[1] > self.max_features:
            raise FeatureResponseError(f"Generated {generated.shape[1]} columns; maximum is {self.max_features}.")
        ridge = RawRidge().fit(np.column_stack((features, generated)), target)
        self.source_ = source
        self.model_ = _FittedGeneratedRegressor(fmap, ridge, features.shape[1])
        self.usage = self.usage + call_usage
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the baseline before predicting.")
        return self.model_.predict(x)


@dataclass
class IterativeScalarLLMFeatureRegressor:
    client: CompletionClient | None
    rounds: int = 4
    max_features: int = 16
    temperature: float = 0.0
    sandbox_policy: SourceSandboxPolicy = field(default_factory=SourceSandboxPolicy)
    token_counter: TokenCounter = _estimated_tokens
    usage_hook: UsageHook | None = None
    usage: LLMCallUsage = field(default_factory=LLMCallUsage, init=False)
    validation_scores_: list[float] = field(default_factory=list, init=False)
    source_: str | None = field(default=None, init=False)
    model_: _FittedGeneratedRegressor | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
    ) -> "IterativeScalarLLMFeatureRegressor":
        train_x = _as_features(x)
        train_y = _as_target(y, train_x.shape[0])
        valid_x = _as_features(validation_x, expected_columns=train_x.shape[1])
        valid_y = _as_target(validation_y, valid_x.shape[0])
        if self.rounds <= 0 or self.max_features <= 0:
            raise ValueError("rounds and max_features must be positive.")

        base_prompt = _dataset_prompt(train_x, train_y, self.max_features)
        best_score = math.inf
        best: _FittedGeneratedRegressor | None = None
        previous_source = "none"
        self.validation_scores_.clear()
        self.usage = LLMCallUsage()
        for round_index in range(self.rounds):
            feedback = (
                "This is the first proposal."
                if round_index == 0
                else f"Previous source: {previous_source}\nPrevious validation RMSE (scalar feedback only): {self.validation_scores_[-1]:.12g}."
            )
            response, call_usage = _complete(
                self.client,
                _SYSTEM_PROMPT,
                f"{base_prompt}\nRound: {round_index + 1}/{self.rounds}.\n{feedback}",
                temperature=self.temperature,
                token_counter=self.token_counter,
                usage_hook=self.usage_hook,
            )
            self.usage = self.usage + call_usage
            source = _feature_source(response)
            fmap = _GeneratedFeatureMap(source, self.sandbox_policy)
            generated = fmap.transform(train_x)
            if generated.shape[1] > self.max_features:
                raise FeatureResponseError(f"Generated {generated.shape[1]} columns; maximum is {self.max_features}.")
            ridge = RawRidge().fit(np.column_stack((train_x, generated)), train_y)
            candidate = _FittedGeneratedRegressor(fmap, ridge, train_x.shape[1])
            score = rmse(valid_y, candidate.predict(valid_x))
            self.validation_scores_.append(score)
            previous_source = source
            if score < best_score:
                best_score, best, self.source_ = score, candidate, source
        assert best is not None
        self.model_ = best
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Fit the baseline before predicting.")
        return self.model_.predict(x)


class CommandRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


def _run_command(
    command: Sequence[str], *, env: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, env=dict(env), timeout=timeout, check=False, text=True, capture_output=True)


@dataclass
class AIDECommandAdapter:
    """Adapter for an explicitly installed external AIDE-style command.

    Fit commands receive ``{dataset}`` (an NPZ with train/validation arrays) and
    ``{artifact_dir}``. Predict commands receive ``{input}``, ``{predictions}``,
    and ``{artifact_dir}``; they must write a one-dimensional ``.npy`` file.
    """

    fit_command: tuple[str, ...]
    predict_command: tuple[str, ...]
    required_env: tuple[str, ...] = ()
    timeout_seconds: float = 3600.0
    runner: CommandRunner = _run_command
    artifact_dir: Path | None = None
    command_calls: int = field(default=0, init=False)
    _temporary_directory: tempfile.TemporaryDirectory[str] | None = field(default=None, init=False, repr=False)

    def _check(self) -> None:
        if not self.fit_command or not self.predict_command:
            raise BaselineCapabilityError("AIDE fit_command and predict_command must both be configured.")
        for executable in dict.fromkeys((self.fit_command[0], self.predict_command[0])):
            if os.path.sep in executable:
                available = Path(executable).is_file() and os.access(executable, os.X_OK)
            else:
                available = shutil.which(executable) is not None
            if not available:
                raise BaselineCapabilityError(f"AIDE command executable is unavailable: {executable!r}.")
        missing = [name for name in self.required_env if not os.environ.get(name)]
        if missing:
            raise BaselineCredentialError(f"AIDE command is missing required environment variables: {', '.join(missing)}.")

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
    ) -> "AIDECommandAdapter":
        self._check()
        train_x = _as_features(x)
        train_y = _as_target(y, train_x.shape[0])
        valid_x = _as_features(validation_x, expected_columns=train_x.shape[1])
        valid_y = _as_target(validation_y, valid_x.shape[0])
        if self.artifact_dir is None:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="evoforest-aide-")
            self.artifact_dir = Path(self._temporary_directory.name)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        dataset = self.artifact_dir / "dataset.npz"
        np.savez(dataset, train_x=train_x, train_y=train_y, validation_x=valid_x, validation_y=valid_y)
        self._invoke(self.fit_command, {"dataset": dataset, "artifact_dir": self.artifact_dir})
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.artifact_dir is None:
            raise RuntimeError("Fit the AIDE adapter before predicting.")
        features = _as_features(x)
        input_path = self.artifact_dir / "predict_input.npy"
        predictions_path = self.artifact_dir / "predictions.npy"
        np.save(input_path, features)
        predictions_path.unlink(missing_ok=True)
        self._invoke(
            self.predict_command,
            {"input": input_path, "predictions": predictions_path, "artifact_dir": self.artifact_dir},
        )
        if not predictions_path.is_file():
            raise BaselineCapabilityError("AIDE predict command did not write the requested predictions file.")
        predictions = np.asarray(np.load(predictions_path, allow_pickle=False), dtype=float)
        if predictions.shape != (features.shape[0],) or not np.all(np.isfinite(predictions)):
            raise BaselineCapabilityError("AIDE predictions must be a finite one-dimensional array with one value per row.")
        return predictions

    def _invoke(self, template: Sequence[str], values: Mapping[str, Path]) -> None:
        try:
            command = tuple(part.format(**{key: str(value) for key, value in values.items()}) for part in template)
        except KeyError as exc:
            raise BaselineCapabilityError(f"AIDE command uses an unavailable placeholder: {exc.args[0]!r}.") from exc
        result = self.runner(command, env=os.environ, timeout=self.timeout_seconds)
        self.command_calls += 1
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "no diagnostic output").strip()[:1000]
            raise BaselineCapabilityError(f"AIDE command failed with exit code {result.returncode}: {detail}")
