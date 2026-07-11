"""Secret-free, fingerprinted execution configuration for confirmatory runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Mapping


SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).parent / "specs" / "execution_config_v1.schema.json"
FROZEN_SCHEMA_SHA256 = "605560c9cbd9596319b21e8b8a901b6cf599bfbf8139984bb3aca4d558e49ba5"
_SECRET_VALUE_PATTERN = re.compile(r"(?i)(api[-_]?key|access[-_]?token|password|secret)\s*=")


@dataclass(frozen=True)
class LLMExecutionConfig:
    provider: str
    model_identifier: str
    temperature: float
    call_budget: int
    token_budget: int
    input_price_usd_per_million_tokens: float
    output_price_usd_per_million_tokens: float
    credential_env_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model_identifier.strip():
            raise ValueError("LLM provider and model_identifier must be non-empty.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("LLM temperature must be between 0 and 2.")
        if self.call_budget <= 0 or self.token_budget <= 0:
            raise ValueError("LLM call and token budgets must be positive.")
        if self.input_price_usd_per_million_tokens < 0 or self.output_price_usd_per_million_tokens < 0:
            raise ValueError("LLM pricing metadata cannot be negative.")
        _validate_env_names(self.credential_env_names, required=True)


@dataclass(frozen=True)
class AIDEExecutionConfig:
    fit_command: tuple[str, ...]
    predict_command: tuple[str, ...]
    timeout_seconds: float
    credential_env_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.fit_command or not self.predict_command:
            raise ValueError("AIDE fit and predict commands must be non-empty.")
        if self.timeout_seconds <= 0:
            raise ValueError("AIDE timeout_seconds must be positive.")
        _validate_env_names(self.credential_env_names, required=False)
        for argument in (*self.fit_command, *self.predict_command):
            if _SECRET_VALUE_PATTERN.search(argument):
                raise ValueError("AIDE commands cannot embed credential values; reference environment variables by name.")


@dataclass(frozen=True)
class ExecutionConfig:
    name: str
    version: int
    llm: LLMExecutionConfig
    aide: AIDEExecutionConfig
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name.strip() or self.version <= 0:
            raise ValueError("Execution config name and version must be positive/non-empty.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported execution-config schema version {self.schema_version}.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def credential_presence(self, environ: Mapping[str, str] | None = None) -> dict[str, dict[str, bool]]:
        source = os.environ if environ is None else environ
        return {
            "llm": {name: bool(source.get(name)) for name in self.llm.credential_env_names},
            "aide": {name: bool(source.get(name)) for name in self.aide.credential_env_names},
        }


@dataclass(frozen=True)
class ExecutionCapability:
    available: bool
    detail: str
    credential_presence: Mapping[str, bool]


def execution_capability_report(
    config: ExecutionConfig,
    environ: Mapping[str, str] | None = None,
) -> dict[str, ExecutionCapability]:
    presence = config.credential_presence(environ)
    llm_missing = sorted(name for name, present in presence["llm"].items() if not present)
    llm = ExecutionCapability(
        available=not llm_missing,
        detail=(
            f"Configured provider/model {config.llm.provider}/{config.llm.model_identifier}; credential names are present."
            if not llm_missing
            else f"Missing credential environment variables: {', '.join(llm_missing)}."
        ),
        credential_presence=presence["llm"],
    )

    missing_commands = [
        command[0]
        for command in (config.aide.fit_command, config.aide.predict_command)
        if not _executable_available(command[0])
    ]
    aide_missing_credentials = sorted(name for name, present in presence["aide"].items() if not present)
    aide_available = not missing_commands and not aide_missing_credentials
    aide_problems: list[str] = []
    if missing_commands:
        aide_problems.append(f"missing executables: {', '.join(dict.fromkeys(missing_commands))}")
    if aide_missing_credentials:
        aide_problems.append(f"missing credential environment variables: {', '.join(aide_missing_credentials)}")
    aide = ExecutionCapability(
        available=aide_available,
        detail="AIDE fit/predict commands and credential names are available." if aide_available else "; ".join(aide_problems) + ".",
        credential_presence=presence["aide"],
    )
    return {"llm": llm, "aide": aide}


def load_execution_config(path: str | Path) -> ExecutionConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Execution config must be a JSON object.")
    allowed = {"name", "version", "schema_version", "llm", "aide", "fingerprint"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown execution-config fields: {', '.join(unknown)}")
    llm_payload = payload.get("llm")
    aide_payload = payload.get("aide")
    if not isinstance(llm_payload, dict) or not isinstance(aide_payload, dict):
        raise ValueError("Execution config requires llm and aide objects.")
    config = ExecutionConfig(
        name=str(payload.get("name", "")),
        version=int(payload.get("version", 0)),
        schema_version=int(payload.get("schema_version", 0)),
        llm=LLMExecutionConfig(**_tuplify(llm_payload, "credential_env_names")),
        aide=AIDEExecutionConfig(**_tuplify(aide_payload, "fit_command", "predict_command", "credential_env_names")),
    )
    expected = payload.get("fingerprint")
    if expected is not None and expected != config.fingerprint():
        raise ValueError("Execution config fingerprint does not match its contents.")
    return config


def write_execution_config(path: str | Path, config: ExecutionConfig) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {**config.to_dict(), "fingerprint": config.fingerprint()}
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def schema_fingerprint() -> str:
    return hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()


def schema_lock_valid() -> bool:
    return schema_fingerprint() == FROZEN_SCHEMA_SHA256


def _validate_env_names(names: tuple[str, ...], *, required: bool) -> None:
    if required and not names:
        raise ValueError("At least one credential environment-variable name is required.")
    invalid = [name for name in names if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)]
    if invalid:
        raise ValueError(f"Invalid credential environment-variable names: {', '.join(invalid)}")


def _tuplify(payload: Mapping[str, object], *names: str) -> dict[str, object]:
    result = dict(payload)
    for name in names:
        if name in result:
            if not isinstance(result[name], list):
                raise ValueError(f"{name} must be a JSON array.")
            result[name] = tuple(str(value) for value in result[name])
    return result


def _executable_available(executable: str) -> bool:
    if os.path.sep in executable:
        path = Path(executable)
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(executable) is not None
