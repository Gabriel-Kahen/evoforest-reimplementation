from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InputSpec:
    name: str
    kind: str = "numeric_matrix"
    description: str = ""
    shape: tuple[str, ...] = ("n_samples", "n_features")
    roles: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(dict.fromkeys(_normalize_role(role) for role in self.roles if str(role).strip())))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "shape": list(self.shape),
            "roles": list(self.roles),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InputSpec":
        roles_payload = payload.get("roles", ())
        if isinstance(roles_payload, str):
            roles = (roles_payload,)
        else:
            roles = tuple(str(role) for role in roles_payload)
        return cls(
            name=str(payload["name"]),
            kind=str(payload.get("kind", "numeric_matrix")),
            description=str(payload.get("description", "")),
            shape=tuple(str(item) for item in payload.get("shape", ("n_samples", "n_features"))),
            roles=roles,
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )

    def role_tokens(self) -> tuple[str, ...]:
        tokens = set(self.roles)
        kind = _normalize_role(self.kind)
        name = _normalize_role(self.name)
        tokens.add(kind)
        if kind in {"group_id", "unit_id", "engine_id", "entity_id"} or name in {"group", "group_id", "unit", "unit_id", "engine", "engine_id"}:
            tokens.update({"group", "unit"})
        if kind in {"time_index", "cycle_index", "timestamp", "sequence_index"} or name in {"time", "time_index", "cycle", "cycle_index", "timestamp"}:
            tokens.update({"time", "sequence"})
        if kind in {"regime_id", "degradation_regime"} or name in {"regime", "regime_id", "degradation_regime"}:
            tokens.add("regime")
        if kind in {"fault_mode", "fault_id"} or name in {"fault", "fault_mode", "fault_id"}:
            tokens.add("fault_mode")
        if kind in {"event_label", "event_indicator"} or name in {"event", "event_label", "event_indicator"}:
            tokens.add("event")
        if kind in {"censoring_indicator", "censored"} or name in {"censoring", "censoring_indicator", "censored"}:
            tokens.add("censoring")
        if kind in {"numeric_matrix", "time_series_matrix"}:
            tokens.add("feature")
        return tuple(sorted(tokens))

    def has_role(self, *roles: str) -> bool:
        wanted = {_normalize_role(role) for role in roles}
        return bool(wanted & set(self.role_tokens()))


@dataclass(frozen=True)
class TaskSchema:
    name: str
    kind: str
    inputs: tuple[InputSpec, ...]
    target: InputSpec = InputSpec("y", "numeric_target", "1-D target aligned with n_samples.", ("n_samples",))
    default_input: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [item.name for item in self.inputs]
        if not names:
            raise ValueError("TaskSchema requires at least one input.")
        if len(set(names)) != len(names):
            raise ValueError("TaskSchema input names must be unique.")
        if self.default_input and self.default_input not in names:
            raise ValueError(f"default_input {self.default_input!r} is not one of the task inputs.")
        if not self.default_input:
            object.__setattr__(self, "default_input", names[0])
        object.__setattr__(self, "metadata", dict(self.metadata))

    def input(self, name: str) -> InputSpec:
        for spec in self.inputs:
            if spec.name == name:
                return spec
        raise KeyError(f"Unknown task input {name!r}.")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inputs": [item.to_dict() for item in self.inputs],
            "target": self.target.to_dict(),
            "default_input": self.default_input,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSchema":
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            inputs=tuple(InputSpec.from_dict(dict(item)) for item in payload.get("inputs", [])),
            target=InputSpec.from_dict(dict(payload.get("target", InputSpec("y").to_dict()))),
            default_input=str(payload.get("default_input", "")),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        )

    def inputs_with_role(self, *roles: str) -> tuple[InputSpec, ...]:
        return tuple(spec for spec in self.inputs if spec.has_role(*roles))

    def input_name_with_role(self, *roles: str) -> str | None:
        matches = self.inputs_with_role(*roles)
        return matches[0].name if matches else None

    def role_map(self) -> dict[str, list[str]]:
        rows: dict[str, list[str]] = {}
        for spec in self.inputs:
            for role in spec.role_tokens():
                rows.setdefault(role, []).append(spec.name)
        return rows

    @classmethod
    def tabular(cls, *, input_name: str = "x", name: str = "generic-tabular") -> "TaskSchema":
        return cls(
            name=name,
            kind="tabular",
            inputs=(
                InputSpec(
                    input_name,
                    "numeric_matrix",
                    "Generic row-aligned numeric feature matrix with shape (n_samples, n_features).",
                    ("n_samples", "n_features"),
                    ("feature",),
                ),
            ),
            default_input=input_name,
        )

    @classmethod
    def structural_break(cls) -> "TaskSchema":
        return cls(
            name="synthetic-structural-break",
            kind="time_series_boundary",
            inputs=(
                InputSpec("series", "time_series_matrix", "Batch of univariate time series with shape (n_samples, length).", ("n_samples", "length"), ("feature", "sequence")),
                InputSpec("boundary", "scalar_index", "Boundary index separating the two task-specific time windows.", (), ("boundary",)),
            ),
            default_input="series",
        )


def task_schema_for_dataset(name: str) -> TaskSchema:
    if name == "synthetic-structural-break":
        return TaskSchema.structural_break()
    if name == "synthetic-tabular":
        return TaskSchema.tabular()
    raise ValueError(f"Unsupported dataset task schema {name!r}.")


def _normalize_role(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
