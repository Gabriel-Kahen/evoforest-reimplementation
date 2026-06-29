from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InputSpec:
    name: str
    kind: str = "numeric_matrix"
    description: str = ""
    shape: tuple[str, ...] = ("n_samples", "n_features")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InputSpec":
        return cls(
            name=str(payload["name"]),
            kind=str(payload.get("kind", "numeric_matrix")),
            description=str(payload.get("description", "")),
            shape=tuple(str(item) for item in payload.get("shape", ("n_samples", "n_features"))),
        )


@dataclass(frozen=True)
class TaskSchema:
    name: str
    kind: str
    inputs: tuple[InputSpec, ...]
    target: InputSpec = InputSpec("y", "numeric_target", "1-D target aligned with n_samples.", ("n_samples",))
    default_input: str = ""

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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSchema":
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            inputs=tuple(InputSpec.from_dict(dict(item)) for item in payload.get("inputs", [])),
            target=InputSpec.from_dict(dict(payload.get("target", InputSpec("y").to_dict()))),
            default_input=str(payload.get("default_input", "")),
        )

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
                InputSpec("series", "time_series_matrix", "Batch of univariate time series with shape (n_samples, length).", ("n_samples", "length")),
                InputSpec("boundary", "scalar_index", "Boundary index separating the two task-specific time windows.", ()),
            ),
            default_input="series",
        )


def task_schema_for_dataset(name: str) -> TaskSchema:
    if name == "synthetic-structural-break":
        return TaskSchema.structural_break()
    if name == "synthetic-tabular":
        return TaskSchema.tabular()
    raise ValueError(f"Unsupported dataset task schema {name!r}.")
