from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GlobalParameter:
    name: str
    value: np.ndarray
    trainable: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.value.shape),
            "trainable": self.trainable,
            "description": self.description,
            "value": self.value.tolist(),
        }


class GlobalStore:
    def __init__(self) -> None:
        self._params: dict[str, GlobalParameter] = {}

    def add(
        self,
        name: str,
        value: np.ndarray | list[float],
        trainable: bool = True,
        description: str = "",
    ) -> None:
        self._params[name] = GlobalParameter(
            name=name,
            value=np.asarray(value, dtype=np.float64),
            trainable=trainable,
            description=description,
        )

    def get(self, name: str) -> np.ndarray:
        if name not in self._params:
            raise KeyError(f"Unknown global parameter {name!r}.")
        return self._params[name].value

    def set(self, name: str, value: np.ndarray | list[float]) -> None:
        if name not in self._params:
            raise KeyError(f"Unknown global parameter {name!r}.")
        self._params[name].value = np.asarray(value, dtype=np.float64)

    def remove(self, name: str) -> None:
        if name not in self._params:
            raise KeyError(f"Unknown global parameter {name!r}.")
        del self._params[name]

    def names(self) -> list[str]:
        return sorted(self._params)

    def trainable_names(self) -> list[str]:
        return sorted(name for name, param in self._params.items() if param.trainable)

    def to_dict(self) -> dict[str, object]:
        return {name: param.to_dict() for name, param in sorted(self._params.items())}

    def clone(self) -> "GlobalStore":
        cloned = GlobalStore()
        for name, param in self._params.items():
            cloned.add(
                name=name,
                value=param.value.copy(),
                trainable=param.trainable,
                description=param.description,
            )
        return cloned
