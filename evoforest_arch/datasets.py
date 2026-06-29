from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import pathlib
from typing import Any, Callable

import numpy as np

from evoforest_arch.synthetic import make_structural_break_data, make_tabular_data
from evoforest_arch.task import TaskSchema, task_schema_for_dataset


DatasetLoader = Callable[[dict[str, Any]], "LoadedDataset"]


@dataclass(frozen=True)
class LoadedDataset:
    inputs: dict[str, object]
    y: np.ndarray
    metadata: dict[str, Any]
    task_schema: TaskSchema


class DatasetLoaderRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, DatasetLoader] = {}

    def register(self, name: str, loader: DatasetLoader) -> None:
        self._loaders[name] = loader

    def load(self, config: dict[str, Any]) -> LoadedDataset:
        name = str(config.get("name", "synthetic-structural-break"))
        if name not in self._loaders:
            raise ValueError(f"Unsupported dataset {name!r}; available loaders: {sorted(self._loaders)}.")
        return self._loaders[name](config)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaders))


def default_dataset_loader_registry() -> DatasetLoaderRegistry:
    registry = DatasetLoaderRegistry()
    registry.register("synthetic-structural-break", _load_structural_break)
    registry.register("synthetic-tabular", _load_tabular)
    registry.register("external-npz", _load_external_npz)
    registry.register("external-manifest", _load_external_manifest)
    registry.register("python-module", _load_python_module)
    return registry


def load_dataset_bundle(config: dict[str, Any], registry: DatasetLoaderRegistry | None = None) -> LoadedDataset:
    return (registry or default_dataset_loader_registry()).load(config)


def _load_structural_break(config: dict[str, Any]) -> LoadedDataset:
    dataset = make_structural_break_data(
        n_series=int(config.get("n_series", 240)),
        length=int(config.get("length", 160)),
        boundary=config.get("boundary"),
        seed=int(config.get("seed", 0)),
    )
    return LoadedDataset(
        inputs=dataset.inputs(),
        y=dataset.y,
        task_schema=TaskSchema.structural_break(),
        metadata={
            "name": "synthetic-structural-break",
            "n_samples": int(dataset.y.shape[0]),
            "target_mean": float(np.mean(dataset.y)),
            "target_std": float(np.std(dataset.y)),
            "target_min": float(np.min(dataset.y)),
            "target_max": float(np.max(dataset.y)),
            "mapping": "synthetic structural break generator",
        },
    )


def _load_tabular(config: dict[str, Any]) -> LoadedDataset:
    dataset = make_tabular_data(
        n_samples=int(config.get("n_samples", config.get("n_series", 240))),
        n_features=int(config.get("n_features", 12)),
        seed=int(config.get("seed", 0)),
    )
    return LoadedDataset(
        inputs=dataset.inputs(),
        y=dataset.y,
        task_schema=TaskSchema.tabular(),
        metadata={
            "name": "synthetic-tabular",
            "n_samples": int(dataset.y.shape[0]),
            "n_features": int(dataset.x.shape[1]),
            "target_mean": float(np.mean(dataset.y)),
            "target_std": float(np.std(dataset.y)),
            "target_min": float(np.min(dataset.y)),
            "target_max": float(np.max(dataset.y)),
            "mapping": "synthetic generic tabular generator",
        },
    )


def _load_external_manifest(config: dict[str, Any]) -> LoadedDataset:
    raw_manifest_path = str(config.get("manifest_path", config.get("path", ""))).strip()
    if not raw_manifest_path:
        raise ValueError("external-manifest datasets require manifest_path.")
    manifest_path = pathlib.Path(raw_manifest_path).expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    merged = {**manifest, **{key: value for key, value in config.items() if key not in {"name", "manifest_path"}}}
    adapter = str(merged.get("adapter", merged.get("name", "external-npz")))
    if adapter == "python-module":
        return _load_python_module({**merged, "base_dir": str(base_dir), "name": "python-module"})
    return _load_external_npz({**merged, "base_dir": str(base_dir), "name": "external-npz"})


def _load_external_npz(config: dict[str, Any]) -> LoadedDataset:
    base_dir = pathlib.Path(str(config.get("base_dir", "."))).expanduser()
    path = pathlib.Path(str(config.get("path", config.get("npz_path", "")))).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"External npz dataset does not exist: {path}")
    target_key = str(config.get("target_key", "y"))
    input_keys = tuple(str(key) for key in config.get("input_keys", ()))
    with np.load(path, allow_pickle=False) as data:
        if target_key not in data:
            raise KeyError(f"Target key {target_key!r} not found in {path}.")
        y = np.asarray(data[target_key], dtype=np.float64).reshape(-1)
        keys = input_keys or tuple(key for key in data.files if key != target_key)
        inputs = {key: np.asarray(data[key]) for key in keys}
    schema_config = dict(config)
    if keys and not schema_config.get("default_input") and not schema_config.get("input_name"):
        schema_config["default_input"] = keys[0]
    task_schema = _task_schema_from_config(schema_config, default_name=str(config.get("task_name", "external-npz")))
    metadata = _metadata(y, name=str(config.get("dataset_name", "external-npz")))
    metadata.update({"path": str(path), "target_key": target_key, "input_keys": list(inputs)})
    return LoadedDataset(inputs=inputs, y=y, metadata=metadata, task_schema=task_schema)


def _load_python_module(config: dict[str, Any]) -> LoadedDataset:
    module_name = str(config.get("module", ""))
    function_name = str(config.get("function", "load_dataset"))
    if not module_name:
        raise ValueError("python-module datasets require module.")
    module = importlib.import_module(module_name)
    loader = getattr(module, function_name)
    kwargs = dict(config.get("kwargs", {})) if isinstance(config.get("kwargs", {}), dict) else {}
    result = loader(**kwargs)
    return _coerce_loaded_dataset(result, config)


def _coerce_loaded_dataset(result: object, config: dict[str, Any]) -> LoadedDataset:
    if isinstance(result, LoadedDataset):
        return result
    if isinstance(result, tuple) and len(result) in {2, 3, 4}:
        inputs = result[0]
        y = result[1]
        metadata = result[2] if len(result) >= 3 else {}
        task_schema = result[3] if len(result) >= 4 else None
    elif isinstance(result, dict):
        inputs = result["inputs"]
        y = result["y"]
        metadata = result.get("metadata", {})
        task_schema = result.get("task_schema")
    else:
        raise TypeError("Python dataset loader must return LoadedDataset, dict, or (inputs, y[, metadata[, task_schema]]).")
    if not isinstance(inputs, dict):
        raise TypeError("Dataset inputs must be a dictionary.")
    schema = _coerce_task_schema(task_schema, config)
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    merged_metadata = _metadata(y_array, name=str(config.get("dataset_name", config.get("name", "python-module"))))
    if isinstance(metadata, dict):
        merged_metadata.update(metadata)
    return LoadedDataset(inputs=dict(inputs), y=y_array, metadata=merged_metadata, task_schema=schema)


def _task_schema_from_config(config: dict[str, Any], *, default_name: str) -> TaskSchema:
    return _coerce_task_schema(config.get("task_schema"), {**config, "task_name": default_name})


def _coerce_task_schema(payload: object, config: dict[str, Any]) -> TaskSchema:
    if isinstance(payload, TaskSchema):
        return payload
    if isinstance(payload, dict):
        return TaskSchema.from_dict(payload)
    if isinstance(payload, str) and payload:
        path = pathlib.Path(payload).expanduser()
        if path.exists():
            return TaskSchema.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return task_schema_for_dataset(payload)
    input_name = str(config.get("default_input", config.get("input_name", "x")))
    return TaskSchema.tabular(input_name=input_name, name=str(config.get("task_name", "external-tabular")))


def _metadata(y: np.ndarray, *, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "n_samples": int(y.shape[0]),
        "target_mean": float(np.mean(y)) if y.size else 0.0,
        "target_std": float(np.std(y)) if y.size else 0.0,
        "target_min": float(np.min(y)) if y.size else 0.0,
        "target_max": float(np.max(y)) if y.size else 0.0,
    }
