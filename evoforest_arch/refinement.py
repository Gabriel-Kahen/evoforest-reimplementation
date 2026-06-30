from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evoforest_arch.graph import Graph, NodeAlternative
from evoforest_arch.readout import Standardizer, select_alpha_and_fit_ridge


@dataclass
class RefinementResult:
    enabled: bool
    steps: int
    initial_loss: float
    final_loss: float
    accepted_updates: int
    trainable_globals: list[str]
    backend: str = "numpy_coordinate"
    requested_backend: str = "auto"
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "accepted_updates": self.accepted_updates,
            "trainable_globals": self.trainable_globals,
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "fallback_reason": self.fallback_reason,
        }


class GlobalRefiner:
    """Global refinement phase with optional PyTorch L-BFGS and NumPy fallback."""

    def __init__(
        self,
        steps: int = 20,
        seed: int = 0,
        initial_step_size: float = 0.1,
        backend: str = "auto",
        device: str | None = None,
    ) -> None:
        self.steps = max(0, int(steps))
        self.seed = int(seed)
        self.initial_step_size = float(initial_step_size)
        if backend not in {"auto", "torch", "numpy"}:
            raise ValueError("Refinement backend must be one of: auto, torch, numpy.")
        self.backend = backend
        self.device = device

    def refine(self, graph: Graph, inputs: dict[str, object], y: np.ndarray, config: dict[str, str]) -> RefinementResult:
        if self.backend in {"auto", "torch"}:
            try:
                return TorchLBFGSRefiner(self.steps, self.backend, self.device).refine(graph, inputs, y, config)
            except UnsupportedTorchRefinement as exc:
                if self.backend == "torch":
                    raise
                loss = self._loss(graph, inputs, y, config)
                return RefinementResult(False, 0, loss, loss, 0, graph.globals.trainable_names(), "torch_l_bfgs", self.backend, str(exc))
        return self._refine_numpy(graph, inputs, y, config)

    def _refine_numpy(
        self,
        graph: Graph,
        inputs: dict[str, object],
        y: np.ndarray,
        config: dict[str, str],
        fallback_reason: str = "",
    ) -> RefinementResult:
        trainable = graph.globals.trainable_names()
        if not trainable or self.steps == 0:
            loss = self._loss(graph, inputs, y, config)
            return RefinementResult(False, 0, loss, loss, 0, trainable, "numpy_coordinate", self.backend, fallback_reason)

        initial_loss = self._loss(graph, inputs, y, config)
        best_loss = initial_loss
        accepted = 0
        coordinates = self._coordinates(graph, trainable)
        if not coordinates:
            return RefinementResult(False, 0, initial_loss, initial_loss, 0, trainable, "numpy_coordinate", self.backend, fallback_reason)

        for step in range(self.steps):
            name, flat_index = coordinates[step % len(coordinates)]
            value = graph.globals.get(name).copy()
            scale = self.initial_step_size / (1.0 + step / max(1, len(coordinates)))
            best_value = value
            for direction in (1.0, -1.0):
                candidate = value.copy()
                candidate.reshape(-1)[flat_index] += direction * scale
                graph.globals.set(name, candidate)
                candidate_loss = self._loss(graph, inputs, y, config)
                if candidate_loss + 1e-12 < best_loss:
                    best_loss = candidate_loss
                    best_value = candidate.copy()
            graph.globals.set(name, best_value)
            if best_value is not value and np.max(np.abs(best_value - value)) > 0.0:
                accepted += 1

        return RefinementResult(True, self.steps, float(initial_loss), float(best_loss), accepted, trainable, "numpy_coordinate", self.backend, fallback_reason)

    @staticmethod
    def _coordinates(graph: Graph, trainable: list[str]) -> list[tuple[str, int]]:
        coordinates: list[tuple[str, int]] = []
        for name in trainable:
            value = graph.globals.get(name)
            coordinates.extend((name, index) for index in range(value.size))
        return coordinates

    @staticmethod
    def _loss(graph: Graph, inputs: dict[str, object], y: np.ndarray, config: dict[str, str]) -> float:
        x, _, _ = graph.evaluate_features(inputs, config=config)
        y = np.asarray(y, dtype=np.float64)
        std = Standardizer.fit(x)
        xs = std.transform(x)
        alpha, model = select_alpha_and_fit_ridge(xs, y)
        residual = y - model.predict(xs)
        return float(np.mean(residual**2))


class UnsupportedTorchRefinement(RuntimeError):
    pass


class TorchGlobalProxy:
    def __init__(self, graph: Graph, parameters: Any, torch_module: Any, device: Any = None) -> None:
        self.graph = graph
        self.parameters = parameters
        self.torch = torch_module
        self.device = device

    def get(self, name: str) -> Any:
        if name in self.parameters:
            return self.parameters[name]
        return self.torch.as_tensor(self.graph.globals.get(name), dtype=self.torch.float64, device=self.device)


class TorchEvalContext:
    def __init__(self, inputs: dict[str, object], globals_proxy: TorchGlobalProxy) -> None:
        self.inputs = inputs
        self.globals = globals_proxy

    def read_input(self, name: str) -> Any:
        if name not in self.inputs:
            raise KeyError(f"Input {name!r} was not supplied.")
        return self.inputs[name]


class TorchFixedPathEvaluator:
    def __init__(
        self,
        graph: Graph,
        inputs: dict[str, object],
        config: dict[str, str],
        parameters: Any,
        torch_module: Any,
        device: Any = None,
    ) -> None:
        self.graph = graph
        self.config = graph.selected_config(config)
        self.torch = torch_module
        self.device = device
        self.inputs = {name: self._torch_input(value) for name, value in inputs.items()}
        self.ctx = TorchEvalContext(self.inputs, TorchGlobalProxy(graph, parameters, self.torch, self.device))
        self.cache: dict[tuple[str, str], Any] = {}

    def _torch_input(self, value: object) -> object:
        if not isinstance(value, np.ndarray):
            return value
        if not (np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_)):
            return value
        return self.torch.as_tensor(value, dtype=self.torch.float64, device=self.device)

    def evaluate_features(self) -> Any:
        blocks = []
        for node_name in self.graph.output_nodes():
            node = self.graph.nodes[node_name]
            for alternative in node.alternatives:
                value = self.evaluate_alternative(node_name, alternative.id)
                if value.ndim == 1:
                    value = value.reshape(-1, 1)
                if value.ndim != 2:
                    raise UnsupportedTorchRefinement(f"Torch output {node_name}.{alternative.id} returned shape {tuple(value.shape)}.")
                blocks.append(value)
        if not blocks:
            raise UnsupportedTorchRefinement("Graph has no output alternatives for torch refinement.")
        return self.torch.cat(blocks, dim=1)

    def evaluate_node(self, name: str) -> Any:
        node = self.graph.nodes[name]
        if node.kind == "input":
            return self.ctx.read_input(name)
        alt_id = self.config.get(name, node.default_alternative_id())
        return self.evaluate_alternative(name, alt_id)

    def evaluate_alternative(self, name: str, alternative_id: str) -> Any:
        key = (name, alternative_id)
        if key in self.cache:
            return self.cache[key]
        node = self.graph.nodes[name]
        alternative = self._alternative(node.alternatives, alternative_id, name)
        if alternative.torch_fn is None:
            raise UnsupportedTorchRefinement(f"Alternative {name}.{alternative_id} has no torch evaluator.")
        parent_values = {parent: self.evaluate_node(parent) for parent in alternative.parents}
        value = alternative.torch_fn(self.ctx, parent_values)
        self.cache[key] = value
        return value

    @staticmethod
    def _alternative(alternatives: list[NodeAlternative], alternative_id: str, node_name: str) -> NodeAlternative:
        for alternative in alternatives:
            if alternative.id == alternative_id:
                return alternative
        raise UnsupportedTorchRefinement(f"Node {node_name!r} has no alternative {alternative_id!r}.")


class TorchLBFGSRefiner:
    def __init__(self, steps: int, requested_backend: str, device: str | None = None) -> None:
        self.steps = max(0, int(steps))
        self.requested_backend = requested_backend
        self.device = device

    def refine(self, graph: Graph, inputs: dict[str, object], y: np.ndarray, config: dict[str, str]) -> RefinementResult:
        try:
            import torch
        except ImportError as exc:
            raise UnsupportedTorchRefinement("PyTorch is not installed; install the torch extra to enable L-BFGS refinement.") from exc
        device = torch.device(self.device) if self.device else None

        trainable = graph.globals.trainable_names()
        if not trainable or self.steps == 0:
            loss = GlobalRefiner(steps=0, backend="numpy")._loss(graph, inputs, y, config)
            return RefinementResult(False, 0, loss, loss, 0, trainable, "torch_l_bfgs", self.requested_backend)

        referenced = selected_output_global_refs(graph, config)
        active = [name for name in trainable if name in referenced]
        if not active:
            raise UnsupportedTorchRefinement("No selected output path references trainable globals.")

        torch.manual_seed(0)
        parameters = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(
                    torch.as_tensor(graph.globals.get(name), dtype=torch.float64, device=device).clone()
                )
                for name in active
            }
        )
        target = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64, device=device)
        probe = TorchFixedPathEvaluator(graph, inputs, config, parameters, torch, device)
        with torch.no_grad():
            initial_features = probe.evaluate_features()
            n_features = int(initial_features.shape[1])
        readout = torch.nn.Linear(n_features, 1, bias=True, dtype=torch.float64, device=device)
        readout.bias.data.fill_(float(torch.mean(target)))
        optimizer = torch.optim.LBFGS(
            list(parameters.parameters()) + list(readout.parameters()),
            lr=1.0,
            max_iter=self.steps,
            line_search_fn="strong_wolfe",
        )

        def loss_value() -> Any:
            evaluator = TorchFixedPathEvaluator(graph, inputs, config, parameters, torch, device)
            features = evaluator.evaluate_features()
            mean = torch.mean(features, dim=0, keepdim=True)
            scale = torch.clamp(torch.std(features, dim=0, unbiased=False, keepdim=True), min=1e-8)
            standardized = (features - mean) / scale
            pred = readout(standardized).reshape(-1)
            return torch.mean((pred - target) ** 2)

        with torch.no_grad():
            initial_loss = float(loss_value().detach().cpu())

        probe_loss = loss_value()
        probe_loss.backward()
        active_with_grad = [
            name
            for name, parameter in parameters.items()
            if parameter.grad is not None
            and bool(torch.all(torch.isfinite(parameter.grad)))
            and float(torch.max(torch.abs(parameter.grad)).detach().cpu()) > 1e-12
        ]
        optimizer.zero_grad()
        if not active_with_grad:
            return RefinementResult(
                False,
                0,
                initial_loss,
                initial_loss,
                0,
                trainable,
                "torch_l_bfgs",
                self.requested_backend,
                "Gradient probe found no active trainable global influence on the selected path.",
            )

        def closure() -> Any:
            optimizer.zero_grad()
            loss = loss_value()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            final_loss = float(loss_value().detach().cpu())
            changed = 0
            for name, parameter in parameters.items():
                old_value = graph.globals.get(name)
                new_value = parameter.detach().cpu().numpy()
                if np.max(np.abs(new_value - old_value)) > 1e-10:
                    changed += 1
                graph.globals.set(name, new_value)
        return RefinementResult(
            True,
            self.steps,
            initial_loss,
            final_loss,
            changed,
            trainable,
            "torch_l_bfgs",
            self.requested_backend,
        )


def selected_output_global_refs(graph: Graph, config: dict[str, str]) -> set[str]:
    refs: set[str] = set()
    for dependencies in graph.output_dependency_map(config).values():
        for dependency in dependencies:
            node_name, alternative_id = dependency.split(".", maxsplit=1)
            node = graph.nodes[node_name]
            for alternative in node.alternatives:
                if alternative.id == alternative_id:
                    refs.update(alternative.global_refs)
                    break
    return refs
