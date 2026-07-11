"""Deterministic, ground-truth DAG tasks for computation-discovery research.

The test sets deliberately separate ordinary interpolation from support
extrapolation.  Targets are evaluated on latent inputs before optional missingness
is applied, so a benchmark can test robustness to corrupted observations without
changing its underlying scientific relationship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class NodeSpec:
    """One named computation in a ground-truth directed acyclic graph."""

    name: str
    operation: str
    inputs: tuple[str, ...]
    parameters: Mapping[str, float | tuple[float, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class MotifSpec:
    """A meaningful subgraph whose recovery can be scored independently."""

    name: str
    nodes: tuple[str, ...]
    description: str
    reuse_count: int = 1


@dataclass(frozen=True)
class TaskSpec:
    """Complete ground truth and sampling controls for one benchmark task."""

    name: str
    n_features: int
    active_variables: tuple[int, ...]
    nodes: tuple[NodeSpec, ...]
    output_node: str
    motifs: tuple[MotifSpec, ...]
    description: str
    noise_std: float = 0.05
    heteroscedastic_node: str | None = None
    heteroscedastic_strength: float = 0.0
    correlated_distractors: Mapping[int, tuple[int, float]] = field(default_factory=dict)
    missing_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError("n_features must be positive")
        if not set(self.active_variables) <= set(range(self.n_features)):
            raise ValueError("active_variables contains an invalid feature index")
        if not 0.0 <= self.missing_rate < 1.0:
            raise ValueError("missing_rate must lie in [0, 1)")
        available = {f"x{i}" for i in range(self.n_features)}
        for node in self.nodes:
            if node.name in available:
                raise ValueError(f"duplicate node name: {node.name}")
            unknown = set(node.inputs) - available
            if unknown:
                raise ValueError(f"node {node.name} has forward/unknown inputs: {sorted(unknown)}")
            available.add(node.name)
        if self.output_node not in available:
            raise ValueError("output_node is not present in the graph")
        node_names = {node.name for node in self.nodes}
        for motif in self.motifs:
            if not set(motif.nodes) <= node_names:
                raise ValueError(f"motif {motif.name} references unknown nodes")
        for distractor, (source, correlation) in self.correlated_distractors.items():
            if distractor in self.active_variables or not 0 <= distractor < self.n_features:
                raise ValueError("correlated distractors must be valid inactive variables")
            if source not in self.active_variables or not 0.0 <= correlation < 1.0:
                raise ValueError("invalid correlated distractor source or correlation")

    def metadata(self) -> dict[str, object]:
        """Return JSON-friendly ground-truth metadata for result artifacts."""

        return {
            "name": self.name,
            "description": self.description,
            "n_features": self.n_features,
            "active_variables": list(self.active_variables),
            "inactive_variables": sorted(set(range(self.n_features)) - set(self.active_variables)),
            "output_node": self.output_node,
            "nodes": [
                {
                    "name": node.name,
                    "operation": node.operation,
                    "inputs": list(node.inputs),
                    "parameters": dict(node.parameters),
                }
                for node in self.nodes
            ],
            "motifs": [
                {
                    "name": motif.name,
                    "nodes": list(motif.nodes),
                    "description": motif.description,
                    "reuse_count": motif.reuse_count,
                }
                for motif in self.motifs
            ],
            "noise_std": self.noise_std,
            "heteroscedastic_node": self.heteroscedastic_node,
            "heteroscedastic_strength": self.heteroscedastic_strength,
            "correlated_distractors": {
                str(key): {"source": value[0], "correlation": value[1]}
                for key, value in self.correlated_distractors.items()
            },
            "missing_rate": self.missing_rate,
        }


@dataclass(frozen=True)
class DatasetSplit:
    """Observed samples plus latent values needed for controlled analysis."""

    X: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    latent_X: np.ndarray
    missing_mask: np.ndarray
    node_values: Mapping[str, np.ndarray]
    regime: str


@dataclass(frozen=True)
class BenchmarkDataset:
    spec: TaskSpec
    seed: int
    feature_names: tuple[str, ...]
    train: DatasetSplit
    validation: DatasetSplit
    test_interpolation: DatasetSplit
    test_extrapolation: DatasetSplit

    def metadata(self) -> dict[str, object]:
        return {
            "task": self.spec.metadata(),
            "seed": self.seed,
            "feature_names": list(self.feature_names),
            "splits": {
                "train": len(self.train.y),
                "validation": len(self.validation.y),
                "test_interpolation": len(self.test_interpolation.y),
                "test_extrapolation": len(self.test_extrapolation.y),
            },
            "extrapolation_support": "each active variable has |x| in [1.25, 2.25]",
            "interpolation_support": "all base variables lie in [-1, 1] before distractor construction",
        }


def task_catalog() -> dict[str, TaskSpec]:
    """Return benchmark tasks spanning reuse, gates, ratios, and corruption."""

    tasks = (
        TaskSpec(
            name="shared_wave_gate",
            n_features=8,
            active_variables=(0, 1, 2, 3, 4),
            nodes=(
                NodeSpec("interaction", "multiply", ("x0", "x1")),
                NodeSpec("shared_wave", "sin", ("interaction",), {"scale": 1.7}),
                NodeSpec("log_magnitude", "log1p_abs", ("x2",)),
                NodeSpec("positive_gate", "indicator_gt", ("x3",), {"threshold": 0.0}),
                NodeSpec("gated_log", "multiply", ("positive_gate", "log_magnitude")),
                NodeSpec("wave_modulation", "multiply", ("shared_wave", "x4")),
                NodeSpec(
                    "target",
                    "weighted_sum",
                    ("shared_wave", "gated_log", "wave_modulation"),
                    {"weights": (1.0, 1.0, 0.2)},
                ),
            ),
            output_node="target",
            motifs=(
                MotifSpec(
                    "shared_oscillation",
                    ("interaction", "shared_wave"),
                    "sinusoid of a two-variable interaction, reused in two target terms",
                    reuse_count=2,
                ),
                MotifSpec(
                    "conditional_log",
                    ("log_magnitude", "positive_gate", "gated_log"),
                    "threshold-gated logarithmic magnitude",
                ),
            ),
            description="Shared nonlinear interaction combined with a conditional branch.",
            correlated_distractors={5: (0, 0.9), 6: (2, 0.75)},
        ),
        TaskSpec(
            name="piecewise_rational",
            n_features=7,
            active_variables=(0, 1, 2, 3),
            nodes=(
                NodeSpec("ratio", "safe_ratio", ("x0", "x1"), {"offset": 1.0}),
                NodeSpec("branch", "indicator_gt", ("x2",), {"threshold": -0.15}),
                NodeSpec("curvature", "square", ("x3",)),
                NodeSpec("selected_curve", "multiply", ("branch", "curvature")),
                NodeSpec(
                    "target", "weighted_sum", ("ratio", "selected_curve", "branch"), {"weights": (1.0, 0.7, -0.25)}
                ),
            ),
            output_node="target",
            motifs=(
                MotifSpec("stable_ratio", ("ratio",), "bounded-denominator rational interaction"),
                MotifSpec(
                    "piecewise_curvature",
                    ("branch", "curvature", "selected_curve"),
                    "quadratic contribution active on only one side of a threshold",
                    reuse_count=2,
                ),
            ),
            description="A rational term plus a discontinuous piecewise quadratic branch.",
            noise_std=0.08,
            correlated_distractors={4: (1, 0.85)},
        ),
        TaskSpec(
            name="heteroscedastic_reuse",
            n_features=9,
            active_variables=(0, 1, 2, 3),
            nodes=(
                NodeSpec("radial", "sum_squares", ("x0", "x1")),
                NodeSpec("compressed", "log1p_abs", ("radial",)),
                NodeSpec("oscillation", "sin", ("x2",), {"scale": 2.2}),
                NodeSpec("coupling", "multiply", ("compressed", "oscillation")),
                NodeSpec(
                    "target", "weighted_sum", ("compressed", "coupling", "x3"), {"weights": (0.8, 1.0, 0.3)}
                ),
            ),
            output_node="target",
            motifs=(
                MotifSpec(
                    "radial_compression",
                    ("radial", "compressed"),
                    "log-compressed radial energy reused directly and in an interaction",
                    reuse_count=2,
                ),
                MotifSpec("modulated_energy", ("oscillation", "coupling"), "oscillatory modulation of radial energy"),
            ),
            description="Repeated radial motif under input-dependent observation noise.",
            noise_std=0.04,
            heteroscedastic_node="compressed",
            heteroscedastic_strength=0.8,
            correlated_distractors={4: (0, 0.95), 5: (2, 0.8)},
        ),
        TaskSpec(
            name="missing_sensor_composition",
            n_features=8,
            active_variables=(0, 1, 2, 3),
            nodes=(
                NodeSpec("contrast", "difference", ("x0", "x1")),
                NodeSpec("smooth_contrast", "tanh", ("contrast",), {"scale": 1.4}),
                NodeSpec("amplitude", "log1p_abs", ("x2",)),
                NodeSpec("modulated", "multiply", ("smooth_contrast", "amplitude")),
                NodeSpec("target", "weighted_sum", ("smooth_contrast", "modulated", "x3"), {"weights": (0.6, 1.0, 0.2)}),
            ),
            output_node="target",
            motifs=(
                MotifSpec(
                    "bounded_contrast",
                    ("contrast", "smooth_contrast"),
                    "bounded contrast between two sensors, reused in two terms",
                    reuse_count=2,
                ),
                MotifSpec("amplitude_modulation", ("amplitude", "modulated"), "magnitude-dependent modulation"),
            ),
            description="A compositional sensor task with target-independent missing observations.",
            noise_std=0.06,
            correlated_distractors={4: (0, 0.8)},
            missing_rate=0.12,
        ),
    )
    return {task.name: task for task in tasks}


def generate_benchmark(
    task: str | TaskSpec,
    *,
    seed: int = 0,
    n_train: int = 512,
    n_validation: int = 256,
    n_test: int = 512,
) -> BenchmarkDataset:
    """Generate independent train, validation, interpolation, and extrapolation splits."""

    spec = task_catalog()[task] if isinstance(task, str) else task
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError("all split sizes must be positive")
    stable_name = int.from_bytes(hashlib.blake2b(spec.name.encode(), digest_size=4).digest(), "little")
    split_seeds = np.random.SeedSequence([seed, stable_name]).spawn(4)
    splits = [
        _make_split(spec, n_train, split_seeds[0], "interpolation"),
        _make_split(spec, n_validation, split_seeds[1], "interpolation"),
        _make_split(spec, n_test, split_seeds[2], "interpolation"),
        _make_split(spec, n_test, split_seeds[3], "extrapolation"),
    ]
    return BenchmarkDataset(
        spec=spec,
        seed=seed,
        feature_names=tuple(f"x{i}" for i in range(spec.n_features)),
        train=splits[0],
        validation=splits[1],
        test_interpolation=splits[2],
        test_extrapolation=splits[3],
    )


def evaluate_ground_truth(spec: TaskSpec, X: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate every ground-truth node, enabling functional motif scoring."""

    if X.ndim != 2 or X.shape[1] != spec.n_features:
        raise ValueError(f"expected X with shape (n, {spec.n_features})")
    values: dict[str, np.ndarray] = {f"x{i}": X[:, i] for i in range(spec.n_features)}
    for node in spec.nodes:
        inputs = [values[name] for name in node.inputs]
        values[node.name] = _apply_operation(node, inputs)
    return {node.name: values[node.name] for node in spec.nodes}


def _make_split(spec: TaskSpec, n: int, seed: np.random.SeedSequence, regime: str) -> DatasetSplit:
    rng = np.random.default_rng(seed)
    latent_X = _sample_inputs(spec, n, rng, extrapolate=regime == "extrapolation")
    node_values = evaluate_ground_truth(spec, latent_X)
    y_clean = node_values[spec.output_node]
    noise_scale = np.full(n, spec.noise_std)
    if spec.heteroscedastic_node is not None:
        driver = np.abs(node_values[spec.heteroscedastic_node])
        noise_scale *= 1.0 + spec.heteroscedastic_strength * driver
    y = y_clean + rng.normal(0.0, noise_scale, size=n)
    missing_mask = rng.random(latent_X.shape) < spec.missing_rate
    X = latent_X.copy()
    X[missing_mask] = np.nan
    return DatasetSplit(
        X=X,
        y=y,
        y_clean=y_clean,
        latent_X=latent_X,
        missing_mask=missing_mask,
        node_values=node_values,
        regime=regime,
    )


def _sample_inputs(spec: TaskSpec, n: int, rng: np.random.Generator, *, extrapolate: bool) -> np.ndarray:
    X = rng.uniform(-1.0, 1.0, size=(n, spec.n_features))
    if extrapolate:
        active = np.asarray(spec.active_variables)
        magnitudes = rng.uniform(1.25, 2.25, size=(n, len(active)))
        signs = rng.choice((-1.0, 1.0), size=(n, len(active)))
        X[:, active] = magnitudes * signs
    for distractor, (source, correlation) in spec.correlated_distractors.items():
        source_values = X[:, source]
        scale = max(float(np.std(source_values)), 1e-12)
        noise = rng.normal(0.0, scale, size=n)
        X[:, distractor] = correlation * source_values + np.sqrt(1.0 - correlation**2) * noise
    return X


def _apply_operation(node: NodeSpec, inputs: Sequence[np.ndarray]) -> np.ndarray:
    operation = node.operation
    if operation == "multiply":
        return inputs[0] * inputs[1]
    if operation == "difference":
        return inputs[0] - inputs[1]
    if operation == "sin":
        return np.sin(float(node.parameters.get("scale", 1.0)) * inputs[0])
    if operation == "tanh":
        return np.tanh(float(node.parameters.get("scale", 1.0)) * inputs[0])
    if operation == "square":
        return np.square(inputs[0])
    if operation == "sum_squares":
        return sum(np.square(value) for value in inputs)
    if operation == "log1p_abs":
        return np.log1p(np.abs(inputs[0]))
    if operation == "indicator_gt":
        return (inputs[0] > float(node.parameters.get("threshold", 0.0))).astype(float)
    if operation == "safe_ratio":
        offset = float(node.parameters.get("offset", 1.0))
        return inputs[0] / (offset + np.abs(inputs[1]))
    if operation == "weighted_sum":
        weights = tuple(node.parameters.get("weights", (1.0,) * len(inputs)))
        if len(weights) != len(inputs):
            raise ValueError(f"node {node.name} has the wrong number of weights")
        return sum(float(weight) * value for weight, value in zip(weights, inputs, strict=True))
    raise ValueError(f"unsupported ground-truth operation: {operation}")
