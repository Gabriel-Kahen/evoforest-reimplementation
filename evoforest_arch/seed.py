from __future__ import annotations

import numpy as np

from evoforest_arch.graph import Graph
from evoforest_arch.primitives import PrimitiveRegistry
from evoforest_arch.task import TaskSchema


def build_seed_graph(task_schema: TaskSchema | None = None) -> Graph:
    schema = task_schema or TaskSchema.tabular()
    if schema.kind == "time_series_boundary":
        return build_structural_break_seed_graph(schema)
    return build_tabular_seed_graph(schema)


def build_tabular_seed_graph(task_schema: TaskSchema | None = None) -> Graph:
    schema = task_schema or TaskSchema.tabular()
    if schema.kind != "tabular":
        raise ValueError(f"Tabular seed graph cannot be built for task kind {schema.kind!r}.")
    registry = PrimitiveRegistry.for_task(schema)
    graph = Graph(name="evoforest_tabular_seed", task_schema=schema.to_dict())
    default_input = schema.default_input
    graph.add_input(default_input, schema.input(default_input).description)

    graph.globals.add("gate_scale", [1.0], trainable=True, description="Shared scale for callable gate families.")
    graph.globals.add(
        "projection_vector",
        np.linspace(0.2, 1.0, 16),
        trainable=True,
        description="Low-dimensional global vector used by projection outputs.",
    )
    graph.globals.add(
        "residual_huber_scale",
        [1.0],
        trainable=False,
        description="Shared threshold for residual-based Ridge reweighting.",
    )

    graph.add_node("base_features", "intermediate", "Direct generic task-input transforms.")
    graph.add_node("nonlinear_features", "intermediate", "Generic nonlinear feature transforms.")
    graph.add_node("summary_features", "intermediate", "Row-level task-input summaries.")
    graph.add_node("activation", "callable", "Reusable callable family node.")
    graph.add_node("output", "output", "Candidate predictive feature alternatives evaluated as an ensemble.")
    graph.add_node("ridge_w", "fitting", "Sample-weight alternatives for the downstream Ridge readout.")
    graph.add_node("ridge_g", "fitting", "Residual-weighting alternatives for iterative Ridge fitting.")

    graph.nodes["base_features"].add_alternative(registry.build("tabular_raw", "raw", (default_input,)))
    graph.nodes["base_features"].add_alternative(registry.build("tabular_centered", "centered", (default_input,)))
    graph.nodes["nonlinear_features"].add_alternative(registry.build("tabular_abs", "abs", (default_input,)))
    graph.nodes["nonlinear_features"].add_alternative(registry.build("tabular_square", "square", (default_input,)))
    graph.nodes["nonlinear_features"].add_alternative(registry.build("tabular_low_rank_interactions", "interactions", (default_input,)))
    graph.nodes["summary_features"].add_alternative(registry.build("tabular_summary", "summary", (default_input,)))
    graph.nodes["activation"].add_alternative(registry.build("identity_callable", "identity", ()))
    graph.nodes["activation"].add_alternative(registry.build("sigmoid_gate_callable", "sigmoid_gate", ()))
    graph.nodes["activation"].add_alternative(registry.build("clipped_linear_callable", "clipped_linear", ()))
    parents = ("base_features", "nonlinear_features", "summary_features")
    graph.nodes["output"].add_alternative(registry.build("pass_outputs", "raw_concat", parents))
    graph.nodes["output"].add_alternative(registry.build("activated_outputs", "activated", (*parents, "activation")))
    graph.nodes["output"].add_alternative(registry.build("projection_outputs", "projection", parents))
    graph.nodes["ridge_w"].add_alternative(registry.build("uniform_sample_weight", "uniform", ()))
    graph.nodes["ridge_g"].add_alternative(registry.build("identity_residual_weight", "identity", ()))
    graph.nodes["ridge_g"].add_alternative(registry.build("huber_residual_weight", "huber", ()))
    return graph


def build_structural_break_seed_graph(task_schema: TaskSchema | None = None) -> Graph:
    schema = task_schema or TaskSchema.structural_break()
    if schema.kind != "time_series_boundary":
        raise ValueError(f"Structural-break seed graph cannot be built for task kind {schema.kind!r}.")
    registry = PrimitiveRegistry.for_task(schema)
    graph = Graph(name="evoforest_structural_break_seed", task_schema=schema.to_dict())
    for input_spec in schema.inputs:
        graph.add_input(input_spec.name, input_spec.description)

    graph.globals.add("gate_scale", [1.0], trainable=True, description="Shared scale for callable gate families.")
    graph.globals.add(
        "projection_vector",
        np.linspace(0.2, 1.0, 16),
        trainable=True,
        description="Low-dimensional global vector used by projection outputs.",
    )
    graph.globals.add(
        "residual_huber_scale",
        [1.0],
        trainable=False,
        description="Shared threshold for residual-based Ridge reweighting.",
    )

    series_input = schema.default_input
    graph.add_node("segment_stats", "intermediate", "Segment-level change statistics.")
    graph.add_node("trend_stats", "intermediate", "Trend summaries around the task boundary.")
    graph.add_node("shape_stats", "intermediate", "Profile and frequency shape statistics.")
    graph.add_node("activation", "callable", "Reusable callable family node.")
    graph.add_node("output", "output", "Candidate predictive feature alternatives evaluated as an ensemble.")
    graph.add_node("ridge_w", "fitting", "Sample-weight alternatives for the downstream Ridge readout.")
    graph.add_node("ridge_g", "fitting", "Residual-weighting alternatives for iterative Ridge fitting.")

    graph.nodes["segment_stats"].add_alternative(registry.build("segment_basic", "basic", (series_input,)))
    graph.nodes["segment_stats"].add_alternative(registry.build("segment_robust", "robust", (series_input,)))
    graph.nodes["trend_stats"].add_alternative(registry.build("trend_basic", "linear", (series_input,)))
    graph.nodes["shape_stats"].add_alternative(registry.build("cusum_basic", "cusum", (series_input,)))
    graph.nodes["shape_stats"].add_alternative(registry.build("spectral_basic", "spectral", (series_input,)))
    graph.nodes["activation"].add_alternative(registry.build("identity_callable", "identity", ()))
    graph.nodes["activation"].add_alternative(registry.build("sigmoid_gate_callable", "sigmoid_gate", ()))
    graph.nodes["activation"].add_alternative(registry.build("clipped_linear_callable", "clipped_linear", ()))
    parents = ("segment_stats", "trend_stats", "shape_stats")
    graph.nodes["output"].add_alternative(registry.build("pass_outputs", "raw_concat", parents))
    graph.nodes["output"].add_alternative(registry.build("activated_outputs", "activated", (*parents, "activation")))
    graph.nodes["output"].add_alternative(registry.build("projection_outputs", "projection", parents))
    graph.nodes["ridge_w"].add_alternative(registry.build("uniform_sample_weight", "uniform", ()))
    graph.nodes["ridge_w"].add_alternative(registry.build("boundary_energy_weight", "boundary_energy", (series_input,)))
    graph.nodes["ridge_g"].add_alternative(registry.build("identity_residual_weight", "identity", ()))
    graph.nodes["ridge_g"].add_alternative(registry.build("huber_residual_weight", "huber", ()))
    return graph
