from __future__ import annotations

import numpy as np

from evoforest_arch.graph import Graph
from evoforest_arch.primitives import PrimitiveRegistry


def build_seed_graph() -> Graph:
    registry = PrimitiveRegistry.default()
    graph = Graph(name="evoforest_seed")
    graph.add_input("series", "Batch of univariate time series with shape (n_series, length).")
    graph.add_input("boundary", "Known boundary index separating pre and post segments.")

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
        trainable=True,
        description="Shared threshold for residual-based Ridge reweighting.",
    )

    graph.add_node("segment_stats", "intermediate", "Segment-level change statistics.")
    graph.add_node("trend_stats", "intermediate", "Trend summaries around the boundary.")
    graph.add_node("shape_stats", "intermediate", "Profile and frequency shape statistics.")
    graph.add_node("activation", "callable", "Reusable callable family node.")
    graph.add_node("output", "output", "Candidate predictive feature alternatives evaluated as an ensemble.")
    graph.add_node("ridge_w", "fitting", "Sample-weight alternatives for the downstream Ridge readout.")
    graph.add_node("ridge_g", "fitting", "Residual-weighting alternatives for iterative Ridge fitting.")

    graph.nodes["segment_stats"].add_alternative(registry.build("segment_basic", "basic", ("series",)))
    graph.nodes["segment_stats"].add_alternative(registry.build("segment_robust", "robust", ("series",)))
    graph.nodes["trend_stats"].add_alternative(registry.build("trend_basic", "linear", ("series",)))
    graph.nodes["shape_stats"].add_alternative(registry.build("cusum_basic", "cusum", ("series",)))
    graph.nodes["shape_stats"].add_alternative(registry.build("spectral_basic", "spectral", ("series",)))
    graph.nodes["activation"].add_alternative(registry.build("identity_callable", "identity", ()))
    graph.nodes["activation"].add_alternative(registry.build("sigmoid_gate_callable", "sigmoid_gate", ()))
    graph.nodes["activation"].add_alternative(registry.build("clipped_linear_callable", "clipped_linear", ()))
    parents = ("segment_stats", "trend_stats", "shape_stats")
    graph.nodes["output"].add_alternative(registry.build("pass_outputs", "raw_concat", parents))
    graph.nodes["output"].add_alternative(registry.build("activated_outputs", "activated", (*parents, "activation")))
    graph.nodes["output"].add_alternative(registry.build("projection_outputs", "projection", parents))
    graph.nodes["ridge_w"].add_alternative(registry.build("uniform_sample_weight", "uniform", ()))
    graph.nodes["ridge_w"].add_alternative(registry.build("boundary_energy_weight", "boundary_energy", ("series",)))
    graph.nodes["ridge_g"].add_alternative(registry.build("identity_residual_weight", "identity", ()))
    graph.nodes["ridge_g"].add_alternative(registry.build("huber_residual_weight", "huber", ()))
    return graph
