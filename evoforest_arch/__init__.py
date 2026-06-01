"""Clean-room EvoForest architecture package."""

from evoforest_arch.agents import EngineerAgent, Hypothesis, ScientistAgent
from evoforest_arch.evaluator import EvaluationResult, RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.graph import EvalContext, FeatureBlock, Graph, GraphNode, NodeAlternative, ResidualWeightRule
from evoforest_arch.llm import (
    DEFAULT_ISLAND_TEMPERATURES,
    HTTPJSONLLMClient,
    LLMEngineerAgent,
    LLMScientistAgent,
    PromptBuilder,
    PromptRecord,
    StaticLLMClient,
)
from evoforest_arch.maintenance import GraphMaintenance, MaintenanceReport
from evoforest_arch.mutations import GlobalSpec, MutationDocument, MutationEngine, MutationSpec, NodeSpec, RemoveSpec
from evoforest_arch.refinement import GlobalRefiner, RefinementResult, TorchLBFGSRefiner
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.source import build_source_alternative, compile_lambda_source
from evoforest_arch.synthetic import make_structural_break_data
from evoforest_arch.task_context import TaskContextSummary, TensorSummary, build_task_context

__all__ = [
    "EvalContext",
    "EvaluationResult",
    "EngineerAgent",
    "EvolutionLoop",
    "FeatureBlock",
    "Graph",
    "GraphNode",
    "GlobalSpec",
    "GlobalRefiner",
    "GraphMaintenance",
    "DEFAULT_ISLAND_TEMPERATURES",
    "HTTPJSONLLMClient",
    "Hypothesis",
    "LLMEngineerAgent",
    "LLMScientistAgent",
    "MaintenanceReport",
    "MutationDocument",
    "MutationEngine",
    "MutationSpec",
    "NodeAlternative",
    "NodeSpec",
    "PromptBuilder",
    "PromptRecord",
    "RefinementResult",
    "RemoveSpec",
    "ResidualWeightRule",
    "RidgeEvaluator",
    "ScientistAgent",
    "StaticLLMClient",
    "TaskContextSummary",
    "TensorSummary",
    "TorchLBFGSRefiner",
    "build_task_context",
    "build_seed_graph",
    "build_source_alternative",
    "compile_lambda_source",
    "make_structural_break_data",
]
