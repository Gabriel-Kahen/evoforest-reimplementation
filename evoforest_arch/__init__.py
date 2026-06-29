"""Clean-room EvoForest architecture package."""

from evoforest_arch.agents import EngineerAgent, Hypothesis, ScientistAgent
from evoforest_arch.evaluator import EvaluationResult, RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.graph import EvalContext, FeatureBlock, Graph, GraphNode, NodeAlternative, ResidualWeightRule
from evoforest_arch.llm import (
    ClaudeLLMClient,
    DEFAULT_ISLAND_TEMPERATURES,
    GeminiLLMClient,
    LLMEngineerAgent,
    LLMMemorandumAgent,
    LLMScientistAgent,
    OpenAILLMClient,
    PromptBuilder,
    PromptRecord,
    SUPPORTED_LLM_PROVIDERS,
    StaticLLMClient,
    llm_client_from_env,
    llm_provider_from_env,
    load_env_file,
)
from evoforest_arch.maintenance import GraphMaintenance, MaintenanceReport
from evoforest_arch.mutations import GlobalSpec, MutationDocument, MutationEngine, MutationSpec, NodeSpec, RemoveSpec
from evoforest_arch.production import ProductionConfig, ProductionEvolutionRunner, export_best_graph, inspect_run, recheck_run
from evoforest_arch.refinement import GlobalRefiner, RefinementResult, TorchLBFGSRefiner
from evoforest_arch.seed import build_seed_graph, build_structural_break_seed_graph, build_tabular_seed_graph
from evoforest_arch.source import SourceExecutionError, SourceSandboxPolicy, SourceTimeoutError, build_source_alternative, compile_lambda_source
from evoforest_arch.synthetic import make_structural_break_data, make_tabular_data
from evoforest_arch.task import InputSpec, TaskSchema, task_schema_for_dataset
from evoforest_arch.task_context import TaskContextSummary, TensorSummary, build_task_context

__all__ = [
    "ClaudeLLMClient",
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
    "GeminiLLMClient",
    "Hypothesis",
    "InputSpec",
    "LLMEngineerAgent",
    "LLMMemorandumAgent",
    "LLMScientistAgent",
    "MaintenanceReport",
    "MutationDocument",
    "MutationEngine",
    "MutationSpec",
    "NodeAlternative",
    "NodeSpec",
    "OpenAILLMClient",
    "PromptBuilder",
    "PromptRecord",
    "ProductionConfig",
    "ProductionEvolutionRunner",
    "RefinementResult",
    "RemoveSpec",
    "ResidualWeightRule",
    "RidgeEvaluator",
    "ScientistAgent",
    "SourceExecutionError",
    "SourceSandboxPolicy",
    "SourceTimeoutError",
    "SUPPORTED_LLM_PROVIDERS",
    "StaticLLMClient",
    "TaskContextSummary",
    "TaskSchema",
    "TensorSummary",
    "TorchLBFGSRefiner",
    "build_task_context",
    "build_seed_graph",
    "build_source_alternative",
    "build_structural_break_seed_graph",
    "compile_lambda_source",
    "export_best_graph",
    "inspect_run",
    "llm_client_from_env",
    "llm_provider_from_env",
    "load_env_file",
    "make_structural_break_data",
    "make_tabular_data",
    "recheck_run",
    "build_tabular_seed_graph",
    "task_schema_for_dataset",
]
