"""Research benchmarks for evaluating EvoForest beyond architecture conformance."""

from .compositional_dags import (
    BenchmarkDataset,
    DatasetSplit,
    MotifSpec,
    NodeSpec,
    TaskSpec,
    generate_benchmark,
    task_catalog,
)
from .metrics import area_under_learning_curve, nrmse, r2_score, rmse
from .baselines import BaselineEvaluation, RandomFeatureRidge, RawRidge
from .evoforest_model import FrozenEvoForestRegressor, fit_frozen_evoforest_regressor
from .external_datasets import ExternalDatasetManifest, FrozenRegressionDataset, load_manifest, load_regression_dataset
from .module_transfer import TransferReport, transfer_alternatives
from .protocol import (
    BudgetAccountant,
    BudgetExceeded,
    BudgetLimits,
    BudgetUsage,
    DatasetPartition,
    EvaluationProtocol,
    ExperimentResultRow,
    ProtocolViolation,
    TestEvaluation,
)

__all__ = [
    "BenchmarkDataset",
    "DatasetSplit",
    "MotifSpec",
    "NodeSpec",
    "TaskSpec",
    "generate_benchmark",
    "task_catalog",
    "area_under_learning_curve",
    "nrmse",
    "r2_score",
    "rmse",
    "BaselineEvaluation",
    "RandomFeatureRidge",
    "RawRidge",
    "FrozenEvoForestRegressor",
    "fit_frozen_evoforest_regressor",
    "ExternalDatasetManifest",
    "FrozenRegressionDataset",
    "load_manifest",
    "load_regression_dataset",
    "TransferReport",
    "transfer_alternatives",
    "BudgetAccountant",
    "BudgetExceeded",
    "BudgetLimits",
    "BudgetUsage",
    "DatasetPartition",
    "EvaluationProtocol",
    "ExperimentResultRow",
    "ProtocolViolation",
    "TestEvaluation",
]
