from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class StudySpec:
    name: str
    version: int
    status: str
    task_families: tuple[str, ...]
    data_seeds: tuple[int, ...]
    search_seeds: tuple[int, ...]
    n_train: int
    n_validation: int
    n_test: int
    evolution_steps: int
    max_configurations: int
    screening_finalists: int
    methods: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    exclusion_rules: tuple[str, ...]
    hypotheses: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"pilot", "confirmatory_locked"}:
            raise ValueError("Study status must be pilot or confirmatory_locked.")
        if min(self.n_train, self.n_validation, self.n_test, self.evolution_steps, self.max_configurations) <= 0:
            raise ValueError("Study sizes and budgets must be positive.")
        if self.screening_finalists > self.max_configurations:
            raise ValueError("screening_finalists cannot exceed max_configurations.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pilot_spec() -> StudySpec:
    return StudySpec(
        name="evoforest-generalization-pilot",
        version=2,
        status="pilot",
        task_families=("shared_wave_gate", "piecewise_rational", "heteroscedastic_reuse"),
        data_seeds=(101,),
        search_seeds=(201, 202, 203),
        n_train=128,
        n_validation=64,
        n_test=128,
        evolution_steps=4,
        max_configurations=8,
        screening_finalists=4,
        methods=("raw_ridge", "random_features_ridge", "evoforest_seed", "evoforest_paper_llm"),
        primary_metrics=("interpolation_nrmse", "extrapolation_nrmse", "validation_aulc_nrmse"),
        secondary_metrics=("r2", "wall_time_seconds", "exact_evaluations", "failure_rate"),
        exclusion_rules=(
            "exclude only predeclared loader or numerical failures",
            "never exclude a completed task based on method rank",
            "keep failed runs in failure-rate denominators",
        ),
        hypotheses=(
            "paper-style LLM-guided EvoForest improves validation AULC over the fixed seed graph",
            "EvoForest gains are larger on reusable or gated compositions than on ordinary interpolation",
            "related module transfer improves early target-task search without increasing negative transfer",
        ),
    )


def confirmatory_spec() -> StudySpec:
    return StudySpec(
        name="evoforest-generalization-confirmatory",
        version=2,
        status="confirmatory_locked",
        task_families=(
            "shared_wave_gate",
            "piecewise_rational",
            "heteroscedastic_reuse",
            "missing_sensor_composition",
        ),
        data_seeds=tuple(range(1001, 1009)),
        search_seeds=tuple(range(2001, 2011)),
        n_train=512,
        n_validation=256,
        n_test=512,
        evolution_steps=24,
        max_configurations=64,
        screening_finalists=16,
        methods=(
            "raw_ridge",
            "random_features_ridge",
            "hist_gradient_boosting",
            "extra_trees",
            "feat",
            "pysr",
            "autofeat",
            "llm_one_shot",
            "llm_scalar_iterative",
            "evoforest_paper_llm",
        ),
        primary_metrics=("interpolation_nrmse", "extrapolation_nrmse", "validation_aulc_nrmse"),
        secondary_metrics=(
            "r2",
            "active_variable_f1",
            "motif_recall",
            "wall_time_seconds",
            "exact_evaluations",
            "screening_evaluations",
            "llm_tokens",
            "failure_rate",
        ),
        exclusion_rules=(
            "task and seed inclusion is fixed before confirmatory results are inspected",
            "exclude only corrupt data, protocol violations, or predeclared numerical failure",
            "failed and budget-exhausted runs remain in reliability summaries",
            "no method-specific task removal",
            "test evaluation occurs once after graph and readout selection",
        ),
        hypotheses=(
            "paper-style LLM-guided EvoForest has lower validation AULC NRMSE than fixed and random feature baselines",
            "EvoForest gains increase with shared motifs, gates, and distractor variables",
            "EvoForest improves extrapolation relative to random features more than ordinary interpolation",
            "related module transfer improves early AULC relative to scratch and unrelated transfer",
        ),
    )


def write_spec(path: Path, spec: StudySpec) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": spec.fingerprint(), "spec": spec.to_dict()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
