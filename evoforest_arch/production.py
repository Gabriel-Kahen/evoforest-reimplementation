from __future__ import annotations

import concurrent.futures
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
import pathlib
import subprocess
from typing import Any

import numpy as np

from evoforest_arch.agents import EngineerAgent, ScientistAgent
from evoforest_arch.evaluator import EvaluationResult, RidgeEvaluator
from evoforest_arch.evolution import CandidateOutcome, EvolutionLoop
from evoforest_arch.feedback import feedback_summary, toon_report
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import graph_hash, graph_from_dict, graph_from_path, write_graph
from evoforest_arch.llm import DEFAULT_ISLAND_TEMPERATURES, LLMEngineerAgent, LLMMemorandumAgent, LLMScientistAgent
from evoforest_arch.mutations import MutationDocument, MutationEngine
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.splits import SplitManifest, make_split_manifest, read_split_manifest, split_dataset, write_split_manifest
from evoforest_arch.synthetic import make_structural_break_data
from evoforest_arch.task_context import build_task_context


SAFE_STAGED_EXECUTION_RULES = (
    "Create a fresh run directory; production evolve never overwrites a non-empty directory.",
    "Persist dataset fingerprint and train/validation/test split indices before the first mutation.",
    "Evolve only on the train split; promote archive entries only after an independent validation recheck.",
    "Keep the test split untouched during evolve; consume it only through an explicit recheck command.",
    "Resume only from serialized run state, graph artifacts, and the original split manifest.",
    "For async island runs, persist every island's state, graph, checkpoint, and memorandum before resuming work.",
    "Run production async islands as one OS process actor per dedicated device.",
    "Record global-best migrations separately and write the target island state immediately.",
    "Keep source-backed mutations disabled unless a trusted sandbox policy is supplied outside this package.",
)

PAPER_ISLAND_COUNT = 4
PAPER_GPU_DEVICES = tuple(f"cuda:{index}" for index in range(PAPER_ISLAND_COUNT))
PRODUCTION_PROFILE = "production"
PAPER_PROFILE = "paper"
SUPPORTED_PRODUCTION_PROFILES = (PRODUCTION_PROFILE, PAPER_PROFILE)
VALIDATION_PROMOTION_POLICY = "validation_gated"
CV_AUC_PROMOTION_POLICY = "cv_auc"
SUPPORTED_PROMOTION_POLICIES = (VALIDATION_PROMOTION_POLICY, CV_AUC_PROMOTION_POLICY)
PAPER_PROFILE_STEPS = 600


@dataclass(frozen=True)
class ProductionConfig:
    output_dir: pathlib.Path
    profile: str = PRODUCTION_PROFILE
    steps: int = 4
    seed: int = 17
    dataset_name: str = "synthetic-structural-break"
    n_series: int = 240
    length: int = 160
    boundary: int | None = None
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    split_seed: int | None = None
    folds: int = 3
    max_configurations: int = 64
    irls_steps: int = 2
    refine_globals: bool = True
    refine_steps: int = 20
    refine_backend: str = "auto"
    promotion_policy: str = VALIDATION_PROMOTION_POLICY
    min_train_improvement: float = 1e-6
    min_validation_improvement: float = 1e-6
    allow_source_mutations: bool = False
    islands: int = PAPER_ISLAND_COUNT
    async_islands: bool = True
    island_workers: int | None = None
    island_devices: tuple[str, ...] | None = None
    migration_interval: int = 10
    torch_device: str | None = None

    def dataset_config(self) -> dict[str, Any]:
        return {
            "name": self.dataset_name,
            "seed": int(self.seed),
            "n_series": int(self.n_series),
            "length": int(self.length),
            "boundary": self.boundary,
        }

    def evaluator_config(self) -> dict[str, Any]:
        return {
            "n_splits": int(self.folds),
            "seed": int(self.seed),
            "max_configurations": int(self.max_configurations),
            "irls_steps": int(self.irls_steps),
            "refine_globals": bool(self.refine_globals),
            "refine_steps": int(self.refine_steps),
            "refine_backend": self.refine_backend,
            "group_key": None,
            "torch_device": self.torch_device,
        }

    @classmethod
    def paper_profile(cls, output_dir: pathlib.Path, **overrides: Any) -> "ProductionConfig":
        defaults = paper_profile_defaults()
        defaults.update(overrides)
        return cls(output_dir=output_dir, **defaults)


def paper_profile_defaults() -> dict[str, Any]:
    return {
        "profile": PAPER_PROFILE,
        "steps": PAPER_PROFILE_STEPS,
        "islands": PAPER_ISLAND_COUNT,
        "async_islands": True,
        "island_workers": PAPER_ISLAND_COUNT,
        "island_devices": PAPER_GPU_DEVICES,
        "max_configurations": 64,
        "refine_globals": True,
        "refine_backend": "torch",
        "promotion_policy": CV_AUC_PROMOTION_POLICY,
        "min_train_improvement": 0.0,
        "min_validation_improvement": 0.0,
    }


@dataclass
class RunState:
    run_id: str
    step: int
    archive_version: int
    best_train_auc: float
    best_validation_auc: float
    best_config: dict[str, str]
    current_graph_path: str
    best_graph_path: str
    rng_state: dict[str, Any]
    history: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    test_recheck_count: int = 0
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": int(self.step),
            "archive_version": int(self.archive_version),
            "best_train_auc": float(self.best_train_auc),
            "best_validation_auc": float(self.best_validation_auc),
            "best_config": dict(self.best_config),
            "current_graph_path": self.current_graph_path,
            "best_graph_path": self.best_graph_path,
            "rng_state": _json_ready(self.rng_state),
            "history": list(self.history[-40:]),
            "errors": list(self.errors[-40:]),
            "test_recheck_count": int(self.test_recheck_count),
            "generation": int(self.generation),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunState":
        return cls(
            run_id=str(payload["run_id"]),
            step=int(payload["step"]),
            archive_version=int(payload["archive_version"]),
            best_train_auc=float(payload["best_train_auc"]),
            best_validation_auc=float(payload["best_validation_auc"]),
            best_config={str(key): str(value) for key, value in dict(payload.get("best_config", {})).items()},
            current_graph_path=str(payload["current_graph_path"]),
            best_graph_path=str(payload["best_graph_path"]),
            rng_state=dict(payload["rng_state"]),
            history=[str(row) for row in payload.get("history", [])],
            errors=[str(row) for row in payload.get("errors", [])],
            test_recheck_count=int(payload.get("test_recheck_count", 0)),
            generation=int(payload.get("generation", 0)),
        )


@dataclass
class ProductionContext:
    run_dir: pathlib.Path
    manifest: dict[str, Any]
    split_manifest: SplitManifest
    splits: dict[str, tuple[dict[str, object], np.ndarray]]
    state: RunState
    current_graph: Graph
    best_graph: Graph
    best_train_result: EvaluationResult
    best_validation_result: EvaluationResult


@dataclass
class ProductionIslandContext:
    island: int
    run_dir: pathlib.Path
    state: RunState
    current_graph: Graph
    best_graph: Graph
    best_train_result: EvaluationResult
    best_validation_result: EvaluationResult
    device: str
    generation: int = 0


@dataclass
class IslandActorSnapshot:
    island: int
    device: str
    worker_id: str
    actor_pid: int
    state: RunState
    current_graph_payload: dict[str, Any]
    best_graph_payload: dict[str, Any]
    best_train_result: EvaluationResult
    best_validation_result: EvaluationResult


@dataclass
class IslandPreparedCandidate:
    island: int
    device: str
    worker_id: str
    actor_pid: int
    job_id: str
    proposed_step: int
    island_step: int
    generation: int
    base_graph_hash: str
    mutation_path: str
    snapshot: IslandActorSnapshot


@dataclass
class PendingIslandCandidate:
    island: ProductionIslandContext
    future: concurrent.futures.Future
    job_id: str
    proposed_step: int
    island_step: int
    generation: int
    base_graph_hash: str
    actor_pid: int | None = None
    prepared: IslandPreparedCandidate | None = None


@dataclass
class IslandCandidateResult:
    document: MutationDocument
    outcome: CandidateOutcome
    validation_result: EvaluationResult | None
    rng_state: dict[str, Any]
    errors: list[str]
    candidate_graph_payload: dict[str, Any] | None = None
    maintenance: dict[str, object] | None = None


@dataclass
class IslandActorCandidateResult:
    actor_pid: int
    candidate: IslandCandidateResult


@dataclass
class IslandCommitDecision:
    global_step: int
    accepted: bool
    global_best: bool
    root_best_train_auc: float
    root_best_validation_auc: float
    previous_global_train_auc: float
    previous_global_validation_auc: float


@dataclass
class IslandActorCommitResult:
    actor_pid: int
    snapshot: IslandActorSnapshot
    event: dict[str, Any]
    terminal_status: str
    terminal_error: str = ""


@dataclass
class IslandActorMigrationResult:
    actor_pid: int
    snapshot: IslandActorSnapshot
    event: dict[str, Any]


class ProductionIslandWorker:
    """Long-lived island owner for proposal, repair, evaluation, and prompt artifacts."""

    def __init__(
        self,
        *,
        island: int,
        device: str,
        evaluator_config: dict[str, Any],
        allow_source_mutations: bool,
        scientist: ScientistAgent | None,
        engineer: EngineerAgent | None,
        memorandum_agent: object | None,
        task_context: str,
        task_sources: tuple[tuple[str, str], ...],
        seed: int,
        train_inputs: dict[str, object] | None = None,
        train_y: np.ndarray | None = None,
        validation_inputs: dict[str, object] | None = None,
        validation_y: np.ndarray | None = None,
        run_dir: pathlib.Path | None = None,
        manifest: dict[str, Any] | None = None,
        split_manifest: SplitManifest | None = None,
        state: RunState | None = None,
        current_graph: Graph | None = None,
        best_graph: Graph | None = None,
        best_train_result: EvaluationResult | None = None,
        best_validation_result: EvaluationResult | None = None,
    ) -> None:
        self.island = int(island)
        self.device = str(device)
        self.worker_id = f"island_{self.island}"
        self.evaluator_config = {**evaluator_config, "torch_device": self.device}
        self.allow_source_mutations = bool(allow_source_mutations)
        self.scientist = _clone_agent_for_island(scientist)
        self.engineer = _clone_agent_for_island(engineer)
        self.memorandum_agent = _clone_agent_for_island(memorandum_agent)
        self.task_context = task_context
        self.task_sources = task_sources
        self.seed = int(seed) + 1009 * (self.island + 1)
        self.train_inputs = train_inputs
        self.train_y = train_y
        self.validation_inputs = validation_inputs
        self.validation_y = validation_y
        self.run_dir = run_dir
        self.manifest = manifest or {}
        self.split_manifest = split_manifest
        self.state = state
        self.current_graph = current_graph
        self.best_graph = best_graph
        self.best_train_result = best_train_result
        self.best_validation_result = best_validation_result
        self._prepared_jobs: dict[str, dict[str, Any]] = {}

    def run_candidate(
        self,
        *,
        base_graph: Graph,
        best_result: EvaluationResult,
        rng_state: dict[str, Any],
        island_step: int,
        output_dir: pathlib.Path,
        train_inputs: dict[str, object] | None = None,
        train_y: np.ndarray | None = None,
        validation_inputs: dict[str, object] | None = None,
        validation_y: np.ndarray | None = None,
        errors: list[str],
    ) -> IslandCandidateResult:
        train_inputs = self.train_inputs if train_inputs is None else train_inputs
        train_y = self.train_y if train_y is None else train_y
        validation_inputs = self.validation_inputs if validation_inputs is None else validation_inputs
        validation_y = self.validation_y if validation_y is None else validation_y
        if train_inputs is None or train_y is None or validation_inputs is None or validation_y is None:
            raise RuntimeError("Production island worker has not been initialized with train and validation data.")
        loop = self._build_loop(base_graph, rng_state)
        loop._install_task_context(self.task_context)
        memorandum = ProductionEvolutionRunner._read_memorandum(output_dir)
        try:
            document = loop._propose_document(
                base_graph,
                best_result,
                island_step,
                island=self.island,
                memorandum=memorandum,
                execution_errors="\n".join(errors[-8:]),
            )
        finally:
            loop._write_prompt_records(output_dir, island_step, island=self.island)
        loop._write_mutation_document(output_dir, island_step, document, island=self.island)
        outcome = loop._try_evaluate_candidate(base_graph, document, train_inputs, train_y)
        worker_errors = list(errors)
        if outcome.failed:
            document, outcome = loop._repair_candidate(
                base_graph,
                best_result,
                island_step,
                island=self.island,
                output_dir=output_dir,
                document=document,
                outcome=outcome,
                inputs=train_inputs,
                y=train_y,
                memorandum=memorandum,
                errors=worker_errors,
            )
        validation_result: EvaluationResult | None = None
        if not outcome.failed and outcome.candidate_graph is not None and outcome.result is not None:
            validation_result = loop.evaluator.evaluate(
                outcome.candidate_graph,
                validation_inputs,
                validation_y,
                config=outcome.result.config,
                update_graph=False,
            )
        candidate_graph_payload = outcome.candidate_graph.to_dict() if outcome.candidate_graph is not None else None
        maintenance = outcome.application.maintenance.to_dict() if outcome.application is not None else None
        process_safe_outcome = CandidateOutcome(
            application=None,
            candidate_graph=None,
            result=outcome.result,
            error=outcome.error,
        )
        return IslandCandidateResult(
            document=document,
            outcome=process_safe_outcome,
            validation_result=validation_result,
            rng_state=loop.rng.bit_generator.state,
            errors=worker_errors[-40:],
            candidate_graph_payload=candidate_graph_payload,
            maintenance=maintenance,
        )

    def write_memorandum(self, context: ProductionContext) -> None:
        loop = self._build_loop(context.current_graph, context.state.rng_state)
        loop._install_task_context(self.task_context)
        ProductionEvolutionRunner._write_memorandum_static(context, loop=loop, island=self.island)

    def snapshot(self) -> IslandActorSnapshot:
        state, current_graph, best_graph, best_train_result, best_validation_result = self._require_island_state()
        return IslandActorSnapshot(
            island=self.island,
            device=self.device,
            worker_id=self.worker_id,
            actor_pid=os.getpid(),
            state=RunState.from_dict(state.to_dict()),
            current_graph_payload=current_graph.to_dict(),
            best_graph_payload=best_graph.to_dict(),
            best_train_result=best_train_result,
            best_validation_result=best_validation_result,
        )

    def prepare_candidate(self, proposed_step: int) -> IslandPreparedCandidate:
        state, current_graph, _best_graph, best_train_result, _best_validation_result = self._require_island_state()
        if self.run_dir is None:
            raise RuntimeError("Production island worker has no run directory.")
        island_step = int(state.step) + 1
        generation = int(state.generation)
        base_graph = current_graph.clone()
        base_graph_hash = graph_hash(base_graph)
        job_id = f"g{int(proposed_step):04d}_i{self.island}_s{island_step:04d}_v{generation:04d}"
        mutation_path = f"islands/island_{self.island}/mutations/island_{self.island}_step_{island_step:04d}.yaml"
        self._prepared_jobs[job_id] = {
            "base_graph": base_graph,
            "best_result": best_train_result,
            "rng_state": copy.deepcopy(state.rng_state),
            "errors": list(state.errors),
            "island_step": island_step,
            "generation": generation,
            "base_graph_hash": base_graph_hash,
        }
        self._write_island_job_event(
            {
                "job_id": job_id,
                "status": "submitted",
                "global_step_hint": int(proposed_step),
                "island": self.island,
                "island_step": island_step,
                "base_generation": generation,
                "base_graph_hash": base_graph_hash,
                "mutation_path": mutation_path,
                "worker_id": self.worker_id,
                "worker_execution": "process_actor",
                "actor_pid": os.getpid(),
                "device": self.device,
            }
        )
        return IslandPreparedCandidate(
            island=self.island,
            device=self.device,
            worker_id=self.worker_id,
            actor_pid=os.getpid(),
            job_id=job_id,
            proposed_step=int(proposed_step),
            island_step=island_step,
            generation=generation,
            base_graph_hash=base_graph_hash,
            mutation_path=mutation_path,
            snapshot=self.snapshot(),
        )

    def run_prepared_candidate(self, job_id: str) -> IslandActorCandidateResult:
        if self.run_dir is None:
            raise RuntimeError("Production island worker has no run directory.")
        prepared = self._prepared_jobs.get(job_id)
        if prepared is None:
            raise RuntimeError(f"Unknown island job {job_id!r}.")
        return IslandActorCandidateResult(
            actor_pid=os.getpid(),
            candidate=self.run_candidate(
                base_graph=prepared["base_graph"],
                best_result=prepared["best_result"],
                rng_state=prepared["rng_state"],
                island_step=int(prepared["island_step"]),
                output_dir=self.run_dir,
                errors=list(prepared["errors"]),
            ),
        )

    def commit_candidate(
        self,
        prepared: IslandPreparedCandidate,
        result: IslandCandidateResult,
        decision: IslandCommitDecision,
    ) -> IslandActorCommitResult:
        state, _current_graph, _best_graph, _best_train_result, _best_validation_result = self._require_island_state()
        self._prepared_jobs.pop(prepared.job_id, None)
        outcome = result.outcome
        candidate_graph = (
            graph_from_dict(result.candidate_graph_payload, allow_source=self.allow_source_mutations)
            if result.candidate_graph_payload is not None
            else None
        )
        state.rng_state = result.rng_state
        state.errors = list(result.errors[-40:])
        if int(state.generation) != int(prepared.generation):
            event = self._stale_event(prepared, decision, result.document)
            return self._finish_commit(prepared, event, terminal_status="stale")
        if outcome.failed:
            event = self._failed_event(prepared, decision, result.document, outcome.error or "Unknown candidate failure.")
            return self._finish_commit(prepared, event, terminal_status="failed", terminal_error=str(event.get("error", "")))
        if (
            candidate_graph is None
            or outcome.result is None
            or result.validation_result is None
            or result.maintenance is None
        ):
            event = self._failed_event(prepared, decision, result.document, "Candidate evaluation returned an incomplete success outcome.")
            return self._finish_commit(prepared, event, terminal_status="failed", terminal_error=str(event.get("error", "")))

        candidate_train = outcome.result
        candidate_validation = result.validation_result
        previous_island_train_auc = float(self.best_train_result.auc) if self.best_train_result is not None else float(state.best_train_auc)
        previous_island_validation_auc = float(self.best_validation_result.auc) if self.best_validation_result is not None else float(state.best_validation_auc)
        if decision.accepted:
            self.current_graph = candidate_graph
            self.best_graph = candidate_graph.clone()
            self.best_train_result = candidate_train
            self.best_validation_result = candidate_validation
            state.archive_version += 1
            state.best_train_auc = float(candidate_train.auc)
            state.best_validation_auc = float(candidate_validation.auc)
            state.best_config = dict(candidate_train.config)
            state.generation += 1
            metadata = {
                "island": self.island,
                "device": self.device,
                "worker_id": self.worker_id,
                "worker_execution": "process_actor",
                "actor_pid": os.getpid(),
                "global_step": int(decision.global_step),
            }
            context = self._context()
            self._write_graph_artifacts(context, step=prepared.island_step, metadata=metadata)
            self._write_archive_entry(context, step=prepared.island_step, metadata={**metadata, "source": "candidate"})
            self._write_checkpoint(context, step=prepared.island_step, metadata=metadata)

        event = self._candidate_event(
            prepared,
            decision,
            mutation=result.document.to_dict(),
            train_result=candidate_train,
            validation_result=candidate_validation,
            maintenance=result.maintenance,
            previous_island_train_auc=previous_island_train_auc,
            previous_island_validation_auc=previous_island_validation_auc,
        )
        return self._finish_commit(prepared, event, terminal_status="completed")

    def apply_migration(
        self,
        *,
        source_island: int,
        global_step: int,
        global_archive_version: int,
        graph_payload: dict[str, Any],
        train_result: EvaluationResult,
        validation_result: EvaluationResult,
        best_config: dict[str, str],
        selection_metric: str,
        global_best_score: float,
    ) -> IslandActorMigrationResult:
        state, _current_graph, _best_graph, best_train_result, best_validation_result = self._require_island_state()
        previous_score = self._frontier_score(best_train_result, best_validation_result)
        previous_best_train_auc = float(best_train_result.auc)
        previous_best_validation_auc = float(best_validation_result.auc)
        migrated_graph = graph_from_dict(graph_payload, allow_source=self.allow_source_mutations)
        self.current_graph = migrated_graph
        self.best_graph = migrated_graph.clone()
        self.best_train_result = train_result
        self.best_validation_result = validation_result
        state.archive_version += 1
        state.best_train_auc = float(train_result.auc)
        state.best_validation_auc = float(validation_result.auc)
        state.best_config = dict(best_config)
        state.generation += 1
        event = {
            "mode": "production_async_migration",
            "global_step": int(global_step),
            "source_island": int(source_island),
            "target_island": self.island,
            "target_device": self.device,
            "target_worker_id": self.worker_id,
            "worker_execution": "process_actor",
            "target_actor_pid": os.getpid(),
            "target_generation": int(state.generation),
            "global_best_version": int(global_archive_version),
            "selection_metric": selection_metric,
            "previous_best_score": previous_score,
            "global_best_score": float(global_best_score),
            "previous_best_train_auc": previous_best_train_auc,
            "previous_best_validation_auc": previous_best_validation_auc,
            "global_best_validation_auc": float(validation_result.auc),
            "global_best_train_auc": float(train_result.auc),
            "graph_hash": graph_hash(migrated_graph),
        }
        self._write_island_migration_event(event)
        state.history.append(
            f"- MIGRATED: global_step={int(global_step)} source_island={source_island} "
            f"{selection_metric}={float(global_best_score):.6f}"
        )
        del state.history[:-40]
        context = self._context()
        metadata = {
            "island": self.island,
            "device": self.device,
            "worker_id": self.worker_id,
            "worker_execution": "process_actor",
            "actor_pid": os.getpid(),
            "global_step": int(global_step),
            "source": "migration",
            "source_island": int(source_island),
            "selection_metric": selection_metric,
        }
        self._write_graph_artifacts(context, step=int(state.step), metadata=metadata)
        self._write_archive_entry(context, step=int(state.step), metadata=metadata)
        self._write_checkpoint(context, step=int(state.step), metadata=metadata)
        self._write_state()
        self.write_memorandum(context)
        return IslandActorMigrationResult(actor_pid=os.getpid(), snapshot=self.snapshot(), event=event)

    def _build_loop(self, graph: Graph, rng_state: dict[str, Any]) -> EvolutionLoop:
        loop = EvolutionLoop(
            graph,
            evaluator=RidgeEvaluator(**self.evaluator_config),
            mutation_engine=MutationEngine(allow_source=self.allow_source_mutations),
            scientist=self.scientist,
            engineer=self.engineer,
            memorandum_agent=self.memorandum_agent,
            task_context=self.task_context,
            task_sources=self.task_sources,
            seed=self.seed,
        )
        loop.rng.bit_generator.state = rng_state
        return loop

    def _require_island_state(self) -> tuple[RunState, Graph, Graph, EvaluationResult, EvaluationResult]:
        if (
            self.state is None
            or self.current_graph is None
            or self.best_graph is None
            or self.best_train_result is None
            or self.best_validation_result is None
        ):
            raise RuntimeError("Production island worker has no actor-owned island state.")
        return self.state, self.current_graph, self.best_graph, self.best_train_result, self.best_validation_result

    def _context(self) -> ProductionContext:
        state, current_graph, best_graph, best_train_result, best_validation_result = self._require_island_state()
        if self.run_dir is None or self.split_manifest is None:
            raise RuntimeError("Production island worker has no run metadata.")
        return ProductionContext(
            run_dir=self.run_dir,
            manifest=self.manifest,
            split_manifest=self.split_manifest,
            splits={},
            state=state,
            current_graph=current_graph,
            best_graph=best_graph,
            best_train_result=best_train_result,
            best_validation_result=best_validation_result,
        )

    def _frontier_score(self, train_result: EvaluationResult, validation_result: EvaluationResult) -> float:
        acceptance = dict(self.manifest.get("acceptance", {}))
        if str(acceptance.get("policy", "")) == "paper_cv_auc_improvement":
            return float(train_result.auc)
        return float(validation_result.auc)

    def _candidate_event(
        self,
        prepared: IslandPreparedCandidate,
        decision: IslandCommitDecision,
        *,
        mutation: dict[str, object],
        train_result: EvaluationResult,
        validation_result: EvaluationResult,
        maintenance: dict[str, object],
        previous_island_train_auc: float,
        previous_island_validation_auc: float,
    ) -> dict[str, Any]:
        state, _current_graph, _best_graph, _best_train_result, _best_validation_result = self._require_island_state()
        return {
            "step": int(decision.global_step),
            "mode": "production_async_island",
            "job_id": prepared.job_id,
            "island": self.island,
            "island_step": int(prepared.island_step),
            "device": self.device,
            "worker_id": self.worker_id,
            "worker_execution": "process_actor",
            "actor_pid": os.getpid(),
            "base_graph_hash": prepared.base_graph_hash,
            "base_generation": int(prepared.generation),
            "generation": int(state.generation),
            "accepted": bool(decision.accepted),
            "global_best": bool(decision.global_best),
            "train_auc": float(train_result.auc),
            "validation_auc": float(validation_result.auc),
            "best_train_auc": float(decision.root_best_train_auc),
            "best_validation_auc": float(decision.root_best_validation_auc),
            "island_best_train_auc": float(state.best_train_auc),
            "island_best_validation_auc": float(state.best_validation_auc),
            "train_delta": float(train_result.auc) - previous_island_train_auc,
            "validation_delta": float(validation_result.auc) - previous_island_validation_auc,
            "global_train_delta": float(train_result.auc) - float(decision.previous_global_train_auc),
            "global_validation_delta": float(validation_result.auc) - float(decision.previous_global_validation_auc),
            "config": train_result.config,
            "mutation": mutation,
            "maintenance": maintenance,
            "promotion_policy": self.manifest["acceptance"],
        }

    def _failed_event(
        self,
        prepared: IslandPreparedCandidate,
        decision: IslandCommitDecision,
        document: MutationDocument,
        error: str,
    ) -> dict[str, Any]:
        state, _current_graph, _best_graph, _best_train_result, _best_validation_result = self._require_island_state()
        return {
            "step": int(decision.global_step),
            "mode": "production_async_island",
            "job_id": prepared.job_id,
            "island": self.island,
            "island_step": int(prepared.island_step),
            "device": self.device,
            "worker_id": self.worker_id,
            "worker_execution": "process_actor",
            "actor_pid": os.getpid(),
            "base_graph_hash": prepared.base_graph_hash,
            "base_generation": int(prepared.generation),
            "generation": int(state.generation),
            "accepted": False,
            "failed": True,
            "error": error,
            "train_auc": None,
            "validation_auc": None,
            "best_train_auc": float(decision.root_best_train_auc),
            "best_validation_auc": float(decision.root_best_validation_auc),
            "island_best_train_auc": float(state.best_train_auc),
            "island_best_validation_auc": float(state.best_validation_auc),
            "config": state.best_config,
            "mutation": document.to_dict(),
        }

    def _stale_event(
        self,
        prepared: IslandPreparedCandidate,
        decision: IslandCommitDecision,
        document: MutationDocument,
    ) -> dict[str, Any]:
        state, _current_graph, _best_graph, _best_train_result, _best_validation_result = self._require_island_state()
        return {
            "step": int(decision.global_step),
            "mode": "production_async_island",
            "job_id": prepared.job_id,
            "island": self.island,
            "island_step": int(prepared.island_step),
            "device": self.device,
            "worker_id": self.worker_id,
            "worker_execution": "process_actor",
            "actor_pid": os.getpid(),
            "base_graph_hash": prepared.base_graph_hash,
            "base_generation": int(prepared.generation),
            "generation": int(state.generation),
            "accepted": False,
            "stale": True,
            "stale_reason": "Island graph generation changed before candidate completion.",
            "train_auc": None,
            "validation_auc": None,
            "best_train_auc": float(decision.root_best_train_auc),
            "best_validation_auc": float(decision.root_best_validation_auc),
            "island_best_train_auc": float(state.best_train_auc),
            "island_best_validation_auc": float(state.best_validation_auc),
            "config": state.best_config,
            "mutation": document.to_dict(),
        }

    def _finish_commit(
        self,
        prepared: IslandPreparedCandidate,
        event: dict[str, Any],
        *,
        terminal_status: str,
        terminal_error: str = "",
    ) -> IslandActorCommitResult:
        state, _current_graph, _best_graph, _best_train_result, _best_validation_result = self._require_island_state()
        state.step = int(event["island_step"])
        ProductionEvolutionRunner._record_event(state, event)
        self._write_island_event(event)
        self._write_state()
        self.write_memorandum(self._context())
        self._write_island_job_event(
            self._job_terminal_payload(
                prepared,
                terminal_status,
                global_step=int(event["step"]),
                error=terminal_error,
            )
        )
        return IslandActorCommitResult(
            actor_pid=os.getpid(),
            snapshot=self.snapshot(),
            event=event,
            terminal_status=terminal_status,
            terminal_error=terminal_error,
        )

    def _write_graph_artifacts(self, context: ProductionContext, step: int, metadata: dict[str, Any] | None = None) -> None:
        metadata = {"run_id": context.state.run_id, "step": int(step), **(metadata or {})}
        write_graph(context.run_dir / context.state.current_graph_path, context.current_graph, metadata={**metadata, "role": "current"})
        write_graph(context.run_dir / context.state.best_graph_path, context.best_graph, metadata={**metadata, "role": "best"})

    def _write_archive_entry(self, context: ProductionContext, step: int, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        archive_dir = context.run_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        filename = f"best_v{context.state.archive_version:04d}_step_{step:04d}.json"
        payload = {
            "version": int(context.state.archive_version),
            "step": int(step),
            **metadata,
            "graph_hash": graph_hash(context.best_graph),
            "graph": context.best_graph.to_dict(),
            "train_result": context.best_train_result.to_dict(),
            "validation_result": context.best_validation_result.to_dict(),
            "feedback": feedback_summary(context.best_train_result),
            "diagnostics_toon": toon_report(context.best_train_result),
        }
        (archive_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        row = {
            "version": int(context.state.archive_version),
            "step": int(step),
            "train_auc": float(context.best_train_result.auc),
            "validation_auc": float(context.best_validation_result.auc),
            "config": context.best_train_result.config,
            "graph_hash": payload["graph_hash"],
            "path": filename,
            **metadata,
        }
        with (archive_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _write_checkpoint(self, context: ProductionContext, step: int, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "run_id": context.state.run_id,
            "step": int(step),
            **(metadata or {}),
            "archive_version": int(context.state.archive_version),
            "current_graph_path": context.state.current_graph_path,
            "best_graph_path": context.state.best_graph_path,
            "best_train_result": context.best_train_result.to_dict(),
            "best_validation_result": context.best_validation_result.to_dict(),
            "feedback": feedback_summary(context.best_train_result),
            "diagnostics_toon": toon_report(context.best_train_result),
        }
        (context.run_dir / "checkpoint.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_state(self) -> None:
        if self.run_dir is None or self.state is None:
            raise RuntimeError("Production island worker has no state path.")
        (self.run_dir / "state.json").write_text(json.dumps(self.state.to_dict(), indent=2), encoding="utf-8")

    def _write_island_event(self, event: dict[str, Any]) -> None:
        if self.run_dir is None:
            raise RuntimeError("Production island worker has no run directory.")
        island_event = {
            **event,
            "best_train_auc": event.get("island_best_train_auc", event["best_train_auc"]),
            "best_validation_auc": event.get("island_best_validation_auc", event["best_validation_auc"]),
        }
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(island_event) + "\n")

    def _write_island_job_event(self, row: dict[str, Any]) -> None:
        if self.run_dir is None:
            raise RuntimeError("Production island worker has no run directory.")
        payload = {**row, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        with (self.run_dir / "jobs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _write_island_migration_event(self, event: dict[str, Any]) -> None:
        if self.run_dir is None:
            raise RuntimeError("Production island worker has no run directory.")
        with (self.run_dir / "migrations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def _job_terminal_payload(
        self,
        prepared: IslandPreparedCandidate,
        status: str,
        *,
        global_step: int | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "job_id": prepared.job_id,
            "status": status,
            "global_step": int(global_step) if global_step is not None else None,
            "global_step_hint": int(prepared.proposed_step),
            "island": self.island,
            "island_step": int(prepared.island_step),
            "base_generation": int(prepared.generation),
            "base_graph_hash": prepared.base_graph_hash,
            "mutation_path": prepared.mutation_path,
            "worker_id": self.worker_id,
            "worker_execution": "process_actor",
            "actor_pid": os.getpid(),
            "device": self.device,
        }
        if error:
            row["error"] = error
        return row


_PROCESS_ISLAND_WORKER: ProductionIslandWorker | None = None


def _initialize_process_island_worker(
    island: int,
    device: str,
    evaluator_config: dict[str, Any],
    allow_source_mutations: bool,
    scientist: ScientistAgent | None,
    engineer: EngineerAgent | None,
    memorandum_agent: object | None,
    task_context: str,
    task_sources: tuple[tuple[str, str], ...],
    seed: int,
    train_inputs: dict[str, object],
    train_y: np.ndarray,
    validation_inputs: dict[str, object],
    validation_y: np.ndarray,
    run_dir: pathlib.Path,
    manifest: dict[str, Any],
    split_manifest: SplitManifest,
    state: RunState,
    current_graph_payload: dict[str, Any],
    best_graph_payload: dict[str, Any],
    best_train_result: EvaluationResult,
    best_validation_result: EvaluationResult,
) -> None:
    global _PROCESS_ISLAND_WORKER
    _PROCESS_ISLAND_WORKER = ProductionIslandWorker(
        island=island,
        device=device,
        evaluator_config=evaluator_config,
        allow_source_mutations=allow_source_mutations,
        scientist=scientist,
        engineer=engineer,
        memorandum_agent=memorandum_agent,
        task_context=task_context,
        task_sources=task_sources,
        seed=seed,
        train_inputs=train_inputs,
        train_y=train_y,
        validation_inputs=validation_inputs,
        validation_y=validation_y,
        run_dir=run_dir,
        manifest=manifest,
        split_manifest=split_manifest,
        state=state,
        current_graph=graph_from_dict(current_graph_payload, allow_source=allow_source_mutations),
        best_graph=graph_from_dict(best_graph_payload, allow_source=allow_source_mutations),
        best_train_result=best_train_result,
        best_validation_result=best_validation_result,
    )


def _process_island_worker() -> ProductionIslandWorker:
    if _PROCESS_ISLAND_WORKER is None:
        raise RuntimeError("Production island process actor was not initialized.")
    return _PROCESS_ISLAND_WORKER


def _process_island_pid() -> int:
    _process_island_worker()
    return os.getpid()


def _process_island_snapshot() -> IslandActorSnapshot:
    return _process_island_worker().snapshot()


def _process_island_prepare_candidate(proposed_step: int) -> IslandPreparedCandidate:
    return _process_island_worker().prepare_candidate(proposed_step)


def _process_island_run_prepared_candidate(job_id: str) -> IslandActorCandidateResult:
    return _process_island_worker().run_prepared_candidate(job_id)


def _process_island_commit_candidate(
    prepared: IslandPreparedCandidate,
    result: IslandCandidateResult,
    decision: IslandCommitDecision,
) -> IslandActorCommitResult:
    return _process_island_worker().commit_candidate(prepared, result, decision)


def _process_island_apply_migration(
    source_island: int,
    global_step: int,
    global_archive_version: int,
    graph_payload: dict[str, Any],
    train_result: EvaluationResult,
    validation_result: EvaluationResult,
    best_config: dict[str, str],
    selection_metric: str,
    global_best_score: float,
) -> IslandActorMigrationResult:
    return _process_island_worker().apply_migration(
        source_island=source_island,
        global_step=global_step,
        global_archive_version=global_archive_version,
        graph_payload=graph_payload,
        train_result=train_result,
        validation_result=validation_result,
        best_config=best_config,
        selection_metric=selection_metric,
        global_best_score=global_best_score,
    )


def _process_island_write_memorandum(payload: dict[str, Any]) -> int:
    worker = _process_island_worker()
    context = ProductionContext(
        run_dir=pathlib.Path(payload["run_dir"]),
        manifest=dict(payload["manifest"]),
        split_manifest=payload["split_manifest"],
        splits={},
        state=payload["state"],
        current_graph=graph_from_dict(payload["current_graph"], allow_source=worker.allow_source_mutations),
        best_graph=graph_from_dict(payload["best_graph"], allow_source=worker.allow_source_mutations),
        best_train_result=payload["best_train_result"],
        best_validation_result=payload["best_validation_result"],
    )
    worker.write_memorandum(context)
    return os.getpid()


def _memorandum_payload_for_actor(context: ProductionContext) -> dict[str, Any]:
    return {
        "run_dir": context.run_dir,
        "manifest": context.manifest,
        "split_manifest": context.split_manifest,
        "state": context.state,
        "current_graph": context.current_graph.to_dict(),
        "best_graph": context.best_graph.to_dict(),
        "best_train_result": context.best_train_result,
        "best_validation_result": context.best_validation_result,
    }


class ProductionIslandProcessActor:
    """Per-island OS process that owns proposal, repair, evaluation, and memoranda."""

    def __init__(
        self,
        *,
        island: int,
        device: str,
        evaluator_config: dict[str, Any],
        allow_source_mutations: bool,
        scientist: ScientistAgent | None,
        engineer: EngineerAgent | None,
        memorandum_agent: object | None,
        task_context: str,
        task_sources: tuple[tuple[str, str], ...],
        seed: int,
        train_inputs: dict[str, object],
        train_y: np.ndarray,
        validation_inputs: dict[str, object],
        validation_y: np.ndarray,
        run_dir: pathlib.Path,
        manifest: dict[str, Any],
        split_manifest: SplitManifest,
        state: RunState,
        current_graph: Graph,
        best_graph: Graph,
        best_train_result: EvaluationResult,
        best_validation_result: EvaluationResult,
    ) -> None:
        self.island = int(island)
        self.device = str(device)
        self.worker_id = f"island_{self.island}"
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_process_island_worker,
            initargs=(
                self.island,
                self.device,
                evaluator_config,
                bool(allow_source_mutations),
                scientist,
                engineer,
                memorandum_agent,
                task_context,
                task_sources,
                int(seed),
                train_inputs,
                train_y,
                validation_inputs,
                validation_y,
                run_dir,
                manifest,
                split_manifest,
                state,
                current_graph.to_dict(),
                best_graph.to_dict(),
                best_train_result,
                best_validation_result,
            ),
        )
        try:
            self.actor_pid = int(self._executor.submit(_process_island_pid).result())
            self._last_snapshot = self.snapshot()
        except Exception:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    def snapshot(self) -> IslandActorSnapshot:
        snapshot = self._executor.submit(_process_island_snapshot).result()
        self._last_snapshot = snapshot
        return snapshot

    def prepare_candidate(self, proposed_step: int) -> IslandPreparedCandidate:
        prepared = self._executor.submit(_process_island_prepare_candidate, int(proposed_step)).result()
        self._last_snapshot = prepared.snapshot
        return prepared

    def submit_candidate(self, prepared: IslandPreparedCandidate) -> concurrent.futures.Future:
        return self._executor.submit(_process_island_run_prepared_candidate, prepared.job_id)

    def commit_candidate(
        self,
        prepared: IslandPreparedCandidate,
        result: IslandCandidateResult,
        decision: IslandCommitDecision,
    ) -> IslandActorCommitResult:
        commit = self._executor.submit(_process_island_commit_candidate, prepared, result, decision).result()
        self._last_snapshot = commit.snapshot
        return commit

    def apply_migration(
        self,
        *,
        source_island: int,
        global_step: int,
        global_archive_version: int,
        graph: Graph,
        train_result: EvaluationResult,
        validation_result: EvaluationResult,
        best_config: dict[str, str],
        selection_metric: str,
        global_best_score: float,
    ) -> IslandActorMigrationResult:
        migration = self._executor.submit(
            _process_island_apply_migration,
            int(source_island),
            int(global_step),
            int(global_archive_version),
            graph.to_dict(),
            train_result,
            validation_result,
            dict(best_config),
            selection_metric,
            float(global_best_score),
        ).result()
        self._last_snapshot = migration.snapshot
        return migration

    def write_memorandum(self, context: ProductionContext) -> int:
        return int(self._executor.submit(_process_island_write_memorandum, _memorandum_payload_for_actor(context)).result())

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


class ProductionEvolutionRunner:
    def __init__(
        self,
        config: ProductionConfig,
        graph: Graph | None = None,
        *,
        scientist: ScientistAgent | None = None,
        engineer: EngineerAgent | None = None,
        memorandum_agent: object | None = None,
        task_context: str = "",
        task_sources: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.config = config
        if self.config.profile not in SUPPORTED_PRODUCTION_PROFILES:
            raise ValueError(f"Unsupported production profile {self.config.profile!r}; expected one of {', '.join(SUPPORTED_PRODUCTION_PROFILES)}.")
        if self.config.promotion_policy not in SUPPORTED_PROMOTION_POLICIES:
            raise ValueError(f"Unsupported promotion policy {self.config.promotion_policy!r}; expected one of {', '.join(SUPPORTED_PROMOTION_POLICIES)}.")
        self.graph = graph or build_seed_graph()
        self.evaluator = RidgeEvaluator(**config.evaluator_config())
        self.mutation_engine = MutationEngine(allow_source=config.allow_source_mutations)
        self.scientist = scientist
        self.engineer = engineer
        self.memorandum_agent = memorandum_agent
        self.task_context = task_context
        self.task_sources = task_sources

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        if self._should_run_async_islands(resume=resume):
            return self._run_async_islands(resume=resume)
        return self._run_single(resume=resume)

    def _should_run_async_islands(self, *, resume: bool) -> bool:
        if resume:
            manifest_path = self.config.output_dir / "run_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                islands = dict(manifest.get("islands", {}))
                return str(islands.get("mode", "single")) == "async" and int(islands.get("count", 1)) > 1
        if int(self.config.islands) <= 1:
            return False
        if not self.config.async_islands:
            raise ValueError("Production island runs require async_islands=True.")
        self._validate_paper_island_topology(self._configured_island_count(), self._configured_island_devices(), self.config.island_workers)
        return True

    def _run_single(self, *, resume: bool = False) -> dict[str, Any]:
        context = self._resume_context() if resume else self._new_context()
        loop = EvolutionLoop(
            context.current_graph,
            evaluator=self.evaluator,
            mutation_engine=self.mutation_engine,
            scientist=self.scientist,
            engineer=self.engineer,
            memorandum_agent=self.memorandum_agent,
            task_context=self.task_context,
            task_sources=self.task_sources,
            seed=self.config.seed,
        )
        loop.rng.bit_generator.state = context.state.rng_state
        events_path = context.run_dir / "events.jsonl"
        mode = "a" if resume else "w"
        train_inputs, train_y = context.splits["train"]
        validation_inputs, validation_y = context.splits["validation"]
        self._install_task_context(context, loop, train_inputs, train_y)
        self._write_memorandum(context, loop=loop)
        with events_path.open(mode, encoding="utf-8") as events:
            for step in range(context.state.step + 1, context.state.step + int(self.config.steps) + 1):
                try:
                    document = loop._propose_document(
                        context.current_graph,
                        context.best_train_result,
                        step,
                        memorandum=self._read_memorandum(context.run_dir),
                        execution_errors="\n".join(context.state.errors[-8:]),
                    )
                finally:
                    loop._write_prompt_records(context.run_dir, step)
                loop._write_mutation_document(context.run_dir, step, document)
                outcome = loop._try_evaluate_candidate(context.current_graph, document, train_inputs, train_y)
                if outcome.failed:
                    document, outcome = loop._repair_candidate(
                        context.current_graph,
                        context.best_train_result,
                        step,
                        island=None,
                        output_dir=context.run_dir,
                        document=document,
                        outcome=outcome,
                        inputs=train_inputs,
                        y=train_y,
                        memorandum=self._read_memorandum(context.run_dir),
                        errors=context.state.errors,
                    )
                if outcome.failed:
                    event = self._failed_event(context, step, document.to_dict(), outcome.error or "Unknown candidate failure.")
                    context.state.step = step
                    context.state.rng_state = loop.rng.bit_generator.state
                    self._record_event(context.state, event)
                    events.write(json.dumps(event) + "\n")
                    self._write_state(context.run_dir, context.state)
                    self._write_memorandum(context, loop=loop)
                    continue

                if outcome.application is None or outcome.candidate_graph is None or outcome.result is None:
                    raise RuntimeError("Candidate evaluation returned an incomplete success outcome.")
                candidate_train = outcome.result
                candidate_validation = self.evaluator.evaluate(
                    outcome.candidate_graph,
                    validation_inputs,
                    validation_y,
                    config=candidate_train.config,
                    update_graph=False,
                )
                previous_best_train_auc = float(context.best_train_result.auc)
                previous_best_validation_auc = float(context.best_validation_result.auc)
                accepted = self._promotes(candidate_train, candidate_validation, context)
                if accepted:
                    context.current_graph = outcome.candidate_graph
                    context.best_graph = outcome.candidate_graph.clone()
                    context.best_train_result = candidate_train
                    context.best_validation_result = candidate_validation
                    context.state.archive_version += 1
                    context.state.best_train_auc = float(candidate_train.auc)
                    context.state.best_validation_auc = float(candidate_validation.auc)
                    context.state.best_config = dict(candidate_train.config)
                    self._write_graph_artifacts(context, step)
                    self._write_archive_entry(context, step)
                    self._write_checkpoint(context, step)

                context.state.step = step
                context.state.rng_state = loop.rng.bit_generator.state
                event = self._candidate_event(
                    context,
                    step,
                    accepted=accepted,
                    mutation=document.to_dict(),
                    train_result=candidate_train,
                    validation_result=candidate_validation,
                    maintenance=outcome.application.maintenance.to_dict(),
                    previous_best_train_auc=previous_best_train_auc,
                    previous_best_validation_auc=previous_best_validation_auc,
                )
                self._record_event(context.state, event)
                events.write(json.dumps(event) + "\n")
                self._write_state(context.run_dir, context.state)
                self._write_memorandum(context, loop=loop)

        return inspect_run(context.run_dir)

    def _run_async_islands(self, *, resume: bool = False) -> dict[str, Any]:
        context, islands = self._resume_async_context() if resume else self._new_async_context()
        train_inputs, train_y = context.splits["train"]
        validation_inputs, validation_y = context.splits["validation"]
        setup_loop = self._build_loop(context.current_graph, context.state.rng_state, island=None)
        self._install_task_context(context, setup_loop, train_inputs, train_y)
        task_context = (context.run_dir / "task_context.md").read_text(encoding="utf-8")
        actors: dict[int, ProductionIslandProcessActor] = {}
        try:
            for island in islands:
                (island.run_dir / "task_context.md").write_text(task_context, encoding="utf-8")
                actors[island.island] = self._build_island_actor(
                    context,
                    island,
                    task_context,
                    train_inputs,
                    train_y,
                    validation_inputs,
                    validation_y,
                )
                actors[island.island].write_memorandum(self._island_context(context, island))

            target_step = int(context.state.step) + int(self.config.steps)
            mode = "a" if resume else "w"
            if not resume:
                for path in ("events.jsonl", "jobs.jsonl", "migrations.jsonl"):
                    output = context.run_dir / path
                    if output.exists():
                        output.unlink()
            else:
                self._abandon_open_jobs_on_resume(context, islands)
            pending: dict[concurrent.futures.Future, PendingIslandCandidate] = {}
            with (context.run_dir / "events.jsonl").open(mode, encoding="utf-8") as root_events:
                while int(context.state.step) < target_step or pending:
                    while int(context.state.step) + len(pending) < target_step:
                        available = [island for island in islands if all(int(job.island.island) != int(island.island) for job in pending.values())]
                        if not available:
                            break
                        island = min(available, key=lambda item: (int(item.state.step), item.island))
                        job = self._submit_island_candidate(
                            context,
                            island,
                            actors[island.island],
                        )
                        pending[job.future] = job
                    if not pending:
                        break
                    done, _pending = concurrent.futures.wait(tuple(pending), return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in sorted(done, key=lambda item: pending[item].proposed_step):
                        job = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            self._write_job_event(
                                context.run_dir,
                                job.island.run_dir,
                                self._job_terminal_payload(job, "failed", error=EvolutionLoop._format_exception(exc)),
                            )
                            raise RuntimeError(f"Island worker {job.island.island} failed for job {job.job_id}.") from exc
                        self._commit_island_candidate(
                            context,
                            islands,
                            actors,
                            job,
                            result,
                            root_events,
                        )
        finally:
            for actor in actors.values():
                actor.shutdown()
        return inspect_run(context.run_dir)

    def _new_async_context(self) -> tuple[ProductionContext, list[ProductionIslandContext]]:
        self._validate_paper_island_topology(self._configured_island_count(), self._configured_island_devices(), self.config.island_workers)
        self._validate_island_device_runtime(self._configured_island_devices(), self.config.evaluator_config())
        context = self._new_context()
        islands = self._initialize_island_contexts(context, self._island_count(context))
        self._install_initial_global_best_from_islands(context, islands)
        return context, islands

    def _resume_async_context(self) -> tuple[ProductionContext, list[ProductionIslandContext]]:
        context = self._resume_context()
        self._validate_paper_island_topology(self._island_count(context), self._island_devices(context), self._manifest_island_workers(context))
        self._validate_island_device_runtime(self._island_devices(context), self._active_evaluator_config())
        islands = self._load_island_contexts(context, self._island_count(context))
        return context, islands

    def _initialize_island_contexts(self, context: ProductionContext, count: int) -> list[ProductionIslandContext]:
        devices = self._island_devices(context)
        islands: list[ProductionIslandContext] = []
        for island_id in range(count):
            device = devices[island_id]
            island_dir = context.run_dir / "islands" / f"island_{island_id}"
            island_dir.mkdir(parents=True, exist_ok=True)
            island_graph = self.graph.clone()
            train_inputs, train_y = context.splits["train"]
            validation_inputs, validation_y = context.splits["validation"]
            island_evaluator = self._island_evaluator(device)
            train_result = island_evaluator.evaluate(island_graph, train_inputs, train_y, update_graph=True)
            validation_result = island_evaluator.evaluate(island_graph, validation_inputs, validation_y, config=train_result.config, update_graph=False)
            rng = np.random.default_rng(int(self.config.seed) + 1009 * (island_id + 1))
            state = RunState(
                run_id=f"{context.state.run_id}_island_{island_id}",
                step=0,
                archive_version=0,
                best_train_auc=float(train_result.auc),
                best_validation_auc=float(validation_result.auc),
                best_config=dict(train_result.config),
                current_graph_path="current_graph.json",
                best_graph_path="best_graph.json",
                rng_state=rng.bit_generator.state,
            )
            island = ProductionIslandContext(
                island=island_id,
                run_dir=island_dir,
                state=state,
                current_graph=island_graph,
                best_graph=island_graph.clone(),
                best_train_result=train_result,
                best_validation_result=validation_result,
                device=device,
            )
            island_context = self._island_context(context, island)
            metadata = {"island": island_id, "device": device, "worker_id": f"island_{island_id}", "global_step": int(context.state.step)}
            self._write_graph_artifacts(island_context, step=0, metadata=metadata)
            self._write_archive_entry(island_context, step=0, metadata={**metadata, "source": "seed"})
            self._write_checkpoint(island_context, step=0, metadata=metadata)
            self._write_state(island_dir, state)
            self._write_memorandum(island_context, island=island_id)
            islands.append(island)
        return islands

    def _load_island_contexts(self, context: ProductionContext, count: int) -> list[ProductionIslandContext]:
        allow_source = bool(dict(context.manifest.get("mutation", {})).get("allow_source_mutations", False)) or self.config.allow_source_mutations
        train_inputs, train_y = context.splits["train"]
        validation_inputs, validation_y = context.splits["validation"]
        devices = self._island_devices(context)
        islands: list[ProductionIslandContext] = []
        for island_id in range(count):
            device = devices[island_id]
            island_dir = context.run_dir / "islands" / f"island_{island_id}"
            state = self._read_state(island_dir)
            current_graph = graph_from_path(island_dir / state.current_graph_path, allow_source=allow_source)
            best_graph = graph_from_path(island_dir / state.best_graph_path, allow_source=allow_source)
            island_evaluator = self._island_evaluator(device)
            train_result = island_evaluator.evaluate(best_graph, train_inputs, train_y, config=state.best_config, update_graph=False)
            validation_result = island_evaluator.evaluate(best_graph, validation_inputs, validation_y, config=state.best_config, update_graph=False)
            islands.append(
                ProductionIslandContext(
                    island=island_id,
                    run_dir=island_dir,
                    state=state,
                    current_graph=current_graph,
                    best_graph=best_graph,
                    best_train_result=train_result,
                    best_validation_result=validation_result,
                    device=device,
                    generation=int(state.generation),
                )
            )
        return islands

    def _install_initial_global_best_from_islands(
        self,
        context: ProductionContext,
        islands: list[ProductionIslandContext],
    ) -> None:
        if not islands:
            return
        best = max(
            islands,
            key=lambda item: (
                self._frontier_score(context, item.best_train_result, item.best_validation_result),
                float(item.best_train_result.auc),
                -item.island,
            ),
        )
        context.current_graph = best.best_graph.clone()
        context.best_graph = best.best_graph.clone()
        context.best_train_result = best.best_train_result
        context.best_validation_result = best.best_validation_result
        context.state.archive_version = 0
        context.state.best_train_auc = float(best.best_train_result.auc)
        context.state.best_validation_auc = float(best.best_validation_result.auc)
        context.state.best_config = dict(best.best_train_result.config)
        context.state.current_graph_path = "current_graph.json"
        context.state.best_graph_path = "best_graph.json"
        archive_dir = context.run_dir / "archive"
        if archive_dir.exists():
            for path in archive_dir.glob("best_v*.json"):
                path.unlink()
            index = archive_dir / "index.jsonl"
            if index.exists():
                index.unlink()
        metadata = {
            "island": best.island,
            "device": best.device,
            "worker_id": f"island_{best.island}",
            "source": "seed",
            "selection_metric": self._frontier_metric(context),
        }
        self._write_graph_artifacts(context, step=0, metadata=metadata)
        self._write_archive_entry(context, step=0, metadata=metadata)
        self._write_checkpoint(context, step=0, metadata=metadata)
        self._write_state(context.run_dir, context.state)
        self._write_memorandum(context)

    def _submit_island_candidate(
        self,
        context: ProductionContext,
        island: ProductionIslandContext,
        actor: ProductionIslandProcessActor,
    ) -> PendingIslandCandidate:
        proposed_step = int(context.state.step) + 1
        prepared = actor.prepare_candidate(proposed_step)
        self._write_root_job_event(
            context.run_dir,
            {
                "job_id": prepared.job_id,
                "status": "submitted",
                "global_step_hint": proposed_step,
                "island": prepared.island,
                "island_step": prepared.island_step,
                "base_generation": int(prepared.generation),
                "base_graph_hash": prepared.base_graph_hash,
                "mutation_path": prepared.mutation_path,
                "worker_id": prepared.worker_id,
                "worker_execution": "process_actor",
                "actor_pid": int(prepared.actor_pid),
                "device": prepared.device,
            },
        )
        future = actor.submit_candidate(prepared)
        return PendingIslandCandidate(
            island=island,
            future=future,
            job_id=prepared.job_id,
            proposed_step=proposed_step,
            island_step=prepared.island_step,
            generation=int(prepared.generation),
            base_graph_hash=prepared.base_graph_hash,
            actor_pid=int(prepared.actor_pid),
            prepared=prepared,
        )

    def _commit_island_candidate(
        self,
        context: ProductionContext,
        islands: list[ProductionIslandContext],
        actors: dict[int, ProductionIslandProcessActor],
        job: PendingIslandCandidate,
        actor_result: IslandActorCandidateResult,
        root_events: Any,
    ) -> None:
        if job.prepared is None:
            raise RuntimeError(f"Island job {job.job_id} was not prepared by an actor.")
        island = self._island_by_id(islands, job.island.island)
        job.actor_pid = int(actor_result.actor_pid)
        result = actor_result.candidate
        outcome = result.outcome
        previous_global_train_auc = float(context.best_train_result.auc)
        previous_global_validation_auc = float(context.best_validation_result.auc)
        accepted = False
        global_best = False
        candidate_graph: Graph | None = None
        if (
            not outcome.failed
            and outcome.result is not None
            and result.validation_result is not None
            and result.candidate_graph_payload is not None
            and result.maintenance is not None
        ):
            allow_source = bool(dict(context.manifest.get("mutation", {})).get("allow_source_mutations", False)) or self.config.allow_source_mutations
            candidate_graph = graph_from_dict(result.candidate_graph_payload, allow_source=allow_source)
            accepted = self._promotes(outcome.result, result.validation_result, self._island_context(context, island))
            global_best = accepted and self._promotes(outcome.result, result.validation_result, context)
        if global_best:
            if candidate_graph is None or outcome.result is None or result.validation_result is None:
                raise RuntimeError("Global best decision requires a complete candidate graph and evaluation result.")
            context.current_graph = candidate_graph
            context.best_graph = candidate_graph.clone()
            context.best_train_result = outcome.result
            context.best_validation_result = result.validation_result
            context.state.archive_version += 1
            context.state.best_train_auc = float(outcome.result.auc)
            context.state.best_validation_auc = float(result.validation_result.auc)
            context.state.best_config = dict(outcome.result.config)
            context.state.generation += 1
            global_metadata = {
                "island": island.island,
                "device": island.device,
                "worker_id": f"island_{island.island}",
                "worker_execution": "process_actor",
                "actor_pid": int(job.actor_pid or 0),
                "island_step": job.island_step,
            }
            self._write_graph_artifacts(context, step=int(context.state.step) + 1, metadata=global_metadata)
            self._write_archive_entry(
                context,
                step=int(context.state.step) + 1,
                metadata={**global_metadata, "source": "global_best"},
            )
            self._write_checkpoint(context, step=int(context.state.step) + 1, metadata=global_metadata)

        decision = IslandCommitDecision(
            global_step=int(context.state.step) + 1,
            accepted=accepted,
            global_best=global_best,
            root_best_train_auc=float(context.state.best_train_auc),
            root_best_validation_auc=float(context.state.best_validation_auc),
            previous_global_train_auc=previous_global_train_auc,
            previous_global_validation_auc=previous_global_validation_auc,
        )
        commit = actors[island.island].commit_candidate(job.prepared, result, decision)
        island = self._replace_island_snapshot(context, islands, commit.snapshot)
        event = commit.event
        context.state.step = int(event["step"])
        self._record_event(context.state, event)
        root_events.write(json.dumps(event) + "\n")
        root_events.flush()
        self._write_state(context.run_dir, context.state)
        self._write_memorandum(context)
        self._write_root_job_event(
            context.run_dir,
            self._job_terminal_payload(job, commit.terminal_status, global_step=int(event["step"]), error=commit.terminal_error),
        )
        if global_best:
            self._migrate_global_best(context, islands, actors=actors, source_island=island.island)
        elif self._migration_interval(context) > 0 and int(context.state.step) % self._migration_interval(context) == 0:
            self._migrate_global_best(context, islands, actors=actors, source_island=island.island)

    def _migrate_global_best(
        self,
        context: ProductionContext,
        islands: list[ProductionIslandContext],
        *,
        actors: dict[int, ProductionIslandProcessActor] | None = None,
        source_island: int,
    ) -> None:
        candidates = [island for island in islands if island.island != source_island]
        global_score = self._frontier_score(context, context.best_train_result, context.best_validation_result)
        weaker = [
            island
            for island in candidates
            if self._frontier_score(context, island.best_train_result, island.best_validation_result) < global_score
        ]
        if not weaker:
            return
        target = min(
            weaker,
            key=lambda item: (
                self._frontier_score(context, item.best_train_result, item.best_validation_result),
                item.island,
            ),
        )
        if actors is None or target.island not in actors:
            raise RuntimeError("Production island migrations require a live target island actor.")
        migration = actors[target.island].apply_migration(
            source_island=source_island,
            global_step=int(context.state.step),
            global_archive_version=int(context.state.archive_version),
            graph=context.best_graph,
            train_result=context.best_train_result,
            validation_result=context.best_validation_result,
            best_config=context.state.best_config,
            selection_metric=self._frontier_metric(context),
            global_best_score=global_score,
        )
        self._replace_island_snapshot(context, islands, migration.snapshot)
        event = migration.event
        with (context.run_dir / "migrations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def _island_by_id(self, islands: list[ProductionIslandContext], island_id: int) -> ProductionIslandContext:
        for island in islands:
            if int(island.island) == int(island_id):
                return island
        raise KeyError(f"Unknown production island {island_id}.")

    def _replace_island_snapshot(
        self,
        context: ProductionContext,
        islands: list[ProductionIslandContext],
        snapshot: IslandActorSnapshot,
    ) -> ProductionIslandContext:
        updated = self._island_from_snapshot(context, snapshot)
        for index, island in enumerate(islands):
            if int(island.island) == int(snapshot.island):
                islands[index] = updated
                return updated
        raise KeyError(f"Unknown production island {snapshot.island}.")

    def _island_from_snapshot(self, context: ProductionContext, snapshot: IslandActorSnapshot) -> ProductionIslandContext:
        allow_source = bool(dict(context.manifest.get("mutation", {})).get("allow_source_mutations", False)) or self.config.allow_source_mutations
        return ProductionIslandContext(
            island=int(snapshot.island),
            run_dir=context.run_dir / "islands" / f"island_{int(snapshot.island)}",
            state=RunState.from_dict(snapshot.state.to_dict()),
            current_graph=graph_from_dict(snapshot.current_graph_payload, allow_source=allow_source),
            best_graph=graph_from_dict(snapshot.best_graph_payload, allow_source=allow_source),
            best_train_result=snapshot.best_train_result,
            best_validation_result=snapshot.best_validation_result,
            device=snapshot.device,
            generation=int(snapshot.state.generation),
        )

    def _island_context(self, context: ProductionContext, island: ProductionIslandContext) -> ProductionContext:
        return ProductionContext(
            run_dir=island.run_dir,
            manifest=context.manifest,
            split_manifest=context.split_manifest,
            splits=context.splits,
            state=island.state,
            current_graph=island.current_graph,
            best_graph=island.best_graph,
            best_train_result=island.best_train_result,
            best_validation_result=island.best_validation_result,
        )

    def _build_loop(self, graph: Graph, rng_state: dict[str, Any], *, island: int | None, device: str | None = None) -> EvolutionLoop:
        evaluator_config = self._active_evaluator_config()
        if device is not None:
            evaluator_config["torch_device"] = device
        loop = EvolutionLoop(
            graph,
            evaluator=RidgeEvaluator(**evaluator_config),
            mutation_engine=MutationEngine(allow_source=self.mutation_engine.allow_source),
            scientist=self.scientist,
            engineer=self.engineer,
            memorandum_agent=self.memorandum_agent,
            task_context=self.task_context,
            task_sources=self.task_sources,
            seed=int(self.config.seed) + (0 if island is None else 1009 * (int(island) + 1)),
        )
        loop.rng.bit_generator.state = rng_state
        return loop

    def _build_island_actor(
        self,
        context: ProductionContext,
        island: ProductionIslandContext,
        task_context: str,
        train_inputs: dict[str, object],
        train_y: np.ndarray,
        validation_inputs: dict[str, object],
        validation_y: np.ndarray,
    ) -> ProductionIslandProcessActor:
        return ProductionIslandProcessActor(
            island=island.island,
            device=island.device,
            evaluator_config=self._active_evaluator_config(),
            allow_source_mutations=self.mutation_engine.allow_source,
            scientist=self.scientist,
            engineer=self.engineer,
            memorandum_agent=self.memorandum_agent,
            task_context=task_context,
            task_sources=self.task_sources,
            seed=int(dict(context.manifest.get("dataset", {})).get("seed", self.config.seed)),
            train_inputs=train_inputs,
            train_y=train_y,
            validation_inputs=validation_inputs,
            validation_y=validation_y,
            run_dir=island.run_dir,
            manifest=context.manifest,
            split_manifest=context.split_manifest,
            state=island.state,
            current_graph=island.current_graph,
            best_graph=island.best_graph,
            best_train_result=island.best_train_result,
            best_validation_result=island.best_validation_result,
        )

    def _active_evaluator_config(self) -> dict[str, Any]:
        return {
            "n_splits": int(self.evaluator.n_splits),
            "seed": int(self.evaluator.seed),
            "max_configurations": int(self.evaluator.max_configurations),
            "irls_steps": int(self.evaluator.irls_steps),
            "refine_globals": bool(self.evaluator.refine_globals),
            "refine_steps": int(self.evaluator.refine_steps),
            "refine_backend": self.evaluator.refine_backend,
            "group_key": self.evaluator.group_key,
            "torch_device": self.evaluator.torch_device,
        }

    def _configured_island_count(self) -> int:
        return max(1, int(self.config.islands))

    def _island_count(self, context: ProductionContext) -> int:
        islands = dict(context.manifest.get("islands", {}))
        return max(1, int(islands.get("count", self.config.islands)))

    def _configured_island_devices(self) -> tuple[str, ...]:
        if self.config.island_devices is None:
            return PAPER_GPU_DEVICES if self._configured_island_count() > 1 else ("cpu",)
        return tuple(str(device) for device in self.config.island_devices)

    def _island_devices(self, context: ProductionContext) -> tuple[str, ...]:
        islands = dict(context.manifest.get("islands", {}))
        devices = islands.get("devices")
        if isinstance(devices, list) and devices:
            return tuple(str(device) for device in devices)
        return self._configured_island_devices()

    def _scientist_temperature_schedule(self) -> tuple[float, ...]:
        temperatures = getattr(self.scientist, "island_temperatures", None)
        if temperatures:
            return tuple(float(item) for item in temperatures)
        return DEFAULT_ISLAND_TEMPERATURES

    def _engineer_temperature(self) -> float | None:
        temperature = getattr(self.engineer, "temperature", None)
        return None if temperature is None else float(temperature)

    def _async_worker_count(self, context: ProductionContext) -> int:
        islands = dict(context.manifest.get("islands", {}))
        configured = islands.get("workers", self.config.island_workers)
        if configured is None:
            return self._island_count(context)
        return max(1, int(configured))

    def _manifest_island_workers(self, context: ProductionContext) -> int | None:
        islands = dict(context.manifest.get("islands", {}))
        workers = islands.get("workers", self.config.island_workers)
        return None if workers is None else int(workers)

    def _island_evaluator(self, device: str) -> RidgeEvaluator:
        return RidgeEvaluator(**{**self._active_evaluator_config(), "torch_device": device})

    @staticmethod
    def _validate_paper_island_topology(count: int, devices: tuple[str, ...], workers: int | None) -> None:
        if count <= 1:
            return
        if count != PAPER_ISLAND_COUNT:
            raise ValueError(f"Production island-native runs require exactly {PAPER_ISLAND_COUNT} islands.")
        if workers is not None and int(workers) != PAPER_ISLAND_COUNT:
            raise ValueError(f"Production island-native runs require one worker per island ({PAPER_ISLAND_COUNT} workers).")
        if len(devices) != PAPER_ISLAND_COUNT:
            raise ValueError(f"Production island-native runs require exactly {PAPER_ISLAND_COUNT} device assignments.")
        if len(set(devices)) != len(devices):
            raise ValueError("Production island-native runs require unique dedicated devices per island.")

    @staticmethod
    def _validate_island_device_runtime(devices: tuple[str, ...], evaluator_config: dict[str, Any]) -> None:
        if not bool(evaluator_config.get("refine_globals", True)):
            return
        if str(evaluator_config.get("refine_backend", "auto")) == "numpy":
            return
        cuda_devices = [device for device in devices if device.startswith("cuda")]
        if not cuda_devices:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Dedicated CUDA island runs require PyTorch when global refinement is enabled.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Dedicated CUDA island runs require CUDA to be available, or pass --island-devices with non-CUDA slots for smoke tests.")
        count = int(torch.cuda.device_count())
        missing = []
        for device in cuda_devices:
            _, _, raw_index = device.partition(":")
            index = int(raw_index or "0")
            if index >= count:
                missing.append(device)
        if missing:
            raise RuntimeError(f"Dedicated CUDA island devices are not available: {', '.join(missing)}.")

    def _migration_interval(self, context: ProductionContext) -> int:
        islands = dict(context.manifest.get("islands", {}))
        return max(0, int(islands.get("migration_interval", self.config.migration_interval)))

    @staticmethod
    def _write_job_event(root_dir: pathlib.Path, island_dir: pathlib.Path, row: dict[str, Any]) -> None:
        payload = {**row, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        for directory in (root_dir, island_dir):
            with (directory / "jobs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    @staticmethod
    def _write_root_job_event(root_dir: pathlib.Path, row: dict[str, Any]) -> None:
        payload = {**row, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        with (root_dir / "jobs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _job_terminal_payload(
        self,
        job: PendingIslandCandidate,
        status: str,
        *,
        global_step: int | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "job_id": job.job_id,
            "status": status,
            "global_step": int(global_step) if global_step is not None else None,
            "global_step_hint": int(job.proposed_step),
            "island": int(job.island.island),
            "island_step": int(job.island_step),
            "base_generation": int(job.generation),
            "base_graph_hash": job.base_graph_hash,
            "mutation_path": f"islands/island_{job.island.island}/mutations/island_{job.island.island}_step_{job.island_step:04d}.yaml",
            "worker_id": f"island_{job.island.island}",
            "worker_execution": "process_actor",
            "actor_pid": int(job.actor_pid) if job.actor_pid is not None else None,
            "device": job.island.device,
        }
        if error:
            row["error"] = error
        return row

    def _abandon_open_jobs_on_resume(
        self,
        context: ProductionContext,
        islands: list[ProductionIslandContext],
    ) -> None:
        jobs_path = context.run_dir / "jobs.jsonl"
        if not jobs_path.exists() or not jobs_path.read_text(encoding="utf-8").strip():
            return
        terminal_statuses = {"completed", "failed", "stale", "abandoned_on_resume"}
        submitted: dict[str, dict[str, Any]] = {}
        terminal: set[str] = set()
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            job_id = str(row.get("job_id", ""))
            if not job_id:
                continue
            status = str(row.get("status", ""))
            if status == "submitted":
                submitted[job_id] = row
            elif status in terminal_statuses:
                terminal.add(job_id)
        island_by_id = {island.island: island for island in islands}
        for job_id, row in sorted(submitted.items()):
            if job_id in terminal:
                continue
            island_id = int(row.get("island", -1))
            island = island_by_id.get(island_id)
            if island is None:
                continue
            self._write_job_event(
                context.run_dir,
                island.run_dir,
                {
                    **row,
                    "status": "abandoned_on_resume",
                    "global_step": int(context.state.step),
                    "abandoned_reason": "Run resumed before the submitted island job reached a terminal commit.",
                },
            )

    def _install_task_context(
        self,
        context: ProductionContext,
        loop: EvolutionLoop,
        train_inputs: dict[str, object],
        train_y: np.ndarray,
    ) -> None:
        task_context = self.task_context.strip()
        if not task_context:
            task_context = build_task_context(
                train_inputs,
                train_y,
                self.evaluator,
                source="production train split",
                task_sources=self.task_sources,
            ).to_text()
        loop._install_task_context(task_context)
        (context.run_dir / "task_context.md").write_text(task_context if task_context.endswith("\n") else task_context + "\n", encoding="utf-8")

    def _new_context(self) -> ProductionContext:
        run_dir = self.config.output_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        inputs, y, dataset_metadata = load_dataset_with_metadata(self.config.dataset_config())
        split_manifest = make_production_split_manifest(
            inputs,
            y,
            dataset_name=self.config.dataset_name,
            seed=self.config.split_seed if self.config.split_seed is not None else self.config.seed,
            validation_fraction=self.config.validation_fraction,
            test_fraction=self.config.test_fraction,
        )
        write_split_manifest(run_dir / "splits.json", split_manifest)
        splits = split_dataset(inputs, y, split_manifest)
        train_inputs, train_y = splits["train"]
        validation_inputs, validation_y = splits["validation"]
        current_graph = self.graph.clone()
        best_train = self.evaluator.evaluate(current_graph, train_inputs, train_y, update_graph=True)
        best_validation = self.evaluator.evaluate(current_graph, validation_inputs, validation_y, config=best_train.config, update_graph=False)
        rng = np.random.default_rng(self.config.seed)
        state = RunState(
            run_id=_run_id(),
            step=0,
            archive_version=0,
            best_train_auc=float(best_train.auc),
            best_validation_auc=float(best_validation.auc),
            best_config=dict(best_train.config),
            current_graph_path="current_graph.json",
            best_graph_path="best_graph.json",
            rng_state=rng.bit_generator.state,
        )
        manifest = self._run_manifest(state.run_id, split_manifest, dataset_metadata=dataset_metadata)
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        context = ProductionContext(
            run_dir=run_dir,
            manifest=manifest,
            split_manifest=split_manifest,
            splits=splits,
            state=state,
            current_graph=current_graph,
            best_graph=current_graph.clone(),
            best_train_result=best_train,
            best_validation_result=best_validation,
        )
        self._write_graph_artifacts(context, step=0)
        self._write_archive_entry(context, step=0)
        self._write_checkpoint(context, step=0)
        self._write_state(run_dir, state)
        self._write_memorandum(context)
        return context

    def _resume_context(self) -> ProductionContext:
        run_dir = self.config.output_dir
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.evaluator = RidgeEvaluator(**manifest["evaluator"])
        allow_source = bool(dict(manifest.get("mutation", {})).get("allow_source_mutations", False)) or self.config.allow_source_mutations
        self.mutation_engine = MutationEngine(allow_source=allow_source)
        split_manifest = read_split_manifest(run_dir / "splits.json")
        inputs, y = load_dataset(dict(manifest["dataset"]))
        splits = split_dataset(inputs, y, split_manifest)
        state = self._read_state(run_dir)
        current_graph = graph_from_path(run_dir / state.current_graph_path, allow_source=allow_source)
        best_graph = graph_from_path(run_dir / state.best_graph_path, allow_source=allow_source)
        train_inputs, train_y = splits["train"]
        validation_inputs, validation_y = splits["validation"]
        best_train = self.evaluator.evaluate(best_graph, train_inputs, train_y, config=state.best_config, update_graph=False)
        best_validation = self.evaluator.evaluate(best_graph, validation_inputs, validation_y, config=state.best_config, update_graph=False)
        return ProductionContext(
            run_dir=run_dir,
            manifest=manifest,
            split_manifest=split_manifest,
            splits=splits,
            state=state,
            current_graph=current_graph,
            best_graph=best_graph,
            best_train_result=best_train,
            best_validation_result=best_validation,
        )

    def _promotes(
        self,
        train_result: EvaluationResult,
        validation_result: EvaluationResult,
        context: ProductionContext,
    ) -> bool:
        train_delta = float(train_result.auc) - float(context.best_train_result.auc)
        validation_delta = float(validation_result.auc) - float(context.best_validation_result.auc)
        acceptance = dict(context.manifest.get("acceptance", {}))
        policy = str(acceptance.get("policy", self._acceptance_policy_name()))
        min_train = float(acceptance.get("min_train_improvement", self.config.min_train_improvement))
        min_validation = float(acceptance.get("min_validation_improvement", self.config.min_validation_improvement))
        if policy == "paper_cv_auc_improvement":
            return train_delta > min_train
        return train_delta > min_train and validation_delta > min_validation

    def _frontier_score(
        self,
        context: ProductionContext,
        train_result: EvaluationResult,
        validation_result: EvaluationResult,
    ) -> float:
        acceptance = dict(context.manifest.get("acceptance", {}))
        if str(acceptance.get("policy", self._acceptance_policy_name())) == "paper_cv_auc_improvement":
            return float(train_result.auc)
        return float(validation_result.auc)

    def _frontier_metric(self, context: ProductionContext) -> str:
        acceptance = dict(context.manifest.get("acceptance", {}))
        return str(acceptance.get("metric", "validation_roc_auc"))

    def _acceptance_policy_name(self) -> str:
        if self.config.promotion_policy == CV_AUC_PROMOTION_POLICY:
            return "paper_cv_auc_improvement"
        if self.config.min_train_improvement < 0.0:
            return "validation_improvement_with_train_regression_floor"
        return "train_improvement_and_validation_improvement"

    def _profile_spec(self) -> dict[str, Any]:
        if self.config.profile == PAPER_PROFILE:
            return {
                "name": PAPER_PROFILE,
                "source": "EvoForest paper long-run contract",
                "target_steps": PAPER_PROFILE_STEPS,
                "islands": PAPER_ISLAND_COUNT,
                "devices": list(PAPER_GPU_DEVICES),
                "scientist_temperature_schedule": list(DEFAULT_ISLAND_TEMPERATURES),
                "engineer_temperature": 0.0,
                "max_configurations": 64,
                "global_refinement": "pytorch_l_bfgs",
                "promotion_metric": "train_cv_roc_auc",
                "validation_gate": False,
            }
        return {
            "name": PRODUCTION_PROFILE,
            "promotion_metric": "validation_roc_auc",
            "validation_gate": True,
        }

    def _run_manifest(
        self,
        run_id: str,
        split_manifest: SplitManifest,
        *,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation": "evoforest-reimplementation",
            "paper": {"arxiv_abs": "https://arxiv.org/abs/2604.19761"},
            "profile": self.config.profile,
            "profile_spec": self._profile_spec(),
            "git_commit": _git_commit(),
            "dataset": self.config.dataset_config(),
            "dataset_metadata": dataset_metadata or {},
            "dataset_fingerprint": split_manifest.dataset_fingerprint,
            "split_manifest_path": "splits.json",
            "evaluator": self.config.evaluator_config(),
            "mutation": {"allow_source_mutations": bool(self.config.allow_source_mutations)},
            "islands": {
                "mode": "async" if int(self.config.islands) > 1 and self.config.async_islands else "single",
                "count": max(1, int(self.config.islands)),
                "topology": "paper_dedicated_gpu" if int(self.config.islands) > 1 else "single",
                "workers": self.config.island_workers if self.config.island_workers is not None else max(1, int(self.config.islands)),
                "worker_execution": "process_actor" if int(self.config.islands) > 1 and self.config.async_islands else "in_process",
                "devices": list(self._configured_island_devices()),
                "scientist_temperature_schedule": list(self._scientist_temperature_schedule()),
                "engineer_temperature": self._engineer_temperature(),
                "migration_interval": max(0, int(self.config.migration_interval)),
                "state_dir": "islands",
            },
            "acceptance": {
                "policy": self._acceptance_policy_name(),
                "metric": "train_cv_roc_auc" if self.config.promotion_policy == CV_AUC_PROMOTION_POLICY else "validation_roc_auc",
                "validation_gate": self.config.promotion_policy == VALIDATION_PROMOTION_POLICY,
                "min_train_improvement": float(self.config.min_train_improvement),
                "min_validation_improvement": float(self.config.min_validation_improvement),
                "validation_config": (
                    "reported_with_candidate_train_best_config_not_used_for_promotion"
                    if self.config.promotion_policy == CV_AUC_PROMOTION_POLICY
                    else "candidate_train_best_config"
                ),
            },
            "test_policy": (
                "test split is not evaluated by evolve; use recheck --include-test to consume it explicitly."
            ),
            "safe_staged_execution_rules": list(SAFE_STAGED_EXECUTION_RULES),
        }

    def _write_graph_artifacts(self, context: ProductionContext, step: int, metadata: dict[str, Any] | None = None) -> None:
        metadata = {"run_id": context.state.run_id, "step": int(step), **(metadata or {})}
        write_graph(context.run_dir / context.state.current_graph_path, context.current_graph, metadata={**metadata, "role": "current"})
        write_graph(context.run_dir / context.state.best_graph_path, context.best_graph, metadata={**metadata, "role": "best"})

    def _write_archive_entry(self, context: ProductionContext, step: int, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        archive_dir = context.run_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        filename = f"best_v{context.state.archive_version:04d}_step_{step:04d}.json"
        payload = {
            "version": int(context.state.archive_version),
            "step": int(step),
            **metadata,
            "graph_hash": graph_hash(context.best_graph),
            "graph": context.best_graph.to_dict(),
            "train_result": context.best_train_result.to_dict(),
            "validation_result": context.best_validation_result.to_dict(),
            "feedback": feedback_summary(context.best_train_result),
            "diagnostics_toon": toon_report(context.best_train_result),
        }
        (archive_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        row = {
            "version": int(context.state.archive_version),
            "step": int(step),
            "train_auc": float(context.best_train_result.auc),
            "validation_auc": float(context.best_validation_result.auc),
            "config": context.best_train_result.config,
            "graph_hash": payload["graph_hash"],
            "path": filename,
            **metadata,
        }
        with (archive_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _write_checkpoint(self, context: ProductionContext, step: int, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "run_id": context.state.run_id,
            "step": int(step),
            **(metadata or {}),
            "archive_version": int(context.state.archive_version),
            "current_graph_path": context.state.current_graph_path,
            "best_graph_path": context.state.best_graph_path,
            "best_train_result": context.best_train_result.to_dict(),
            "best_validation_result": context.best_validation_result.to_dict(),
            "feedback": feedback_summary(context.best_train_result),
            "diagnostics_toon": toon_report(context.best_train_result),
        }
        (context.run_dir / "checkpoint.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_memorandum(self, context: ProductionContext, loop: EvolutionLoop | None = None, island: int | None = None) -> None:
        self._write_memorandum_static(context, loop=loop, island=island)

    @staticmethod
    def _write_memorandum_static(context: ProductionContext, loop: EvolutionLoop | None = None, island: int | None = None) -> None:
        if loop is not None and getattr(loop, "memorandum_agent", None) is not None:
            loop._write_memorandum(
                context.run_dir,
                context.best_train_result,
                context.state.history,
                context.state.errors,
                step=context.state.step,
                island=island,
            )
            return
        feedback = feedback_summary(context.best_train_result)
        lines = [
            "# Production Evolution Memorandum",
            "",
            "[OUTCOME HISTORY]",
        ]
        lines.extend(context.state.history[-12:] or ["- No mutation outcomes recorded yet."])
        lines.extend(
            [
                "",
                "[STATE]",
                f"- Run id: {context.state.run_id}.",
                f"- Step: {context.state.step}.",
                f"- Best train AUC: {context.state.best_train_auc:.6f}.",
                f"- Best validation AUC: {context.state.best_validation_auc:.6f}.",
                f"- Test rechecks: {context.state.test_recheck_count}.",
            ]
        )
        scoring = feedback.get("scoring_context", {})
        if isinstance(scoring, dict):
            lines.append(
                "- Representation: "
                f"effective_rank={float(scoring.get('effective_rank', 0.0)):.4f}, "
                f"mean_max_corr={float(scoring.get('mean_max_corr', 0.0)):.4f}."
            )
        lines.extend(["", "[WHAT WORKS]"])
        lines.extend([row for row in context.state.history if "ACCEPTED" in row][-6:] or ["- No accepted mutation patterns recorded yet."])
        lines.extend(["", "[WHAT FAILED]"])
        lines.extend([row for row in context.state.history if "REJECTED" in row or "FAILED" in row][-6:] or ["- No rejected or failed mutation patterns recorded yet."])
        lines.extend(
            [
                "",
                "[ERROR LOG]",
            ]
        )
        lines.extend(context.state.errors[-12:] or ["- No runtime errors recorded."])
        (context.run_dir / "memorandum.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _candidate_event(
        self,
        context: ProductionContext,
        step: int,
        *,
        accepted: bool,
        mutation: dict[str, object],
        train_result: EvaluationResult,
        validation_result: EvaluationResult,
        maintenance: dict[str, object],
        previous_best_train_auc: float,
        previous_best_validation_auc: float,
    ) -> dict[str, Any]:
        return {
            "step": int(step),
            "mode": "production_single",
            "accepted": bool(accepted),
            "train_auc": float(train_result.auc),
            "validation_auc": float(validation_result.auc),
            "best_train_auc": float(context.state.best_train_auc),
            "best_validation_auc": float(context.state.best_validation_auc),
            "train_delta": float(train_result.auc) - float(previous_best_train_auc),
            "validation_delta": float(validation_result.auc) - float(previous_best_validation_auc),
            "config": train_result.config,
            "mutation": mutation,
            "maintenance": maintenance,
            "promotion_policy": context.manifest["acceptance"],
        }

    def _failed_event(self, context: ProductionContext, step: int, mutation: dict[str, object], error: str) -> dict[str, Any]:
        return {
            "step": int(step),
            "mode": "production_single",
            "accepted": False,
            "failed": True,
            "error": error,
            "train_auc": None,
            "validation_auc": None,
            "best_train_auc": float(context.state.best_train_auc),
            "best_validation_auc": float(context.state.best_validation_auc),
            "config": context.state.best_config,
            "mutation": mutation,
        }

    @staticmethod
    def _record_event(state: RunState, event: dict[str, Any]) -> None:
        status = "FAILED" if event.get("failed") else "ACCEPTED" if event.get("accepted") else "REJECTED"
        train = event.get("train_auc")
        validation = event.get("validation_auc")
        train_text = "n/a" if train is None else f"{float(train):.6f}"
        validation_text = "n/a" if validation is None else f"{float(validation):.6f}"
        state.history.append(
            f"- {status}: step={int(event['step'])} train={train_text} validation={validation_text} "
            f"best_validation={float(event['best_validation_auc']):.6f}"
        )
        del state.history[:-40]
        if event.get("failed"):
            state.errors.append(f"- step={int(event['step'])}: {event.get('error', '')}")
            del state.errors[:-40]

    @staticmethod
    def _read_state(run_dir: pathlib.Path) -> RunState:
        return RunState.from_dict(json.loads((run_dir / "state.json").read_text(encoding="utf-8")))

    @staticmethod
    def _write_state(run_dir: pathlib.Path, state: RunState) -> None:
        (run_dir / "state.json").write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def _read_memorandum(run_dir: pathlib.Path) -> str:
        path = run_dir / "memorandum.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""


def load_dataset(dataset_config: dict[str, Any]) -> tuple[dict[str, object], np.ndarray]:
    inputs, y, _metadata = load_dataset_with_metadata(dataset_config)
    return inputs, y


def _clone_agent_for_island(agent: object | None) -> object | None:
    if agent is None:
        return None
    if isinstance(agent, LLMScientistAgent):
        return LLMScientistAgent(
            agent.client,
            prompt_builder=copy.deepcopy(agent.prompt_builder),
            temperature=agent.temperature,
            island_temperatures=agent.island_temperatures,
        )
    if isinstance(agent, LLMEngineerAgent):
        return LLMEngineerAgent(
            agent.client,
            prompt_builder=copy.deepcopy(agent.prompt_builder),
            registry=agent.registry,
            temperature=agent.temperature,
            allow_source=agent.allow_source,
        )
    if isinstance(agent, LLMMemorandumAgent):
        return LLMMemorandumAgent(
            agent.client,
            prompt_builder=copy.deepcopy(agent.prompt_builder),
            temperature=agent.temperature,
        )
    try:
        return copy.deepcopy(agent)
    except Exception:
        return agent


def load_dataset_with_metadata(dataset_config: dict[str, Any]) -> tuple[dict[str, object], np.ndarray, dict[str, Any]]:
    name = str(dataset_config.get("name", "synthetic-structural-break"))
    if name == "synthetic-structural-break":
        dataset = make_structural_break_data(
            n_series=int(dataset_config.get("n_series", 240)),
            length=int(dataset_config.get("length", 160)),
            boundary=dataset_config.get("boundary"),
            seed=int(dataset_config.get("seed", 0)),
        )
        return dataset.inputs(), dataset.y, {
            "name": name,
            "n_samples": int(dataset.y.shape[0]),
            "positive_count": int(np.sum(dataset.y)),
            "negative_count": int(dataset.y.shape[0] - np.sum(dataset.y)),
            "mapping": "synthetic structural break generator",
        }
    raise ValueError(f"Unsupported dataset {name!r}.")


def load_external_reduced_test(dataset_config: dict[str, Any]) -> tuple[dict[str, object], np.ndarray, dict[str, Any]] | None:
    _ = dataset_config
    return None


def make_production_split_manifest(
    inputs: dict[str, object],
    y: np.ndarray,
    *,
    dataset_name: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> SplitManifest:
    _ = dataset_name
    return make_split_manifest(
        inputs,
        y,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def inspect_run(run_dir: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(run_dir)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    split_manifest = read_split_manifest(path / "splits.json")
    state = RunState.from_dict(json.loads((path / "state.json").read_text(encoding="utf-8")))
    events_path = path / "events.jsonl"
    event_count = 0
    if events_path.exists() and events_path.read_text(encoding="utf-8").strip():
        event_count = len(events_path.read_text(encoding="utf-8").splitlines())
    summary = {
        "run_id": state.run_id,
        "run_dir": str(path),
        "profile": manifest.get("profile", PRODUCTION_PROFILE),
        "dataset": manifest["dataset"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "step": int(state.step),
        "archive_version": int(state.archive_version),
        "best_train_auc": float(state.best_train_auc),
        "best_validation_auc": float(state.best_validation_auc),
        "best_config": state.best_config,
        "split_sizes": {
            "train": len(split_manifest.train_indices),
            "validation": len(split_manifest.validation_indices),
            "test": len(split_manifest.test_indices),
        },
        "event_count": event_count,
        "test_recheck_count": int(state.test_recheck_count),
        "acceptance": manifest.get("acceptance", {}),
        "artifacts": {
            "run_manifest": str(path / "run_manifest.json"),
            "splits": str(path / "splits.json"),
            "state": str(path / "state.json"),
            "best_graph": str(path / state.best_graph_path),
            "current_graph": str(path / state.current_graph_path),
            "archive_index": str(path / "archive" / "index.jsonl"),
        },
        "safe_staged_execution_rules": manifest.get("safe_staged_execution_rules", []),
    }
    islands = dict(manifest.get("islands", {}))
    if str(islands.get("mode", "single")) == "async" and int(islands.get("count", 1)) > 1:
        island_rows = []
        for island_id in range(int(islands.get("count", 1))):
            island_dir = path / "islands" / f"island_{island_id}"
            island_state = RunState.from_dict(json.loads((island_dir / "state.json").read_text(encoding="utf-8")))
            island_events_path = island_dir / "events.jsonl"
            island_events = 0
            if island_events_path.exists() and island_events_path.read_text(encoding="utf-8").strip():
                island_events = len(island_events_path.read_text(encoding="utf-8").splitlines())
            island_rows.append(
                {
                    "island": island_id,
                    "device": str((islands.get("devices") or [""] * int(islands.get("count", 1)))[island_id]),
                    "worker_id": f"island_{island_id}",
                    "worker_execution": islands.get("worker_execution", "unknown"),
                    "step": int(island_state.step),
                    "generation": int(island_state.generation),
                    "archive_version": int(island_state.archive_version),
                    "best_train_auc": float(island_state.best_train_auc),
                    "best_validation_auc": float(island_state.best_validation_auc),
                    "event_count": island_events,
                    "state": str(island_dir / "state.json"),
                    "best_graph": str(island_dir / island_state.best_graph_path),
                    "current_graph": str(island_dir / island_state.current_graph_path),
                    "memorandum": str(island_dir / "memorandum.md"),
                }
            )
        migrations_path = path / "migrations.jsonl"
        migration_count = 0
        if migrations_path.exists() and migrations_path.read_text(encoding="utf-8").strip():
            migration_count = len(migrations_path.read_text(encoding="utf-8").splitlines())
        summary["islands"] = {
            "mode": "async",
            "topology": islands.get("topology", "async"),
            "count": int(islands.get("count", 1)),
            "workers": islands.get("workers"),
            "worker_execution": islands.get("worker_execution", "unknown"),
            "devices": islands.get("devices", []),
            "scientist_temperature_schedule": islands.get("scientist_temperature_schedule", []),
            "migration_interval": int(islands.get("migration_interval", 0)),
            "migration_count": migration_count,
            "items": island_rows,
        }
        summary["artifacts"]["migrations"] = str(migrations_path)
        summary["artifacts"]["jobs"] = str(path / "jobs.jsonl")
    return summary


def export_best_graph(run_dir: str | pathlib.Path, output_path: str | pathlib.Path, *, allow_source: bool = False) -> pathlib.Path:
    path = pathlib.Path(run_dir)
    state = RunState.from_dict(json.loads((path / "state.json").read_text(encoding="utf-8")))
    graph = graph_from_path(path / state.best_graph_path, allow_source=allow_source)
    return write_graph(output_path, graph, metadata={"source_run": state.run_id, "source_step": state.step, "role": "exported_best"})


def recheck_run(run_dir: str | pathlib.Path, *, include_test: bool = False, allow_source: bool = False) -> dict[str, Any]:
    path = pathlib.Path(run_dir)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    split_manifest = read_split_manifest(path / "splits.json")
    inputs, y = load_dataset(dict(manifest["dataset"]))
    splits = split_dataset(inputs, y, split_manifest)
    state = RunState.from_dict(json.loads((path / "state.json").read_text(encoding="utf-8")))
    graph = graph_from_path(path / state.best_graph_path, allow_source=allow_source)
    evaluator = RidgeEvaluator(**manifest["evaluator"])
    result: dict[str, Any] = {
        "run_id": state.run_id,
        "step": int(state.step),
        "graph_hash": graph_hash(graph),
        "config": state.best_config,
        "splits": {},
        "test_included": bool(include_test),
    }
    for split_name in ("train", "validation"):
        split_inputs, split_y = splits[split_name]
        split_result = evaluator.evaluate(graph, split_inputs, split_y, config=state.best_config, update_graph=False)
        result["splits"][split_name] = {
            "auc": float(split_result.auc),
            "n_samples": int(split_y.shape[0]),
            "config": split_result.config,
        }
    if include_test:
        test_inputs, test_y = splits["test"]
        test_result = evaluator.evaluate(graph, test_inputs, test_y, config=state.best_config, update_graph=False)
        result["splits"]["test"] = {
            "auc": float(test_result.auc),
            "n_samples": int(test_y.shape[0]),
            "config": test_result.config,
        }
        reduced_test = load_external_reduced_test(dict(manifest["dataset"]))
        if reduced_test is not None:
            reduced_inputs, reduced_y, reduced_metadata = reduced_test
            reduced_result = evaluator.evaluate(graph, reduced_inputs, reduced_y, config=state.best_config, update_graph=False)
            result["splits"]["reduced_test"] = {
                "auc": float(reduced_result.auc),
                "n_samples": int(reduced_y.shape[0]),
                "config": reduced_result.config,
                "metadata": reduced_metadata,
                "official_metric_note": "Reduced test is evaluated only because include_test=True was explicitly requested.",
            }
        state.test_recheck_count += 1
        (path / "state.json").write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (path / "validation_rechecks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    return result


def _run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
