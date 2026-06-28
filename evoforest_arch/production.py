from __future__ import annotations

import concurrent.futures
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import pathlib
import subprocess
from typing import Any

import numpy as np

from evoforest_arch.agents import EngineerAgent, ScientistAgent
from evoforest_arch.evaluator import EvaluationResult, RidgeEvaluator
from evoforest_arch.evolution import CandidateOutcome, EvolutionLoop
from evoforest_arch.feedback import feedback_summary, toon_report
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import graph_hash, graph_from_path, write_graph
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
class PendingIslandCandidate:
    island: ProductionIslandContext
    future: concurrent.futures.Future
    job_id: str
    proposed_step: int
    island_step: int
    generation: int
    base_graph_hash: str


@dataclass
class IslandCandidateResult:
    document: MutationDocument
    outcome: CandidateOutcome
    validation_result: EvaluationResult | None
    rng_state: dict[str, Any]
    errors: list[str]


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

    def run_candidate(
        self,
        *,
        base_graph: Graph,
        best_result: EvaluationResult,
        rng_state: dict[str, Any],
        island_step: int,
        output_dir: pathlib.Path,
        train_inputs: dict[str, object],
        train_y: np.ndarray,
        validation_inputs: dict[str, object],
        validation_y: np.ndarray,
        errors: list[str],
    ) -> IslandCandidateResult:
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
        return IslandCandidateResult(
            document=document,
            outcome=outcome,
            validation_result=validation_result,
            rng_state=loop.rng.bit_generator.state,
            errors=worker_errors[-40:],
        )

    def write_memorandum(self, context: ProductionContext) -> None:
        loop = self._build_loop(context.current_graph, context.state.rng_state)
        loop._install_task_context(self.task_context)
        ProductionEvolutionRunner._write_memorandum_static(context, loop=loop, island=self.island)

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
        workers = {
            island.island: self._build_island_worker(context, island, task_context)
            for island in islands
        }
        for island in islands:
            (island.run_dir / "task_context.md").write_text(task_context, encoding="utf-8")
            workers[island.island].write_memorandum(self._island_context(context, island))

        target_step = int(context.state.step) + int(self.config.steps)
        mode = "a" if resume else "w"
        if not resume:
            for path in ("events.jsonl", "jobs.jsonl", "migrations.jsonl"):
                output = context.run_dir / path
                if output.exists():
                    output.unlink()
        else:
            self._abandon_open_jobs_on_resume(context, islands)
        worker_count = self._async_worker_count(context)
        pending: dict[concurrent.futures.Future, PendingIslandCandidate] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            with (context.run_dir / "events.jsonl").open(mode, encoding="utf-8") as root_events:
                while int(context.state.step) < target_step or pending:
                    while int(context.state.step) + len(pending) < target_step:
                        available = [island for island in islands if all(job.island is not island for job in pending.values())]
                        if not available:
                            break
                        island = min(available, key=lambda item: (int(item.state.step), item.island))
                        job = self._submit_island_candidate(
                            context,
                            island,
                            workers[island.island],
                            train_inputs,
                            train_y,
                            validation_inputs,
                            validation_y,
                            executor,
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
                            workers,
                            job,
                            result,
                            root_events,
                        )
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
        worker: ProductionIslandWorker,
        train_inputs: dict[str, object],
        train_y: np.ndarray,
        validation_inputs: dict[str, object],
        validation_y: np.ndarray,
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> PendingIslandCandidate:
        island_step = int(island.state.step) + 1
        proposed_step = int(context.state.step) + 1
        base_graph = island.current_graph.clone()
        base_graph_hash = graph_hash(base_graph)
        job_id = f"g{proposed_step:04d}_i{island.island}_s{island_step:04d}_v{island.generation:04d}"
        mutation_path = f"islands/island_{island.island}/mutations/island_{island.island}_step_{island_step:04d}.yaml"
        self._write_job_event(
            context.run_dir,
            island.run_dir,
            {
                "job_id": job_id,
                "status": "submitted",
                "global_step_hint": proposed_step,
                "island": island.island,
                "island_step": island_step,
                "base_generation": int(island.generation),
                "base_graph_hash": base_graph_hash,
                "mutation_path": mutation_path,
                "worker_id": worker.worker_id,
                "device": worker.device,
            },
        )
        future = executor.submit(
            worker.run_candidate,
            base_graph=base_graph,
            best_result=island.best_train_result,
            rng_state=island.state.rng_state,
            island_step=island_step,
            output_dir=island.run_dir,
            train_inputs=train_inputs,
            train_y=train_y,
            validation_inputs=validation_inputs,
            validation_y=validation_y,
            errors=list(island.state.errors),
        )
        return PendingIslandCandidate(
            island=island,
            future=future,
            job_id=job_id,
            proposed_step=proposed_step,
            island_step=island_step,
            generation=int(island.generation),
            base_graph_hash=base_graph_hash,
        )

    def _commit_island_candidate(
        self,
        context: ProductionContext,
        islands: list[ProductionIslandContext],
        workers: dict[int, ProductionIslandWorker],
        job: PendingIslandCandidate,
        result: IslandCandidateResult,
        root_events: Any,
    ) -> None:
        island = job.island
        outcome = result.outcome
        island.state.rng_state = result.rng_state
        island.state.errors = list(result.errors[-40:])
        if int(island.generation) != int(job.generation):
            event = self._stale_event(context, island, job, result.document)
            self._commit_async_event(context, island, event, root_events, worker=workers[island.island])
            self._write_job_event(context.run_dir, island.run_dir, self._job_terminal_payload(job, "stale", global_step=int(event["step"])))
            return
        if outcome.failed:
            event = self._async_failed_event(context, island, job, result.document, outcome.error or "Unknown candidate failure.")
            self._commit_async_event(context, island, event, root_events, worker=workers[island.island])
            self._write_job_event(context.run_dir, island.run_dir, self._job_terminal_payload(job, "failed", global_step=int(event["step"]), error=str(event.get("error", ""))))
            return
        if (
            outcome.application is None
            or outcome.candidate_graph is None
            or outcome.result is None
            or result.validation_result is None
        ):
            event = self._async_failed_event(context, island, job, result.document, "Candidate evaluation returned an incomplete success outcome.")
            self._commit_async_event(context, island, event, root_events, worker=workers[island.island])
            self._write_job_event(context.run_dir, island.run_dir, self._job_terminal_payload(job, "failed", global_step=int(event["step"]), error=str(event.get("error", ""))))
            return

        candidate_train = outcome.result
        candidate_validation = result.validation_result
        previous_island_train_auc = float(island.best_train_result.auc)
        previous_island_validation_auc = float(island.best_validation_result.auc)
        previous_global_train_auc = float(context.best_train_result.auc)
        previous_global_validation_auc = float(context.best_validation_result.auc)
        accepted = self._promotes(candidate_train, candidate_validation, self._island_context(context, island))
        global_best = accepted and self._promotes(candidate_train, candidate_validation, context)
        if accepted:
            island.current_graph = outcome.candidate_graph
            island.best_graph = outcome.candidate_graph.clone()
            island.best_train_result = candidate_train
            island.best_validation_result = candidate_validation
            island.state.archive_version += 1
            island.state.best_train_auc = float(candidate_train.auc)
            island.state.best_validation_auc = float(candidate_validation.auc)
            island.state.best_config = dict(candidate_train.config)
            island.generation += 1
            island.state.generation = int(island.generation)
            island_context = self._island_context(context, island)
            island_metadata = {
                "island": island.island,
                "device": island.device,
                "worker_id": f"island_{island.island}",
                "global_step": int(context.state.step) + 1,
            }
            self._write_graph_artifacts(island_context, step=job.island_step, metadata=island_metadata)
            self._write_archive_entry(
                island_context,
                step=job.island_step,
                metadata={**island_metadata, "source": "candidate"},
            )
            self._write_checkpoint(island_context, step=job.island_step, metadata=island_metadata)
        if global_best:
            context.current_graph = outcome.candidate_graph
            context.best_graph = outcome.candidate_graph.clone()
            context.best_train_result = candidate_train
            context.best_validation_result = candidate_validation
            context.state.archive_version += 1
            context.state.best_train_auc = float(candidate_train.auc)
            context.state.best_validation_auc = float(candidate_validation.auc)
            context.state.best_config = dict(candidate_train.config)
            context.state.generation += 1
            global_metadata = {
                "island": island.island,
                "device": island.device,
                "worker_id": f"island_{island.island}",
                "island_step": job.island_step,
            }
            self._write_graph_artifacts(context, step=int(context.state.step) + 1, metadata=global_metadata)
            self._write_archive_entry(
                context,
                step=int(context.state.step) + 1,
                metadata={**global_metadata, "source": "global_best"},
            )
            self._write_checkpoint(context, step=int(context.state.step) + 1, metadata=global_metadata)

        event = self._async_candidate_event(
            context,
            island,
            job,
            accepted=accepted,
            global_best=global_best,
            mutation=result.document.to_dict(),
            train_result=candidate_train,
            validation_result=candidate_validation,
            maintenance=outcome.application.maintenance.to_dict(),
            previous_island_train_auc=previous_island_train_auc,
            previous_island_validation_auc=previous_island_validation_auc,
            previous_global_train_auc=previous_global_train_auc,
            previous_global_validation_auc=previous_global_validation_auc,
        )
        self._commit_async_event(context, island, event, root_events, worker=workers[island.island])
        self._write_job_event(context.run_dir, island.run_dir, self._job_terminal_payload(job, "completed", global_step=int(event["step"])))
        if global_best:
            self._migrate_global_best(context, islands, source_island=island.island)
        elif self._migration_interval(context) > 0 and int(context.state.step) % self._migration_interval(context) == 0:
            self._migrate_global_best(context, islands, source_island=island.island)

    def _commit_async_event(
        self,
        context: ProductionContext,
        island: ProductionIslandContext,
        event: dict[str, Any],
        root_events: Any,
        *,
        worker: ProductionIslandWorker | None = None,
    ) -> None:
        context.state.step = int(event["step"])
        island.state.step = int(event["island_step"])
        self._record_event(context.state, event)
        island_event = {
            **event,
            "best_train_auc": event.get("island_best_train_auc", event["best_train_auc"]),
            "best_validation_auc": event.get("island_best_validation_auc", event["best_validation_auc"]),
        }
        self._record_event(island.state, island_event)
        root_events.write(json.dumps(event) + "\n")
        root_events.flush()
        with (island.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        self._write_state(context.run_dir, context.state)
        self._write_state(island.run_dir, island.state)
        if worker is not None:
            worker.write_memorandum(self._island_context(context, island))
        else:
            island_loop = self._build_loop(island.current_graph, island.state.rng_state, island=island.island, device=island.device)
            self._write_memorandum(self._island_context(context, island), loop=island_loop, island=island.island)
        self._write_memorandum(context)

    def _migrate_global_best(
        self,
        context: ProductionContext,
        islands: list[ProductionIslandContext],
        *,
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
        previous_score = self._frontier_score(context, target.best_train_result, target.best_validation_result)
        target.current_graph = context.best_graph.clone()
        target.best_graph = context.best_graph.clone()
        target.best_train_result = context.best_train_result
        target.best_validation_result = context.best_validation_result
        target.state.archive_version += 1
        target.state.best_train_auc = float(context.best_train_result.auc)
        target.state.best_validation_auc = float(context.best_validation_result.auc)
        target.state.best_config = dict(context.state.best_config)
        target.generation += 1
        target.state.generation = int(target.generation)
        event = {
            "mode": "production_async_migration",
            "global_step": int(context.state.step),
            "source_island": int(source_island),
            "target_island": int(target.island),
            "target_device": target.device,
            "target_generation": int(target.generation),
            "global_best_version": int(context.state.archive_version),
            "selection_metric": self._frontier_metric(context),
            "previous_best_score": previous_score,
            "global_best_score": global_score,
            "previous_best_train_auc": float(target.best_train_result.auc),
            "previous_best_validation_auc": float(target.best_validation_result.auc),
            "global_best_validation_auc": float(context.best_validation_result.auc),
            "global_best_train_auc": float(context.best_train_result.auc),
            "graph_hash": graph_hash(context.best_graph),
        }
        with (context.run_dir / "migrations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        with (target.run_dir / "migrations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        target.state.history.append(
            f"- MIGRATED: global_step={int(context.state.step)} source_island={source_island} "
            f"{self._frontier_metric(context)}={global_score:.6f}"
        )
        del target.state.history[:-40]
        target_context = self._island_context(context, target)
        metadata = {
            "island": target.island,
            "device": target.device,
            "worker_id": f"island_{target.island}",
            "global_step": int(context.state.step),
            "source": "migration",
            "source_island": source_island,
            "selection_metric": self._frontier_metric(context),
        }
        self._write_graph_artifacts(target_context, step=int(target.state.step), metadata=metadata)
        self._write_archive_entry(target_context, step=int(target.state.step), metadata=metadata)
        self._write_checkpoint(target_context, step=int(target.state.step), metadata=metadata)
        self._write_state(target.run_dir, target.state)
        self._write_memorandum(target_context, loop=self._build_loop(target.current_graph, target.state.rng_state, island=target.island, device=target.device), island=target.island)

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

    def _build_island_worker(
        self,
        context: ProductionContext,
        island: ProductionIslandContext,
        task_context: str,
    ) -> ProductionIslandWorker:
        return ProductionIslandWorker(
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

    def _async_candidate_event(
        self,
        context: ProductionContext,
        island: ProductionIslandContext,
        job: PendingIslandCandidate,
        *,
        accepted: bool,
        global_best: bool,
        mutation: dict[str, object],
        train_result: EvaluationResult,
        validation_result: EvaluationResult,
        maintenance: dict[str, object],
        previous_island_train_auc: float,
        previous_island_validation_auc: float,
        previous_global_train_auc: float,
        previous_global_validation_auc: float,
    ) -> dict[str, Any]:
        step = int(context.state.step) + 1
        return {
            "step": step,
            "mode": "production_async_island",
            "job_id": job.job_id,
            "island": int(island.island),
            "island_step": int(job.island_step),
            "device": island.device,
            "worker_id": f"island_{island.island}",
            "base_graph_hash": job.base_graph_hash,
            "base_generation": int(job.generation),
            "generation": int(island.generation),
            "accepted": bool(accepted),
            "global_best": bool(global_best),
            "train_auc": float(train_result.auc),
            "validation_auc": float(validation_result.auc),
            "best_train_auc": float(context.state.best_train_auc),
            "best_validation_auc": float(context.state.best_validation_auc),
            "island_best_train_auc": float(island.state.best_train_auc),
            "island_best_validation_auc": float(island.state.best_validation_auc),
            "train_delta": float(train_result.auc) - previous_island_train_auc,
            "validation_delta": float(validation_result.auc) - previous_island_validation_auc,
            "global_train_delta": float(train_result.auc) - previous_global_train_auc,
            "global_validation_delta": float(validation_result.auc) - previous_global_validation_auc,
            "config": train_result.config,
            "mutation": mutation,
            "maintenance": maintenance,
            "promotion_policy": context.manifest["acceptance"],
        }

    def _async_failed_event(
        self,
        context: ProductionContext,
        island: ProductionIslandContext,
        job: PendingIslandCandidate,
        document: MutationDocument,
        error: str,
    ) -> dict[str, Any]:
        return {
            "step": int(context.state.step) + 1,
            "mode": "production_async_island",
            "job_id": job.job_id,
            "island": int(island.island),
            "island_step": int(job.island_step),
            "device": island.device,
            "worker_id": f"island_{island.island}",
            "base_graph_hash": job.base_graph_hash,
            "base_generation": int(job.generation),
            "generation": int(island.generation),
            "accepted": False,
            "failed": True,
            "error": error,
            "train_auc": None,
            "validation_auc": None,
            "best_train_auc": float(context.state.best_train_auc),
            "best_validation_auc": float(context.state.best_validation_auc),
            "island_best_train_auc": float(island.state.best_train_auc),
            "island_best_validation_auc": float(island.state.best_validation_auc),
            "config": island.state.best_config,
            "mutation": document.to_dict(),
        }

    def _stale_event(
        self,
        context: ProductionContext,
        island: ProductionIslandContext,
        job: PendingIslandCandidate,
        document: MutationDocument,
    ) -> dict[str, Any]:
        return {
            "step": int(context.state.step) + 1,
            "mode": "production_async_island",
            "job_id": job.job_id,
            "island": int(island.island),
            "island_step": int(job.island_step),
            "device": island.device,
            "worker_id": f"island_{island.island}",
            "base_graph_hash": job.base_graph_hash,
            "base_generation": int(job.generation),
            "generation": int(island.generation),
            "accepted": False,
            "stale": True,
            "stale_reason": "Island graph generation changed before candidate completion.",
            "train_auc": None,
            "validation_auc": None,
            "best_train_auc": float(context.state.best_train_auc),
            "best_validation_auc": float(context.state.best_validation_auc),
            "island_best_train_auc": float(island.state.best_train_auc),
            "island_best_validation_auc": float(island.state.best_validation_auc),
            "config": island.state.best_config,
            "mutation": document.to_dict(),
        }

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
