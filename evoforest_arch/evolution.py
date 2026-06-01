from __future__ import annotations

import asyncio
import concurrent.futures
import copy
from dataclasses import dataclass, field
import inspect
import json
import pathlib

import numpy as np

from evoforest_arch.agents import EngineerAgent, ScientistAgent
from evoforest_arch.evaluator import EvaluationResult, RidgeEvaluator
from evoforest_arch.feedback import feedback_summary, toon_report
from evoforest_arch.graph import Graph
from evoforest_arch.mutations import MutationDocument, MutationEngine, MutationSpec
from evoforest_arch.task_context import build_task_context


DEFAULT_TASK_CONTEXT_PREFIX = "Clean-room EvoForest reimplementation"


@dataclass
class EvolutionEvent:
    step: int
    accepted: bool
    score: float | None
    best_score: float
    mutation: dict[str, object]
    config: dict[str, str]
    island: int | None = None
    global_best_score: float | None = None
    maintenance: dict[str, object] | None = None
    salvaged: list[str] | None = None
    failed: bool = False
    error: str | None = None
    mode: str = "single"
    round_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        row = {
            "step": self.step,
            "mode": self.mode,
            "accepted": self.accepted,
            "score": self.score,
            "best_score": self.best_score,
            "mutation": self.mutation,
            "config": self.config,
        }
        if self.island is not None:
            row["island"] = self.island
        if self.global_best_score is not None:
            row["global_best_score"] = self.global_best_score
        if self.maintenance is not None:
            row["maintenance"] = self.maintenance
        if self.salvaged is not None:
            row["salvaged"] = self.salvaged
        if self.failed:
            row["failed"] = True
        if self.error:
            row["error"] = self.error
        if self.round_index is not None:
            row["round"] = self.round_index
        return row


@dataclass
class CandidateOutcome:
    application: object | None = None
    candidate_graph: Graph | None = None
    result: EvaluationResult | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class IslandState:
    island: int
    current_graph: Graph
    best_graph: Graph
    best_result: EvaluationResult
    history: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class EvolutionLoop:
    def __init__(
        self,
        graph: Graph,
        evaluator: RidgeEvaluator | None = None,
        mutation_engine: MutationEngine | None = None,
        scientist: ScientistAgent | None = None,
        engineer: EngineerAgent | None = None,
        task_context: str = "",
        seed: int = 0,
    ) -> None:
        self.graph = graph
        self.evaluator = evaluator or RidgeEvaluator(seed=seed)
        self.mutation_engine = mutation_engine or MutationEngine()
        self.scientist = scientist or ScientistAgent()
        self.engineer = engineer or EngineerAgent()
        self.task_context = task_context
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        inputs: dict[str, object],
        y: np.ndarray,
        steps: int,
        output_dir: pathlib.Path,
    ) -> EvaluationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_task_context(output_dir, inputs, y)
        self._reset_archive(output_dir)
        events_path = output_dir / "events.jsonl"
        current_graph = self.graph.clone()
        best_result = self.evaluator.evaluate(current_graph, inputs, y, update_graph=True)
        best_graph = current_graph.clone()
        self._write_checkpoint(output_dir, best_graph, best_result, step=0)
        best_version = 0
        self._write_archive_entry(output_dir, best_graph, best_result, step=0, version=best_version, mode="single")
        history: list[str] = []
        errors: list[str] = []
        self._write_memorandum(output_dir, best_result, history, errors)
        with events_path.open("w", encoding="utf-8") as events:
            for step in range(1, steps + 1):
                memorandum = self._read_memorandum(output_dir)
                document = self._propose_document(
                    current_graph,
                    best_result,
                    step,
                    memorandum=memorandum,
                    execution_errors=self._error_context(errors),
                )
                self._write_prompt_records(output_dir, step)
                self._write_mutation_document(output_dir, step, document)
                outcome = self._try_evaluate_candidate(current_graph, document, inputs, y)
                if outcome.failed:
                    event = self._failed_event(
                        step=step,
                        mode="single",
                        document=document,
                        best_result=best_result,
                        error=outcome.error or "Unknown candidate failure.",
                    )
                    events.write(json.dumps(event.to_dict()) + "\n")
                    self._record_event(history, errors, event)
                    self._write_memorandum(output_dir, best_result, history, errors)
                    continue
                application = outcome.application
                candidate_graph = outcome.candidate_graph
                result = outcome.result
                if application is None or candidate_graph is None or result is None:
                    raise RuntimeError("Candidate evaluation returned an incomplete success outcome.")
                accepted = result.auc >= best_result.auc
                salvaged: list[str] = []
                if accepted:
                    current_graph = candidate_graph
                    best_graph = candidate_graph.clone()
                    best_result = result
                    self._write_checkpoint(output_dir, best_graph, best_result, step=step)
                    best_version += 1
                    self._write_archive_entry(output_dir, best_graph, best_result, step=step, version=best_version, mode="single")
                else:
                    current_graph, best_graph, best_result, salvaged = self._salvage(
                        current_graph,
                        best_graph,
                        best_result,
                        candidate_graph,
                        document,
                        inputs,
                        y,
                    )
                    if salvaged:
                        self._write_checkpoint(output_dir, best_graph, best_result, step=step)
                        best_version += 1
                        self._write_archive_entry(output_dir, best_graph, best_result, step=step, version=best_version, mode="single")
                event = EvolutionEvent(
                    step=step,
                    mode="single",
                    accepted=accepted,
                    score=float(result.auc),
                    best_score=float(best_result.auc),
                    mutation=document.to_dict(),
                    config=result.config,
                    maintenance=application.maintenance.to_dict(),
                    salvaged=salvaged,
                )
                events.write(json.dumps(event.to_dict()) + "\n")
                self._record_event(history, errors, event)
                self._write_memorandum(output_dir, best_result, history, errors)
        self._write_memorandum(output_dir, best_result, history, errors)
        return best_result

    def run_islands(
        self,
        inputs: dict[str, object],
        y: np.ndarray,
        islands: int,
        steps_per_island: int,
        output_dir: pathlib.Path,
        migration_interval: int = 10,
    ) -> EvaluationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        task_context = self._prepare_task_context(output_dir, inputs, y)
        self._reset_archive(output_dir)
        events_path = output_dir / "events.jsonl"
        islands = max(1, int(islands))
        steps_per_island = max(0, int(steps_per_island))
        states: list[IslandState] = []
        for island_id in range(islands):
            island_graph = self.graph.clone()
            result = self.evaluator.evaluate(island_graph, inputs, y, update_graph=True)
            island_dir = output_dir / f"island_{island_id}"
            island_dir.mkdir(parents=True, exist_ok=True)
            self._write_task_context(island_dir, task_context)
            self._write_checkpoint(island_dir, island_graph, result, step=0, island=island_id)
            state = IslandState(island_id, island_graph, island_graph.clone(), result)
            self._write_memorandum(island_dir, result, state.history, state.errors)
            states.append(state)

        global_best_state = max(states, key=lambda state: state.best_result.auc)
        global_best_graph = global_best_state.best_graph.clone()
        global_best_result = global_best_state.best_result
        self._write_checkpoint(output_dir, global_best_graph, global_best_result, step=0, island=global_best_state.island)
        global_best_version = 0
        self._write_archive_entry(
            output_dir,
            global_best_graph,
            global_best_result,
            step=0,
            version=global_best_version,
            mode="sequential_island",
            island=global_best_state.island,
        )
        global_history: list[str] = []
        global_errors: list[str] = []
        self._write_memorandum(output_dir, global_best_result, global_history, global_errors)

        total_steps = islands * steps_per_island
        with events_path.open("w", encoding="utf-8") as events:
            for global_step in range(1, total_steps + 1):
                state = states[(global_step - 1) % islands]
                island_dir = output_dir / f"island_{state.island}"
                memorandum = self._read_memorandum(island_dir)
                document = self._propose_document(
                    state.current_graph,
                    state.best_result,
                    global_step,
                    island=state.island,
                    memorandum=memorandum,
                    execution_errors=self._error_context(state.errors),
                )
                self._write_prompt_records(output_dir, global_step, island=state.island)
                self._write_mutation_document(output_dir, global_step, document, island=state.island)
                outcome = self._try_evaluate_candidate(state.current_graph, document, inputs, y)
                if outcome.failed:
                    event = self._failed_event(
                        step=global_step,
                        mode="sequential_island",
                        document=document,
                        best_result=state.best_result,
                        error=outcome.error or "Unknown candidate failure.",
                        island=state.island,
                        global_best_score=float(global_best_result.auc),
                    )
                    events.write(json.dumps(event.to_dict()) + "\n")
                    self._record_event(state.history, state.errors, event)
                    self._record_event(global_history, global_errors, event)
                    self._write_memorandum(output_dir / f"island_{state.island}", state.best_result, state.history, state.errors)
                    self._write_memorandum(output_dir, global_best_result, global_history, global_errors)
                    continue
                application = outcome.application
                candidate_graph = outcome.candidate_graph
                result = outcome.result
                if application is None or candidate_graph is None or result is None:
                    raise RuntimeError("Candidate evaluation returned an incomplete success outcome.")
                accepted = result.auc >= state.best_result.auc
                salvaged: list[str] = []
                if accepted:
                    state.current_graph = candidate_graph
                    state.best_graph = candidate_graph.clone()
                    state.best_result = result
                    self._write_checkpoint(
                        output_dir / f"island_{state.island}",
                        state.best_graph,
                        state.best_result,
                        step=global_step,
                        island=state.island,
                    )
                else:
                    state.current_graph, state.best_graph, state.best_result, salvaged = self._salvage(
                        state.current_graph,
                        state.best_graph,
                        state.best_result,
                        candidate_graph,
                        document,
                        inputs,
                        y,
                    )
                    if salvaged:
                        self._write_checkpoint(
                            output_dir / f"island_{state.island}",
                            state.best_graph,
                            state.best_result,
                            step=global_step,
                            island=state.island,
                        )
                if state.best_result.auc >= global_best_result.auc:
                    global_best_graph = state.best_graph.clone()
                    global_best_result = state.best_result
                    self._write_checkpoint(output_dir, global_best_graph, global_best_result, step=global_step, island=state.island)
                    global_best_version += 1
                    self._write_archive_entry(
                        output_dir,
                        global_best_graph,
                        global_best_result,
                        step=global_step,
                        version=global_best_version,
                        mode="sequential_island",
                        island=state.island,
                    )
                if migration_interval > 0 and global_step % migration_interval == 0:
                    weakest = min(states, key=lambda item: item.best_result.auc)
                    if weakest.best_result.auc < global_best_result.auc:
                        weakest.current_graph = global_best_graph.clone()
                        weakest.best_graph = global_best_graph.clone()
                        weakest.best_result = global_best_result

                event = EvolutionEvent(
                    step=global_step,
                    mode="sequential_island",
                    island=state.island,
                    accepted=accepted,
                    score=float(result.auc),
                    best_score=float(state.best_result.auc),
                    global_best_score=float(global_best_result.auc),
                    mutation=document.to_dict(),
                    config=result.config,
                    maintenance=application.maintenance.to_dict(),
                    salvaged=salvaged,
                )
                events.write(json.dumps(event.to_dict()) + "\n")
                self._record_event(state.history, state.errors, event)
                self._record_event(global_history, global_errors, event)
                self._write_memorandum(output_dir / f"island_{state.island}", state.best_result, state.history, state.errors)
                self._write_memorandum(output_dir, global_best_result, global_history, global_errors)

        for state in states:
            self._write_memorandum(output_dir / f"island_{state.island}", state.best_result, state.history, state.errors)
        self._write_memorandum(output_dir, global_best_result, global_history, global_errors)
        return global_best_result

    def run_async_islands(
        self,
        inputs: dict[str, object],
        y: np.ndarray,
        islands: int,
        steps_per_island: int,
        output_dir: pathlib.Path,
        migration_interval: int = 10,
        max_workers: int | None = None,
    ) -> EvaluationResult:
        return asyncio.run(
            self._run_async_islands(
                inputs=inputs,
                y=y,
                islands=islands,
                steps_per_island=steps_per_island,
                output_dir=output_dir,
                migration_interval=migration_interval,
                max_workers=max_workers,
            )
        )

    async def _run_async_islands(
        self,
        inputs: dict[str, object],
        y: np.ndarray,
        islands: int,
        steps_per_island: int,
        output_dir: pathlib.Path,
        migration_interval: int,
        max_workers: int | None,
    ) -> EvaluationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        task_context = self._prepare_task_context(output_dir, inputs, y)
        self._reset_archive(output_dir)
        events_path = output_dir / "events.jsonl"
        islands = max(1, int(islands))
        steps_per_island = max(0, int(steps_per_island))
        states = self._initialize_island_states(inputs, y, islands, output_dir)
        for state in states:
            self._write_task_context(output_dir / f"island_{state.island}", task_context)
        global_best_state = max(states, key=lambda state: state.best_result.auc)
        global_best_graph = global_best_state.best_graph.clone()
        global_best_result = global_best_state.best_result
        self._write_checkpoint(output_dir, global_best_graph, global_best_result, step=0, island=global_best_state.island)
        global_best_version = 0
        self._write_archive_entry(
            output_dir,
            global_best_graph,
            global_best_result,
            step=0,
            version=global_best_version,
            mode="async_island",
            island=global_best_state.island,
        )
        global_history: list[str] = []
        global_errors: list[str] = []
        self._write_memorandum(output_dir, global_best_result, global_history, global_errors)

        global_step = 0
        worker_count = max_workers or islands
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
            with events_path.open("w", encoding="utf-8") as events:
                for round_index in range(1, steps_per_island + 1):
                    tasks = []
                    for state in states:
                        next_step = global_step + len(tasks) + 1
                        island_dir = output_dir / f"island_{state.island}"
                        memorandum = self._read_memorandum(island_dir)
                        document = self._propose_document(
                            state.current_graph,
                            state.best_result,
                            next_step,
                            island=state.island,
                            memorandum=memorandum,
                            execution_errors=self._error_context(state.errors),
                        )
                        self._write_prompt_records(output_dir, next_step, island=state.island)
                        self._write_mutation_document(output_dir, next_step, document, island=state.island)
                        base_graph = state.current_graph.clone()
                        task = asyncio.get_running_loop().run_in_executor(
                            executor,
                            self._try_evaluate_candidate,
                            base_graph,
                            document,
                            inputs,
                            y,
                        )
                        tasks.append(asyncio.create_task(self._with_candidate_metadata(state, next_step, document, task)))

                    for future in asyncio.as_completed(tasks):
                        state, proposed_step, document, outcome = await future
                        global_step += 1
                        event_step = global_step
                        if outcome.failed:
                            event = self._failed_event(
                                step=event_step,
                                mode="async_island",
                                document=document,
                                best_result=state.best_result,
                                error=outcome.error or "Unknown candidate failure.",
                                island=state.island,
                                global_best_score=float(global_best_result.auc),
                                round_index=round_index,
                            )
                            event_row = event.to_dict()
                            event_row["proposed_step"] = proposed_step
                            events.write(json.dumps(event_row) + "\n")
                            self._record_event(state.history, state.errors, event)
                            self._record_event(global_history, global_errors, event)
                            self._write_memorandum(output_dir / f"island_{state.island}", state.best_result, state.history, state.errors)
                            self._write_memorandum(output_dir, global_best_result, global_history, global_errors)
                            continue
                        application = outcome.application
                        candidate_graph = outcome.candidate_graph
                        result = outcome.result
                        if application is None or candidate_graph is None or result is None:
                            raise RuntimeError("Candidate evaluation returned an incomplete success outcome.")
                        accepted = result.auc >= state.best_result.auc
                        salvaged: list[str] = []
                        if accepted:
                            state.current_graph = candidate_graph
                            state.best_graph = candidate_graph.clone()
                            state.best_result = result
                            self._write_checkpoint(
                                output_dir / f"island_{state.island}",
                                state.best_graph,
                                state.best_result,
                                step=event_step,
                                island=state.island,
                            )
                        else:
                            state.current_graph, state.best_graph, state.best_result, salvaged = self._salvage(
                                state.current_graph,
                                state.best_graph,
                                state.best_result,
                                candidate_graph,
                                document,
                                inputs,
                                y,
                            )
                            if salvaged:
                                self._write_checkpoint(
                                    output_dir / f"island_{state.island}",
                                    state.best_graph,
                                    state.best_result,
                                    step=event_step,
                                    island=state.island,
                                )
                        if state.best_result.auc >= global_best_result.auc:
                            global_best_graph = state.best_graph.clone()
                            global_best_result = state.best_result
                            self._write_checkpoint(output_dir, global_best_graph, global_best_result, step=event_step, island=state.island)
                            global_best_version += 1
                            self._write_archive_entry(
                                output_dir,
                                global_best_graph,
                                global_best_result,
                                step=event_step,
                                version=global_best_version,
                                mode="async_island",
                                island=state.island,
                            )
                        if migration_interval > 0 and event_step % migration_interval == 0:
                            global_best_graph, global_best_result = self._migrate_global_best(states, global_best_graph, global_best_result)

                        event = EvolutionEvent(
                            step=event_step,
                            mode="async_island",
                            round_index=round_index,
                            island=state.island,
                            accepted=accepted,
                            score=float(result.auc),
                            best_score=float(state.best_result.auc),
                            global_best_score=float(global_best_result.auc),
                            mutation=document.to_dict(),
                            config=result.config,
                            maintenance=application.maintenance.to_dict(),
                            salvaged=salvaged,
                        )
                        event_row = event.to_dict()
                        event_row["proposed_step"] = proposed_step
                        events.write(json.dumps(event_row) + "\n")
                        self._record_event(state.history, state.errors, event)
                        self._record_event(global_history, global_errors, event)
                        self._write_memorandum(output_dir / f"island_{state.island}", state.best_result, state.history, state.errors)
                        self._write_memorandum(output_dir, global_best_result, global_history, global_errors)

        for state in states:
            self._write_memorandum(output_dir / f"island_{state.island}", state.best_result, state.history, state.errors)
        self._write_memorandum(output_dir, global_best_result, global_history, global_errors)
        return global_best_result

    def _propose(self, step: int) -> MutationSpec:
        template = self.engineer.templates[0]
        return MutationSpec(
            kind=template.kind,
            target_node=template.target_node,
            primitive=template.primitive,
            alternative_id=f"{template.alternative_id}_{step}",
            parents=template.parents,
            description=template.description,
        )

    def _propose_document(
        self,
        graph: Graph,
        result: EvaluationResult,
        step: int,
        island: int | None = None,
        memorandum: str = "",
        execution_errors: str = "",
    ) -> MutationDocument:
        hypothesis_kwargs = self._supported_kwargs(
            self.scientist.generate,
            {
                "step": step,
                "island": island,
                "memorandum": memorandum,
            },
        )
        hypotheses = self.scientist.generate(graph, result, **hypothesis_kwargs)
        engineer_kwargs = self._supported_kwargs(
            self.engineer.synthesize,
            {
                "memorandum": memorandum,
                "execution_errors": execution_errors,
            },
        )
        return self.engineer.synthesize(graph, result, hypotheses, step, island, self.rng, **engineer_kwargs)

    def _initialize_island_states(self, inputs: dict[str, object], y: np.ndarray, islands: int, output_dir: pathlib.Path) -> list[IslandState]:
        states: list[IslandState] = []
        for island_id in range(islands):
            island_graph = self.graph.clone()
            result = self.evaluator.evaluate(island_graph, inputs, y, update_graph=True)
            island_dir = output_dir / f"island_{island_id}"
            island_dir.mkdir(parents=True, exist_ok=True)
            self._write_checkpoint(island_dir, island_graph, result, step=0, island=island_id)
            state = IslandState(island_id, island_graph, island_graph.clone(), result)
            self._write_memorandum(island_dir, result, state.history, state.errors)
            states.append(state)
        return states

    def _try_evaluate_candidate(
        self,
        base_graph: Graph,
        document: MutationDocument,
        inputs: dict[str, object],
        y: np.ndarray,
    ) -> CandidateOutcome:
        try:
            application = self.mutation_engine.apply_document(base_graph, document)
            candidate_graph = application.graph
            result = self.evaluator.evaluate(candidate_graph, inputs, y, update_graph=True)
            return CandidateOutcome(application=application, candidate_graph=candidate_graph, result=result)
        except Exception as exc:
            return CandidateOutcome(error=self._format_exception(exc))

    @staticmethod
    async def _with_candidate_metadata(
        state: IslandState,
        proposed_step: int,
        document: MutationDocument,
        task: asyncio.Future[CandidateOutcome],
    ) -> tuple[IslandState, int, MutationDocument, CandidateOutcome]:
        outcome = await task
        return state, proposed_step, document, outcome

    @staticmethod
    def _migrate_global_best(
        states: list[IslandState],
        global_best_graph: Graph,
        global_best_result: EvaluationResult,
    ) -> tuple[Graph, EvaluationResult]:
        weakest = min(states, key=lambda item: item.best_result.auc)
        if weakest.best_result.auc < global_best_result.auc:
            weakest.current_graph = global_best_graph.clone()
            weakest.best_graph = global_best_graph.clone()
            weakest.best_result = global_best_result
        return global_best_graph, global_best_result

    def _salvage(
        self,
        current_graph: Graph,
        best_graph: Graph,
        best_result: EvaluationResult,
        candidate_graph: Graph,
        document: MutationDocument,
        inputs: dict[str, object],
        y: np.ndarray,
    ) -> tuple[Graph, Graph, EvaluationResult, list[str]]:
        salvaged: list[str] = []
        incumbent = current_graph
        incumbent_best_graph = best_graph
        incumbent_best_result = best_result
        for spec in document.add:
            if spec.target_node not in candidate_graph.nodes or spec.target_node not in incumbent.nodes:
                continue
            if any(existing.id == spec.alternative_id for existing in incumbent.nodes[spec.target_node].alternatives):
                continue
            candidate_alt = next(
                (alternative for alternative in candidate_graph.nodes[spec.target_node].alternatives if alternative.id == spec.alternative_id),
                None,
            )
            if candidate_alt is None:
                continue
            trial = incumbent.clone()
            for global_spec in document.globals:
                if global_spec.name not in trial.globals.names():
                    trial.globals.add(
                        global_spec.name,
                        global_spec.value,
                        trainable=global_spec.trainable,
                        description=global_spec.description,
                    )
            trial.nodes[spec.target_node].add_alternative(copy.deepcopy(candidate_alt))
            trial, _report = self.mutation_engine.maintenance.clean(trial)
            result = self.evaluator.evaluate(trial, inputs, y, update_graph=True)
            if result.auc >= incumbent_best_result.auc:
                incumbent = trial
                incumbent_best_graph = trial.clone()
                incumbent_best_result = result
                salvaged.append(f"{spec.target_node}.{spec.alternative_id}")
        return incumbent, incumbent_best_graph, incumbent_best_result, salvaged

    @staticmethod
    def _write_checkpoint(output_dir: pathlib.Path, graph: Graph, result: EvaluationResult, step: int, island: int | None = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "step": step,
            "island": island,
            "graph": graph.to_dict(),
            "result": result.to_dict(),
            "feedback": feedback_summary(result),
            "diagnostics_toon": toon_report(result),
        }
        (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    @staticmethod
    def _reset_archive(output_dir: pathlib.Path) -> None:
        archive_dir = output_dir / "archive"
        if not archive_dir.exists():
            return
        for path in archive_dir.glob("global_best_v*.json"):
            path.unlink()
        index_path = archive_dir / "index.jsonl"
        if index_path.exists():
            index_path.unlink()

    @staticmethod
    def _write_archive_entry(
        output_dir: pathlib.Path,
        graph: Graph,
        result: EvaluationResult,
        *,
        step: int,
        version: int,
        mode: str,
        island: int | None = None,
    ) -> None:
        archive_dir = output_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        island_part = "single" if island is None else f"island_{island}"
        filename = f"global_best_v{version:04d}_step_{step:04d}_{island_part}.json"
        entry = {
            "version": version,
            "step": step,
            "mode": mode,
            "island": island,
            "auc": float(result.auc),
            "config": result.config,
            "graph": graph.to_dict(),
            "result": result.to_dict(),
            "feedback": feedback_summary(result),
            "diagnostics_toon": toon_report(result),
        }
        (archive_dir / filename).write_text(json.dumps(entry, indent=2), encoding="utf-8")
        index_row = {
            "version": version,
            "step": step,
            "mode": mode,
            "island": island,
            "auc": float(result.auc),
            "config": result.config,
            "path": filename,
        }
        with (archive_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(index_row) + "\n")

    def _prepare_task_context(self, output_dir: pathlib.Path, inputs: dict[str, object], y: np.ndarray) -> str:
        context = self.task_context.strip()
        if not context:
            context = build_task_context(inputs, y, self.evaluator, source="runtime inputs").to_text()
        self._write_task_context(output_dir, context)
        self._install_task_context(context)
        return context

    @staticmethod
    def _write_task_context(output_dir: pathlib.Path, context: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "task_context.md").write_text(context if context.endswith("\n") else context + "\n", encoding="utf-8")

    def _install_task_context(self, context: str) -> None:
        for agent in (self.scientist, self.engineer):
            prompt_builder = getattr(agent, "prompt_builder", None)
            if prompt_builder is None or not hasattr(prompt_builder, "task_context"):
                continue
            current = str(getattr(prompt_builder, "task_context", ""))
            if self.task_context or current.startswith(DEFAULT_TASK_CONTEXT_PREFIX):
                prompt_builder.task_context = context

    @staticmethod
    def _write_memorandum(
        output_dir: pathlib.Path,
        result: EvaluationResult,
        history: list[str] | None = None,
        error_log: list[str] | None = None,
    ) -> None:
        feedback = feedback_summary(result)
        lines = [
            "# Evolution Memorandum",
            "",
            "[OUTCOME HISTORY]",
        ]
        lines.extend((history or [])[-8:] or ["- No mutation outcomes recorded yet."])
        lines.extend(["", "[STATE]"])
        lines.extend(EvolutionLoop._memorandum_state_lines(result, feedback))
        lines.extend(["", "[WHAT WORKS]"])
        lines.extend(EvolutionLoop._memorandum_works(history or [], feedback))
        lines.extend(["", "[WHAT FAILED]"])
        lines.extend(EvolutionLoop._memorandum_failed(history or [], feedback))
        lines.extend(["", "[ERROR LOG]"])
        lines.extend((error_log or [])[-8:] or ["- No runtime errors recorded."])
        lines.extend(["", "TOON diagnostics:", "```", toon_report(result), "```"])
        (output_dir / "memorandum.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _memorandum_state_lines(result: EvaluationResult, feedback: dict[str, object]) -> list[str]:
        scoring = feedback.get("scoring_context", {})
        search = feedback.get("configuration_search", {})
        cache = search.get("cache", {}) if isinstance(search, dict) else {}
        features = feedback.get("top_features", [])
        subnodes = feedback.get("top_subnodes", [])
        alternatives = feedback.get("top_alternatives", [])
        lines = [
            f"- Best AUC: {result.auc:.6f}; config={result.config}.",
        ]
        if isinstance(search, dict):
            lines.append(
                "- Configuration search: "
                f"{int(search.get('evaluated', 1))}/{int(search.get('total', 1))} evaluated, "
                f"capped={bool(search.get('capped', False))}."
            )
        if isinstance(scoring, dict):
            lines.append(
                "- Representation: "
                f"effective_rank={float(scoring.get('effective_rank', 0.0)):.4f}, "
                f"mean_max_corr={float(scoring.get('mean_max_corr', 0.0)):.4f}, "
                f"global_ridge_auc={float(scoring.get('global_ridge_auc', 0.0)):.6f}."
            )
        if isinstance(cache, dict):
            lines.append(
                "- Cache: "
                f"hits={int(cache.get('hits', 0))}, entries={int(cache.get('entries', 0))}, "
                f"key={cache.get('key', 'unknown')}."
            )
        if isinstance(features, list) and features:
            top = features[0]
            if isinstance(top, dict):
                lines.append(
                    "- Dominant feature: "
                    f"{top.get('name', '')} imp={float(top.get('importance', 0.0)):.4f}, "
                    f"ind_auc={float(top.get('individual_auc', 0.0)):.4f}, "
                    f"resid={float(top.get('residual_corr', 0.0)):.4f}."
                )
        if isinstance(subnodes, list) and subnodes:
            top = subnodes[0]
            if isinstance(top, dict):
                lines.append(
                    "- Dominant subnode: "
                    f"{top.get('name', '')} imp={float(top.get('importance', 0.0)):.4f}, "
                    f"features={int(top.get('feature_count', 0))}."
                )
        if isinstance(alternatives, list) and alternatives:
            top = alternatives[0]
            if isinstance(top, dict):
                lines.append(
                    "- Dominant alternative: "
                    f"{top.get('name', '')} age={int(top.get('age', 0))}, "
                    f"participations={int(top.get('participation_count', 0))}."
                )
        return lines

    @staticmethod
    def _memorandum_works(history: list[str], feedback: dict[str, object]) -> list[str]:
        lines = [line for line in history if "ACCEPTED" in line or "SALVAGED" in line][-4:]
        subnodes = feedback.get("top_subnodes", [])
        subnode_rows = subnodes[:3] if isinstance(subnodes, list) else []
        for row in subnode_rows:
            if isinstance(row, dict) and float(row.get("importance", 0.0)) > 0.0:
                lines.append(
                    f"- Productive substructure: {row.get('name', '')} "
                    f"aggregates imp={float(row.get('importance', 0.0)):.4f}."
                )
        return lines or ["- No accepted or salvaged mutation patterns recorded yet."]

    @staticmethod
    def _memorandum_failed(history: list[str], feedback: dict[str, object]) -> list[str]:
        lines = [line for line in history if "REJECTED" in line or "FAILED" in line][-4:]
        risky = feedback.get("risky_features", [])
        risky_rows = risky[:4] if isinstance(risky, list) else []
        for row in risky_rows:
            if isinstance(row, dict):
                lines.append(
                    f"- Risky feature: {row.get('name', '')} "
                    f"redundancy={float(row.get('redundancy', 0.0)):.4f}, "
                    f"stability={float(row.get('weight_stability', 0.0)):.4f}."
                )
        return lines or ["- No rejected, failed, redundant, or unstable patterns recorded yet."]

    @staticmethod
    def _write_mutation_document(output_dir: pathlib.Path, step: int, document: MutationDocument, island: int | None = None) -> None:
        mutation_dir = output_dir / "mutations"
        mutation_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"island_{island}_" if island is not None else ""
        (mutation_dir / f"{prefix}step_{step:04d}.yaml").write_text(document.to_yaml(), encoding="utf-8")

    @staticmethod
    def _read_memorandum(output_dir: pathlib.Path) -> str:
        path = output_dir / "memorandum.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_prompt_records(self, output_dir: pathlib.Path, step: int, island: int | None = None) -> None:
        records = []
        for source in (self.scientist, self.engineer):
            drain = getattr(source, "pop_prompt_records", None)
            if callable(drain):
                records.extend(drain())
        if not records:
            return
        prompt_dir = output_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"island_{island}_" if island is not None else ""
        for index, record in enumerate(records):
            stage = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(record.stage))
            path = prompt_dir / f"{prefix}step_{step:04d}_{index:02d}_{stage}.md"
            path.write_text(record.to_text(), encoding="utf-8")

    @staticmethod
    def _supported_kwargs(func: object, kwargs: dict[str, object]) -> dict[str, object]:
        parameters = inspect.signature(func).parameters
        return {key: value for key, value in kwargs.items() if key in parameters}

    @staticmethod
    def _failed_event(
        step: int,
        mode: str,
        document: MutationDocument,
        best_result: EvaluationResult,
        error: str,
        island: int | None = None,
        global_best_score: float | None = None,
        round_index: int | None = None,
    ) -> EvolutionEvent:
        return EvolutionEvent(
            step=step,
            mode=mode,
            island=island,
            round_index=round_index,
            accepted=False,
            failed=True,
            error=error,
            score=None,
            best_score=float(best_result.auc),
            global_best_score=global_best_score,
            mutation=document.to_dict(),
            config=best_result.config,
            maintenance=None,
            salvaged=[],
        )

    @staticmethod
    def _record_event(history: list[str], errors: list[str], event: EvolutionEvent) -> None:
        history.append(EvolutionLoop._format_history_event(event))
        if event.failed and event.error:
            errors.append(EvolutionLoop._format_error_event(event))
            del errors[:-20]

    @staticmethod
    def _format_history_event(event: EvolutionEvent) -> str:
        status = "FAILED" if event.failed else "ACCEPTED" if event.accepted else "REJECTED"
        if event.salvaged:
            status = f"{status}+SALVAGED"
        location = f" island={event.island}" if event.island is not None else ""
        score = "n/a" if event.score is None else f"{event.score:.6f}"
        suffix = f" error={event.error}" if event.failed and event.error else ""
        return f"- {status}: step={event.step}{location} score={score} best={event.best_score:.6f}{suffix}"

    @staticmethod
    def _format_error_event(event: EvolutionEvent) -> str:
        location = f" island={event.island}" if event.island is not None else ""
        return f"- step={event.step}{location}: {event.error}"

    @staticmethod
    def _error_context(errors: list[str]) -> str:
        return "\n".join(errors[-8:])

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"
