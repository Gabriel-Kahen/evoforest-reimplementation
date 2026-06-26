from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import pathlib
import subprocess
from typing import Any

import numpy as np

from evoforest_arch.agents import EngineerAgent, ScientistAgent
from evoforest_arch.evaluator import EvaluationResult, RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.feedback import feedback_summary, toon_report
from evoforest_arch.graph import Graph
from evoforest_arch.graph_io import graph_hash, graph_from_path, write_graph
from evoforest_arch.mutations import MutationEngine
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
    "Keep source-backed mutations disabled unless a trusted sandbox policy is supplied outside this package.",
)


@dataclass(frozen=True)
class ProductionConfig:
    output_dir: pathlib.Path
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
    min_train_improvement: float = 1e-6
    min_validation_improvement: float = 1e-6
    allow_source_mutations: bool = False

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
        self.graph = graph or build_seed_graph()
        self.evaluator = RidgeEvaluator(**config.evaluator_config())
        self.mutation_engine = MutationEngine(allow_source=config.allow_source_mutations)
        self.scientist = scientist
        self.engineer = engineer
        self.memorandum_agent = memorandum_agent
        self.task_context = task_context
        self.task_sources = task_sources

    def run(self, *, resume: bool = False) -> dict[str, Any]:
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
        return (
            train_delta > float(self.config.min_train_improvement)
            and validation_delta > float(self.config.min_validation_improvement)
        )

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
            "git_commit": _git_commit(),
            "dataset": self.config.dataset_config(),
            "dataset_metadata": dataset_metadata or {},
            "dataset_fingerprint": split_manifest.dataset_fingerprint,
            "split_manifest_path": "splits.json",
            "evaluator": self.config.evaluator_config(),
            "mutation": {"allow_source_mutations": bool(self.config.allow_source_mutations)},
            "acceptance": {
                "policy": (
                    "validation_improvement_with_train_regression_floor"
                    if self.config.min_train_improvement < 0.0
                    else "train_improvement_and_validation_improvement"
                ),
                "min_train_improvement": float(self.config.min_train_improvement),
                "min_validation_improvement": float(self.config.min_validation_improvement),
                "validation_config": "candidate_train_best_config",
            },
            "test_policy": (
                "test split is not evaluated by evolve; use recheck --include-test to consume it explicitly."
            ),
            "safe_staged_execution_rules": list(SAFE_STAGED_EXECUTION_RULES),
        }

    def _write_graph_artifacts(self, context: ProductionContext, step: int) -> None:
        metadata = {"run_id": context.state.run_id, "step": int(step)}
        write_graph(context.run_dir / context.state.current_graph_path, context.current_graph, metadata={**metadata, "role": "current"})
        write_graph(context.run_dir / context.state.best_graph_path, context.best_graph, metadata={**metadata, "role": "best"})

    def _write_archive_entry(self, context: ProductionContext, step: int) -> None:
        archive_dir = context.run_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        filename = f"best_v{context.state.archive_version:04d}_step_{step:04d}.json"
        payload = {
            "version": int(context.state.archive_version),
            "step": int(step),
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
        }
        with (archive_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _write_checkpoint(self, context: ProductionContext, step: int) -> None:
        payload = {
            "run_id": context.state.run_id,
            "step": int(step),
            "archive_version": int(context.state.archive_version),
            "current_graph_path": context.state.current_graph_path,
            "best_graph_path": context.state.best_graph_path,
            "best_train_result": context.best_train_result.to_dict(),
            "best_validation_result": context.best_validation_result.to_dict(),
            "feedback": feedback_summary(context.best_train_result),
            "diagnostics_toon": toon_report(context.best_train_result),
        }
        (context.run_dir / "checkpoint.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_memorandum(self, context: ProductionContext, loop: EvolutionLoop | None = None) -> None:
        if loop is not None and getattr(loop, "memorandum_agent", None) is not None:
            loop._write_memorandum(
                context.run_dir,
                context.best_train_result,
                context.state.history,
                context.state.errors,
                step=context.state.step,
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
    return {
        "run_id": state.run_id,
        "run_dir": str(path),
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
