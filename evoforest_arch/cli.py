from __future__ import annotations

import argparse
import json
import pathlib

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.competition import COMPETITION_DATASET_NAME, COMPETITION_ROW_DATASET_NAME, DEFAULT_COMPETITION_DATA_DIR, competition_data_summary
from evoforest_arch.llm import (
    DEFAULT_ISLAND_TEMPERATURES,
    HTTPJSONLLMClient,
    LLMEngineerAgent,
    LLMScientistAgent,
    PromptBuilder,
)
from evoforest_arch.mutations import MutationEngine
from evoforest_arch.production import ProductionConfig, ProductionEvolutionRunner, export_best_graph, inspect_run, recheck_run
from evoforest_arch.seed import build_seed_graph
from evoforest_arch.synthetic import make_structural_break_data


def parse_temperature_schedule(value: str) -> tuple[float, ...]:
    text = value.strip()
    if text.lower() in {"none", "single"}:
        return ()
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("temperature schedule must be comma-separated floats or 'none'") from exc
    if not values:
        raise argparse.ArgumentTypeError("temperature schedule must include at least one value, or use 'none'")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evoforest-arch")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run a small architecture evolution demo.")
    demo.add_argument("--steps", type=int, default=12)
    demo.add_argument("--n-series", type=int, default=240)
    demo.add_argument("--length", type=int, default=160)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--folds", type=int, default=3)
    demo.add_argument("--max-configurations", type=int, default=64)
    demo.add_argument("--irls-steps", type=int, default=2)
    demo.add_argument("--refine-globals", action="store_true")
    demo.add_argument("--refine-steps", type=int, default=20)
    demo.add_argument("--refine-backend", choices=("auto", "numpy", "torch"), default="auto")
    demo.add_argument("--islands", type=int, default=1)
    demo.add_argument("--async-islands", action="store_true")
    demo.add_argument("--island-workers", type=int, default=None)
    demo.add_argument("--llm-provider", choices=("none", "http-json"), default="none")
    demo.add_argument("--llm-scientist-temperature", type=float, default=0.35)
    demo.add_argument(
        "--llm-island-temperatures",
        type=parse_temperature_schedule,
        default=DEFAULT_ISLAND_TEMPERATURES,
        help=(
            "Comma-separated scientist temperatures for island mode, or 'none' to use "
            "--llm-scientist-temperature for every island."
        ),
    )
    demo.add_argument("--llm-engineer-temperature", type=float, default=0.0)
    demo.add_argument("--task-context-file", type=pathlib.Path, default=None)
    demo.add_argument("--allow-source-mutations", action="store_true")
    demo.add_argument("--output", type=pathlib.Path, default=pathlib.Path("runs/demo"))
    demo.set_defaults(func=run_demo)

    evolve = sub.add_parser("evolve", help="Run resume-safe production evolution with fixed split manifests.")
    evolve.add_argument("--steps", type=int, default=4, help="New steps to run. With --resume, this is additional steps.")
    evolve.add_argument("--dataset", choices=("synthetic-structural-break", COMPETITION_DATASET_NAME, COMPETITION_ROW_DATASET_NAME), default="synthetic-structural-break")
    evolve.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_COMPETITION_DATA_DIR)
    evolve.add_argument("--n-series", type=int, default=240)
    evolve.add_argument("--length", type=int, default=160)
    evolve.add_argument("--boundary", type=int, default=None)
    evolve.add_argument("--competition-series-length", type=int, default=160)
    evolve.add_argument("--max-samples", type=int, default=None)
    evolve.add_argument("--max-ids", type=int, default=None)
    evolve.add_argument("--max-rows-per-id", type=int, default=None)
    evolve.add_argument("--row-stride", type=int, default=1)
    evolve.add_argument("--seed", type=int, default=17)
    evolve.add_argument("--split-seed", type=int, default=None)
    evolve.add_argument("--validation-fraction", type=float, default=0.2)
    evolve.add_argument("--test-fraction", type=float, default=0.2)
    evolve.add_argument("--folds", type=int, default=3)
    evolve.add_argument("--max-configurations", type=int, default=64)
    evolve.add_argument("--irls-steps", type=int, default=2)
    evolve.add_argument("--refine-globals", action="store_true")
    evolve.add_argument("--refine-steps", type=int, default=20)
    evolve.add_argument("--refine-backend", choices=("auto", "numpy", "torch"), default="auto")
    evolve.add_argument("--min-train-improvement", type=float, default=1e-6)
    evolve.add_argument("--min-validation-improvement", type=float, default=1e-6)
    evolve.add_argument("--allow-source-mutations", action="store_true")
    evolve.add_argument("--resume", action="store_true")
    evolve.add_argument("--output", type=pathlib.Path, required=True)
    evolve.set_defaults(func=run_evolve)

    inspect_parser = sub.add_parser("inspect", help="Inspect a production evolution run directory.")
    inspect_parser.add_argument("run_dir", type=pathlib.Path)
    inspect_parser.set_defaults(func=run_inspect)

    export_parser = sub.add_parser("export-best", help="Export the best graph from a production run.")
    export_parser.add_argument("run_dir", type=pathlib.Path)
    export_parser.add_argument("--output", type=pathlib.Path, required=True)
    export_parser.add_argument("--allow-source", action="store_true")
    export_parser.set_defaults(func=run_export_best)

    recheck_parser = sub.add_parser("recheck", help="Re-evaluate the best graph on fixed production splits.")
    recheck_parser.add_argument("run_dir", type=pathlib.Path)
    recheck_parser.add_argument("--include-test", action="store_true", help="Explicitly consume and report the held-out test split.")
    recheck_parser.add_argument("--allow-source", action="store_true")
    recheck_parser.set_defaults(func=run_recheck)

    data_parser = sub.add_parser("data-summary", help="Summarize the competition parquet event dataset without running evolution.")
    data_parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_COMPETITION_DATA_DIR)
    data_parser.add_argument("--competition-series-length", type=int, default=160)
    data_parser.add_argument("--max-samples", type=int, default=None)
    data_parser.add_argument("--row-level", action="store_true")
    data_parser.add_argument("--max-ids", type=int, default=None)
    data_parser.add_argument("--max-rows-per-id", type=int, default=None)
    data_parser.add_argument("--row-stride", type=int, default=1)
    data_parser.add_argument("--include-reduced-test", action="store_true")
    data_parser.set_defaults(func=run_data_summary)
    args = parser.parse_args(argv)
    return args.func(args)


def run_demo(args: argparse.Namespace) -> int:
    dataset = make_structural_break_data(n_series=args.n_series, length=args.length, seed=args.seed)
    graph = build_seed_graph()
    evaluator = RidgeEvaluator(
        n_splits=args.folds,
        seed=args.seed,
        max_configurations=args.max_configurations,
        irls_steps=args.irls_steps,
        refine_globals=args.refine_globals,
        refine_steps=args.refine_steps,
        refine_backend=args.refine_backend,
    )
    scientist = None
    engineer = None
    task_context = args.task_context_file.read_text(encoding="utf-8") if args.task_context_file is not None else ""
    mutation_engine = MutationEngine(allow_source=args.allow_source_mutations)
    if args.llm_provider == "http-json":
        client = HTTPJSONLLMClient.from_env()
        prompt_builder = PromptBuilder(allow_source=args.allow_source_mutations)
        scientist = LLMScientistAgent(
            client,
            prompt_builder=prompt_builder,
            temperature=args.llm_scientist_temperature,
            island_temperatures=args.llm_island_temperatures,
        )
        engineer = LLMEngineerAgent(
            client,
            prompt_builder=prompt_builder,
            temperature=args.llm_engineer_temperature,
            allow_source=args.allow_source_mutations,
        )
    loop = EvolutionLoop(
        graph,
        evaluator=evaluator,
        mutation_engine=mutation_engine,
        scientist=scientist,
        engineer=engineer,
        task_context=task_context,
        seed=args.seed,
    )
    if args.islands > 1 and args.async_islands:
        result = loop.run_async_islands(
            dataset.inputs(),
            dataset.y,
            islands=args.islands,
            steps_per_island=args.steps,
            output_dir=args.output,
            max_workers=args.island_workers,
        )
    elif args.islands > 1:
        result = loop.run_islands(dataset.inputs(), dataset.y, islands=args.islands, steps_per_island=args.steps, output_dir=args.output)
    else:
        result = loop.run(dataset.inputs(), dataset.y, steps=args.steps, output_dir=args.output)
    summary = {
        "auc": result.auc,
        "config": result.config,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def run_evolve(args: argparse.Namespace) -> int:
    config = ProductionConfig(
        output_dir=args.output,
        steps=args.steps,
        seed=args.seed,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        n_series=args.n_series,
        length=args.length,
        boundary=args.boundary,
        competition_series_length=args.competition_series_length,
        max_samples=args.max_samples,
        competition_max_ids=args.max_ids,
        competition_max_rows_per_id=args.max_rows_per_id,
        competition_row_stride=args.row_stride,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        folds=args.folds,
        max_configurations=args.max_configurations,
        irls_steps=args.irls_steps,
        refine_globals=args.refine_globals,
        refine_steps=args.refine_steps,
        refine_backend=args.refine_backend,
        min_train_improvement=args.min_train_improvement,
        min_validation_improvement=args.min_validation_improvement,
        allow_source_mutations=args.allow_source_mutations,
    )
    summary = ProductionEvolutionRunner(config).run(resume=args.resume)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def run_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_run(args.run_dir), indent=2), flush=True)
    return 0


def run_export_best(args: argparse.Namespace) -> int:
    output_path = export_best_graph(args.run_dir, args.output, allow_source=args.allow_source)
    print(json.dumps({"output": str(output_path)}, indent=2), flush=True)
    return 0


def run_recheck(args: argparse.Namespace) -> int:
    print(json.dumps(recheck_run(args.run_dir, include_test=args.include_test, allow_source=args.allow_source), indent=2), flush=True)
    return 0


def run_data_summary(args: argparse.Namespace) -> int:
    summary = competition_data_summary(
        args.data_dir,
        series_length=args.competition_series_length,
        max_samples=args.max_samples,
        include_reduced_test=args.include_reduced_test,
        row_level=args.row_level,
        max_ids=args.max_ids,
        max_rows_per_id=args.max_rows_per_id,
        row_stride=args.row_stride,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
