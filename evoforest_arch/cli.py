from __future__ import annotations

import argparse
import json
import os
import pathlib

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.llm import (
    DEFAULT_ISLAND_TEMPERATURES,
    LLMEngineerAgent,
    LLMMemorandumAgent,
    LLMScientistAgent,
    PromptBuilder,
    SUPPORTED_LLM_PROVIDERS,
    llm_client_from_env,
    llm_provider_from_env,
    load_env_file,
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
    demo.add_argument("--refine-globals", dest="refine_globals", action="store_true", default=True)
    demo.add_argument("--no-refine-globals", dest="refine_globals", action="store_false")
    demo.add_argument("--refine-steps", type=int, default=20)
    demo.add_argument("--refine-backend", choices=("auto", "numpy", "torch"), default="auto")
    demo.add_argument("--islands", type=int, default=1)
    demo.add_argument("--async-islands", action="store_true")
    demo.add_argument("--island-workers", type=int, default=None)
    llm_provider_choices = ("none", "env", *SUPPORTED_LLM_PROVIDERS)
    demo.add_argument("--llm-provider", choices=llm_provider_choices, default="none")
    demo.add_argument("--env-file", type=pathlib.Path, default=pathlib.Path(".env"))
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
    demo.add_argument("--task-source-file", action="append", type=pathlib.Path, default=[])
    demo.add_argument("--allow-source-mutations", action="store_true")
    demo.add_argument("--output", type=pathlib.Path, default=pathlib.Path("runs/demo"))
    demo.set_defaults(func=run_demo)

    evolve = sub.add_parser("evolve", help="Run resume-safe production evolution with fixed split manifests.")
    evolve.add_argument("--steps", type=int, default=4, help="New steps to run. With --resume, this is additional steps.")
    evolve.add_argument("--dataset", choices=("synthetic-structural-break",), default="synthetic-structural-break")
    evolve.add_argument("--n-series", type=int, default=240)
    evolve.add_argument("--length", type=int, default=160)
    evolve.add_argument("--boundary", type=int, default=None)
    evolve.add_argument("--seed", type=int, default=17)
    evolve.add_argument("--split-seed", type=int, default=None)
    evolve.add_argument("--validation-fraction", type=float, default=0.2)
    evolve.add_argument("--test-fraction", type=float, default=0.2)
    evolve.add_argument("--folds", type=int, default=3)
    evolve.add_argument("--max-configurations", type=int, default=64)
    evolve.add_argument("--irls-steps", type=int, default=2)
    evolve.add_argument("--refine-globals", dest="refine_globals", action="store_true", default=True)
    evolve.add_argument("--no-refine-globals", dest="refine_globals", action="store_false")
    evolve.add_argument("--refine-steps", type=int, default=20)
    evolve.add_argument("--refine-backend", choices=("auto", "numpy", "torch"), default="auto")
    evolve.add_argument("--min-train-improvement", type=float, default=1e-6)
    evolve.add_argument("--min-validation-improvement", type=float, default=1e-6)
    evolve.add_argument("--allow-source-mutations", action="store_true")
    evolve.add_argument("--llm-provider", choices=llm_provider_choices, default="none")
    evolve.add_argument("--env-file", type=pathlib.Path, default=pathlib.Path(".env"))
    evolve.add_argument("--llm-scientist-temperature", type=float, default=0.35)
    evolve.add_argument(
        "--llm-island-temperatures",
        type=parse_temperature_schedule,
        default=DEFAULT_ISLAND_TEMPERATURES,
        help=(
            "Comma-separated scientist temperatures for island mode, or 'none' to use "
            "--llm-scientist-temperature for every island."
        ),
    )
    evolve.add_argument("--llm-engineer-temperature", type=float, default=0.0)
    evolve.add_argument("--task-context-file", type=pathlib.Path, default=None)
    evolve.add_argument("--task-source-file", action="append", type=pathlib.Path, default=[])
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

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


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
    task_context = args.task_context_file.read_text(encoding="utf-8") if args.task_context_file is not None else ""
    task_sources = read_task_sources(args.task_source_file)
    scientist, engineer, memorandum_agent = build_llm_agents(args, task_context=task_context)
    source_allowed = args.allow_source_mutations or scientist is not None or engineer is not None
    mutation_engine = MutationEngine(allow_source=source_allowed)
    loop = EvolutionLoop(
        graph,
        evaluator=evaluator,
        mutation_engine=mutation_engine,
        scientist=scientist,
        engineer=engineer,
        memorandum_agent=memorandum_agent,
        task_context=task_context,
        task_sources=task_sources,
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
    task_context = args.task_context_file.read_text(encoding="utf-8") if args.task_context_file is not None else ""
    task_sources = read_task_sources(args.task_source_file)
    scientist, engineer, memorandum_agent = build_llm_agents(args, task_context=task_context)
    config = ProductionConfig(
        output_dir=args.output,
        steps=args.steps,
        seed=args.seed,
        dataset_name=args.dataset,
        n_series=args.n_series,
        length=args.length,
        boundary=args.boundary,
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
        allow_source_mutations=args.allow_source_mutations or scientist is not None or engineer is not None,
    )
    summary = ProductionEvolutionRunner(
        config,
        scientist=scientist,
        engineer=engineer,
        memorandum_agent=memorandum_agent,
        task_context=task_context,
        task_sources=task_sources,
    ).run(resume=args.resume)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def build_llm_agents(
    args: argparse.Namespace,
    *,
    task_context: str = "",
) -> tuple[LLMScientistAgent | None, LLMEngineerAgent | None, LLMMemorandumAgent | None]:
    provider = resolve_llm_provider(args.llm_provider, args.env_file)
    if provider is None:
        return None, None, None
    client = llm_client_from_env(args.env_file)
    source_allowed = args.allow_source_mutations or provider is not None
    prompt_builder = PromptBuilder(task_context=task_context or PromptBuilder().task_context, allow_source=source_allowed)
    return (
        LLMScientistAgent(
            client,
            prompt_builder=prompt_builder,
            temperature=args.llm_scientist_temperature,
            island_temperatures=args.llm_island_temperatures,
        ),
        LLMEngineerAgent(
            client,
            prompt_builder=prompt_builder,
            temperature=args.llm_engineer_temperature,
            allow_source=source_allowed,
        ),
        LLMMemorandumAgent(
            client,
            prompt_builder=prompt_builder,
            temperature=0.0,
        ),
    )


def read_task_sources(paths: list[pathlib.Path]) -> tuple[tuple[str, str], ...]:
    return tuple((str(path), path.read_text(encoding="utf-8")) for path in paths)


def resolve_llm_provider(provider: str, env_file: pathlib.Path) -> str | None:
    if provider == "none":
        return None
    if provider == "env":
        return llm_provider_from_env(env_file, required=True)
    load_env_file(env_file)
    os.environ["EVOFOREST_LLM_PROVIDER"] = provider
    return provider


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


if __name__ == "__main__":
    raise SystemExit(main())
