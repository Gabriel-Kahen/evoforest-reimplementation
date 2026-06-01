from __future__ import annotations

import argparse
import json
import pathlib

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.llm import (
    DEFAULT_ISLAND_TEMPERATURES,
    HTTPJSONLLMClient,
    LLMEngineerAgent,
    LLMScientistAgent,
    PromptBuilder,
)
from evoforest_arch.mutations import MutationEngine
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


if __name__ == "__main__":
    raise SystemExit(main())
