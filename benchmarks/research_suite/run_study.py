from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.common import output_argument, print_report_paths, quick_argument, seed_argument, write_report
from benchmarks.research_suite import controlled_benchmark, external_benchmark
from benchmarks.research_suite.compositional_dags import generate_benchmark
from evoforest_arch.paper_agents import PaperAgentBundle, build_gemini_paper_agents


def build_study(
    output_dir: Path,
    *,
    agents: PaperAgentBundle,
    seed: int = 17,
    quick: bool = False,
    manifests: list[Path] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    controlled_paths = controlled_benchmark.run(output_dir, seed=seed, quick=quick)

    external_paths: tuple[Path, Path] | None = None
    if manifests:
        external_paths = external_benchmark.run(output_dir, manifests, seed)

    return {
        "study": "when_does_graph_evolution_help",
        "quick": quick,
        "seed": seed,
        "phases": [
            "controlled_compositional_dags",
            "paper_style_gemini_evolution",
            "external_regression" if manifests else "external_regression_pending_manifests",
        ],
        "artifacts": {
            "controlled": [str(path) for path in controlled_paths],
            "external": None if external_paths is None else [str(path) for path in external_paths],
        },
        "paper_agents": type(agents.scientist).__name__,
    }


def markdown_report(payload: dict[str, object]) -> str:
    lines = [
        "# EvoForest Research Study Index",
        "",
        "Phases run in order: controlled prediction, paper-style Gemini evolution, then optional external manifests.",
    ]
    if payload["artifacts"]["external"] is None:  # type: ignore[index]
        lines.extend(["", "External SR/real datasets were not run because no frozen manifests were supplied."])
    return "\n".join(lines)


def run(
    output_dir: Path,
    *,
    agents: PaperAgentBundle,
    seed: int = 17,
    quick: bool = False,
    manifests: list[Path] | None = None,
) -> tuple[Path, Path]:
    payload = build_study(output_dir, agents=agents, seed=seed, quick=quick, manifests=manifests)
    return write_report(output_dir, "research_study_index", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the EvoForest generalization study in preregistered phase order.")
    output_argument(parser)
    seed_argument(parser)
    quick_argument(parser)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    agents = build_gemini_paper_agents(args.env_file)
    print_report_paths(run(args.output, agents=agents, seed=args.seed, quick=args.quick, manifests=args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
