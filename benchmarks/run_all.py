from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from benchmarks import ablation_suite, conformance_report, runtime_scaling, synthetic_suite
from benchmarks.common import markdown_table, output_argument, print_report_paths, quick_argument, report_scope, seed_argument, write_report


SuiteRunner = Callable[[Path, int, bool], tuple[Path, Path]]


def build_report(output_dir: Path, seed: int = 17, quick: bool = False) -> dict[str, object]:
    suites: list[tuple[str, SuiteRunner, int]] = [
        ("conformance_report", conformance_report.run, seed),
        ("synthetic_suite", synthetic_suite.run, seed + 6),
        ("ablation_suite", ablation_suite.run, seed + 12),
        ("runtime_scaling", runtime_scaling.run, seed + 20),
    ]
    rows = []
    for name, runner, suite_seed in suites:
        json_path, markdown_path = runner(output_dir, suite_seed, quick)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "suite": name,
                "seed": suite_seed,
                "json": str(json_path),
                "markdown": str(markdown_path),
                "summary": payload.get("summary", {}),
            }
        )
    return {
        "benchmark": "all",
        "scope": report_scope(),
        "seed": seed,
        "quick": quick,
        "summary": {
            "suites": len(rows),
            "all_passed": all(suite_passed(row["summary"]) for row in rows),
        },
        "suites": rows,
    }


def suite_passed(summary: object) -> bool:
    if not isinstance(summary, dict):
        return False
    if "all_passed" in summary:
        return bool(summary["all_passed"])
    if "passed" in summary and isinstance(summary["passed"], bool):
        return bool(summary["passed"])
    return bool(summary)


def markdown_report(payload: dict[str, object]) -> str:
    rows = []
    for suite in payload["suites"]:
        summary = suite["summary"]
        all_passed = suite_passed(summary)
        rows.append(
            [
                suite["suite"],
                "PASS" if all_passed else "CHECK",
                suite["seed"],
                suite["json"],
                suite["markdown"],
            ]
        )
    return "\n\n".join(
        [
            "# EvoForest Benchmark Index",
            str(payload["scope"]),
            f"Base seed: `{payload['seed']}`",
            f"All suites passed: `{payload['summary']['all_passed']}`",
            markdown_table(["Suite", "Status", "Seed", "JSON", "Markdown"], rows),
        ]
    )


def run(output_dir: Path, seed: int = 17, quick: bool = False) -> tuple[Path, Path]:
    payload = build_report(output_dir, seed=seed, quick=quick)
    return write_report(output_dir, "benchmark_index", payload, markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all EvoForest faithfulness benchmarks.")
    output_argument(parser)
    seed_argument(parser, default=17)
    quick_argument(parser)
    args = parser.parse_args(argv)
    print_report_paths(run(args.output, seed=args.seed, quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
