from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from benchmarks.common import fmt_float, markdown_table, output_argument, print_report_paths, report_scope, seed_argument, write_report
from benchmarks.competition_event_benchmark import build_report, markdown_report as event_markdown_report
from evoforest_arch.competition import DEFAULT_COMPETITION_DATA_DIR


def build_campaign_report(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seeds: tuple[int, ...] = (211, 223, 227),
    series_length: int = 160,
    max_samples: int | None = None,
    steps: int = 96,
    folds: int = 3,
    max_configurations: int = 64,
    include_source_mutations: bool = True,
) -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_start = time.perf_counter()
        payload = build_report(
            seed_dir,
            data_dir=data_dir,
            seed=seed,
            series_length=series_length,
            max_samples=max_samples,
            steps=steps,
            folds=folds,
            max_configurations=max_configurations,
            include_source_mutations=include_source_mutations,
        )
        json_path = seed_dir / "competition_event_benchmark.json"
        markdown_path = seed_dir / "competition_event_benchmark.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        markdown_path.write_text(event_markdown_report(payload).rstrip() + "\n", encoding="utf-8")
        seed_rows.append(
            {
                "seed": int(seed),
                "json": str(json_path),
                "markdown": str(markdown_path),
                "seconds": float(time.perf_counter() - seed_start),
                "baseline_validation_auc": float(payload["baseline"]["validation_auc"]),
                "seed_graph_validation_auc": float(payload["seed_graph"]["validation_auc"]),
                "evolved_validation_auc": float(payload["evolved_graph"]["validation_auc"]),
                "evolved_delta_vs_baseline": float(payload["evolved_graph"]["validation_delta_vs_baseline"]),
                "best_ensemble_name": payload["ensembles"]["best"]["name"] if payload["ensembles"]["best"] else "",
                "best_ensemble_validation_auc": float(payload["ensembles"]["best"]["validation_auc"]) if payload["ensembles"]["best"] else 0.5,
                "best_ensemble_delta_vs_baseline": float(payload["ensembles"]["best"]["validation_delta_vs_baseline"]) if payload["ensembles"]["best"] else 0.0,
                "accepted_mutations": int(payload["evolution"]["accepted_mutations"]),
                "source_backed_candidates": int(payload["evolution"]["source_backed_candidates"]),
                "failed_candidates": int(payload["evolution"]["failed_candidates"]),
                "reduced_test_accessed": bool(payload["reduced_test_access"]["accessed"]),
                "group_overlap": dict(payload["split"]["audit"]["group_overlaps"]),
            }
        )
    best_evolved = max(seed_rows, key=lambda row: float(row["evolved_validation_auc"])) if seed_rows else None
    best_ensemble = max(seed_rows, key=lambda row: float(row["best_ensemble_validation_auc"])) if seed_rows else None
    return {
        "benchmark": "competition_event_campaign",
        "scope": report_scope(),
        "data_dir": str(data_dir),
        "seeds": list(seeds),
        "config": {
            "series_length": int(series_length),
            "max_samples": max_samples,
            "steps": int(steps),
            "folds": int(folds),
            "max_configurations": int(max_configurations),
            "include_source_mutations": bool(include_source_mutations),
        },
        "summary": {
            "seed_count": len(seed_rows),
            "seconds": float(time.perf_counter() - start),
            "accepted_mutations": sum(int(row["accepted_mutations"]) for row in seed_rows),
            "source_backed_candidates": sum(int(row["source_backed_candidates"]) for row in seed_rows),
            "failed_candidates": sum(int(row["failed_candidates"]) for row in seed_rows),
            "any_reduced_test_accessed": any(bool(row["reduced_test_accessed"]) for row in seed_rows),
            "best_evolved": best_evolved,
            "best_ensemble": best_ensemble,
        },
        "seeds_detail": seed_rows,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    rows = [
        [
            row["seed"],
            fmt_float(row["baseline_validation_auc"]),
            fmt_float(row["evolved_validation_auc"]),
            fmt_float(row["evolved_delta_vs_baseline"]),
            row["best_ensemble_name"],
            fmt_float(row["best_ensemble_validation_auc"]),
            fmt_float(row["best_ensemble_delta_vs_baseline"]),
            row["accepted_mutations"],
            row["source_backed_candidates"],
            row["seconds"],
        ]
        for row in payload["seeds_detail"]
    ]
    summary = payload["summary"]
    best_ensemble = summary["best_ensemble"] or {}
    return "\n\n".join(
        [
            "# Competition Id-Level Campaign",
            str(payload["scope"]),
            f"Data dir: `{payload['data_dir']}`",
            f"Config: `{payload['config']}`",
            (
                f"Best ensemble seed: `{best_ensemble.get('seed', 'n/a')}`; "
                f"validation AUC `{fmt_float(best_ensemble.get('best_ensemble_validation_auc'))}`; "
                f"delta vs baseline `{fmt_float(best_ensemble.get('best_ensemble_delta_vs_baseline'))}`."
            ),
            f"Reduced test accessed: `{summary['any_reduced_test_accessed']}`.",
            markdown_table(
                [
                    "Seed",
                    "Baseline Val AUC",
                    "Evolved Val AUC",
                    "Evolved Delta",
                    "Best Ensemble",
                    "Ensemble Val AUC",
                    "Ensemble Delta",
                    "Accepted",
                    "Source Candidates",
                    "Seconds",
                ],
                rows,
            ),
        ]
    )


def run(
    output_dir: Path,
    *,
    data_dir: Path = DEFAULT_COMPETITION_DATA_DIR,
    seeds: tuple[int, ...] = (211, 223, 227),
    series_length: int = 160,
    max_samples: int | None = None,
    steps: int = 96,
    folds: int = 3,
    max_configurations: int = 64,
    include_source_mutations: bool = True,
) -> tuple[Path, Path]:
    payload = build_campaign_report(
        output_dir,
        data_dir=data_dir,
        seeds=seeds,
        series_length=series_length,
        max_samples=max_samples,
        steps=steps,
        folds=folds,
        max_configurations=max_configurations,
        include_source_mutations=include_source_mutations,
    )
    return write_report(output_dir, "competition_event_campaign", payload, markdown_report(payload))


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a multi-seed id-level structural-break campaign.")
    output_argument(parser)
    seed_argument(parser, default=211)
    parser.add_argument("--seeds", type=parse_seeds, default=None, help="Comma-separated seeds. Overrides --seed.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_COMPETITION_DATA_DIR)
    parser.add_argument("--series-length", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-configurations", type=int, default=64)
    parser.add_argument("--disable-source-mutations", action="store_true")
    args = parser.parse_args(argv)
    seeds = args.seeds if args.seeds is not None else (args.seed, args.seed + 12, args.seed + 16)
    print_report_paths(
        run(
            args.output,
            data_dir=args.data_dir,
            seeds=seeds,
            series_length=args.series_length,
            max_samples=args.max_samples,
            steps=args.steps,
            folds=args.folds,
            max_configurations=args.max_configurations,
            include_source_mutations=not args.disable_source_mutations,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
