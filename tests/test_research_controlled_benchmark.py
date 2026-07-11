from __future__ import annotations

from benchmarks.research_suite.controlled_benchmark import ControlledRunConfig, build_report, run_controlled_benchmark


def test_controlled_runner_reports_both_holdout_regimes() -> None:
    rows = run_controlled_benchmark(
        ControlledRunConfig(
            task_names=("shared_wave_gate",),
            seeds=(13,),
            n_train=48,
            n_validation=24,
            n_test=32,
            random_features=16,
        )
    )

    assert len(rows) == 6
    assert {row.method for row in rows} == {"raw_ridge", "random_features_ridge", "evoforest_seed"}
    assert {row.split_id for row in rows} == {"interpolation", "extrapolation"}
    assert all(row.metrics["nrmse"] >= 0.0 for row in rows)


def test_controlled_quick_report_is_json_ready() -> None:
    report = build_report(seed=19, quick=True)

    assert report["benchmark"] == "controlled_compositional_dags"
    assert report["results"]
