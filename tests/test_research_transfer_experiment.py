from __future__ import annotations

from benchmarks.research_suite.compositional_dags import generate_benchmark
from benchmarks.research_suite.transfer_experiment import run_transfer_experiment
from tests.paper_test_support import paper_test_agents


def test_transfer_experiment_compares_scratch_modules_and_whole_graph(tmp_path) -> None:
    source = generate_benchmark("shared_wave_gate", seed=61, n_train=36, n_validation=20, n_test=20)
    target = generate_benchmark("heteroscedastic_reuse", seed=62, n_train=36, n_validation=20, n_test=20)

    results = run_transfer_experiment(
        source,
        target,
        tmp_path,
        agents=paper_test_agents(),
        source_steps=1,
        target_steps=1,
        max_configurations=2,
    )

    assert {row.condition for row in results} == {"scratch", "related_modules", "whole_graph"}
    related = next(row for row in results if row.condition == "related_modules")
    assert related.transfer_report is not None
    assert all(len(row.test_rows) == 2 for row in results)
