from __future__ import annotations

from benchmarks.research_suite.compositional_dags import generate_benchmark
from benchmarks.research_suite.evolution_experiment import EvolutionExperimentConfig, run_evolution_experiment
from tests.paper_test_support import paper_test_agents


def test_evolution_experiment_uses_validation_archive_selection_and_sealed_tests(tmp_path) -> None:
    dataset = generate_benchmark("piecewise_rational", seed=41, n_train=40, n_validation=24, n_test=24)
    rows, selected = run_evolution_experiment(
        dataset,
        tmp_path,
        agents=paper_test_agents(),
        config=EvolutionExperimentConfig(steps=1, max_configurations=3, screening_finalists=2),
    )

    assert selected.path.exists()
    assert selected.validation_nrmse >= 0.0
    assert {row.split_id for row in rows} == {"interpolation", "extrapolation"}
    assert all(row.metadata["archive_version"] == selected.archive_version for row in rows)
    assert all(row.usage.exact_evaluations >= 2 for row in rows)
