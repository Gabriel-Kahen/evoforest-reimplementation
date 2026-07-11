from __future__ import annotations

from benchmarks.research_suite.run_study import build_study
from tests.paper_test_support import paper_test_agents


def test_quick_study_runs_phases_in_order(tmp_path) -> None:
    report = build_study(tmp_path, agents=paper_test_agents(), seed=81, quick=True)

    assert report["phases"] == [
        "controlled_compositional_dags",
        "paper_style_gemini_evolution",
        "external_regression_pending_manifests",
    ]
    assert report["paper_agents"] == "LLMScientistAgent"
