from __future__ import annotations

from dataclasses import replace

from benchmarks.research_suite.pilot_runner import run_medium_pilot
from benchmarks.research_suite.study_spec import pilot_spec
from tests.paper_test_support import paper_test_agents


def test_small_pilot_produces_assessment_without_dropping_failures(tmp_path) -> None:
    tiny = replace(
        pilot_spec(),
        task_families=("shared_wave_gate", "piecewise_rational", "heteroscedastic_reuse"),
        data_seeds=(9,),
        search_seeds=(10,),
        n_train=36,
        n_validation=20,
        n_test=20,
        evolution_steps=1,
        max_configurations=2,
        screening_finalists=2,
    )

    payload = run_medium_pilot(tmp_path, tiny, agents=paper_test_agents())

    assert payload["pilot_spec_fingerprint"] == tiny.fingerprint()
    assert payload["assessment"]["task_difficulty_flag"] in {"too_easy", "too_hard", "usable"}
    assert isinstance(payload["failures"], list)
