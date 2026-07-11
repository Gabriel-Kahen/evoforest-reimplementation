from __future__ import annotations

import json

from benchmarks.research_suite.study_spec import confirmatory_spec, pilot_spec, write_spec


def test_study_specs_are_deterministic_and_confirmatory_is_locked(tmp_path) -> None:
    pilot = pilot_spec()
    confirmatory = confirmatory_spec()

    assert pilot.evolution_steps == 4
    assert confirmatory.status == "confirmatory_locked"
    assert confirmatory.fingerprint() == "924a56ffccc3a5ebeaa2f5e575897c8aed5c164c6a396a6c9b38486416551232"
    assert len(confirmatory.data_seeds) * len(confirmatory.task_families) == 32
    assert confirmatory.fingerprint() == confirmatory_spec().fingerprint()

    path = write_spec(tmp_path / "confirmatory.json", confirmatory)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["fingerprint"] == confirmatory.fingerprint()
