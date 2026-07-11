from __future__ import annotations

import json

from benchmarks import run_all


def test_benchmark_bundle_quick_reports(tmp_path) -> None:
    json_path, markdown_path = run_all.run(tmp_path, seed=17, quick=True)

    assert json_path.exists()
    assert markdown_path.exists()
    index = json.loads(json_path.read_text(encoding="utf-8"))
    assert index["summary"]["all_passed"] is True
    assert {row["suite"] for row in index["suites"]} == {
        "conformance_report",
        "synthetic_suite",
        "ablation_suite",
        "runtime_scaling",
    }

    for row in index["suites"]:
        suite_json = tmp_path / f"{row['suite']}.json"
        suite_markdown = tmp_path / f"{row['suite']}.md"
        assert suite_json.exists()
        assert suite_markdown.exists()
        payload = json.loads(suite_json.read_text(encoding="utf-8"))
        assert payload["scope"].startswith("These benchmarks validate architecture-level behavior")

    conformance = json.loads((tmp_path / "conformance_report.json").read_text(encoding="utf-8"))
    assert conformance["summary"]["all_passed"] is True
    assert conformance["summary"]["passed"] == conformance["summary"]["total"]

    synthetic = json.loads((tmp_path / "synthetic_suite.json").read_text(encoding="utf-8"))
    assert synthetic["summary"]["all_passed"] is True
    assert len(synthetic["cases"]) == 5

    runtime = json.loads((tmp_path / "runtime_scaling.json").read_text(encoding="utf-8"))
    assert runtime["summary"]["passed"] is True
    assert len(runtime["scenarios"]) >= 8
