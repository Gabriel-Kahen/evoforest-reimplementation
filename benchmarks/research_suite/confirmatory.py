from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

from benchmarks.research_suite.execution_config import (
    ExecutionConfig,
    execution_capability_report,
    schema_fingerprint,
    schema_lock_valid,
)
from benchmarks.research_suite.optional_baselines import capability_report
from benchmarks.research_suite.study_spec import confirmatory_spec


def verify_confirmatory_readiness(
    *,
    llm_configured: bool = False,
    aide_configured: bool = False,
    execution_config: ExecutionConfig | None = None,
    external_manifests: tuple[Path, ...] | None = None,
    feat_executable: Path | None = None,
) -> dict[str, object]:
    root = Path(__file__).parent
    lock = json.loads((root / "specs" / "confirmatory_v2.lock.json").read_text(encoding="utf-8"))
    spec = confirmatory_spec()
    if external_manifests is None:
        external_manifests = tuple(sorted((root / "manifests" / "confirmatory_v1").glob("*.manifest.json")))
    lock_valid = lock.get("fingerprint") == spec.fingerprint()
    capabilities = {name: asdict(status) for name, status in capability_report().items()}
    execution_capabilities = (
        {name: asdict(status) for name, status in execution_capability_report(execution_config).items()}
        if execution_config is not None
        else {
            "llm": {"available": bool(llm_configured), "detail": "No frozen execution config supplied.", "credential_presence": {}},
            "aide": {"available": bool(aide_configured), "detail": "No frozen execution config supplied.", "credential_presence": {}},
        }
    )
    missing: list[str] = []
    if not lock_valid:
        missing.append("confirmatory spec fingerprint does not match its lock")
    if not schema_lock_valid():
        missing.append("execution-config schema fingerprint does not match its lock")
    isolated_feat_available = bool(
        feat_executable is not None and feat_executable.is_file() and os.access(feat_executable, os.X_OK)
    )
    if not capabilities["feat_command"]["available"] and not isolated_feat_available:
        missing.append("FEAT executable and pinned command template")
    if not execution_capabilities["llm"]["available"]:
        missing.append("configured LLM client/model and frozen pricing metadata")
    if len(external_manifests) != 12:
        missing.append("12 frozen confirmatory external dataset manifests")
    return {
        "ready": not missing,
        "lock_valid": lock_valid,
        "fingerprint": spec.fingerprint(),
        "spec": spec.to_dict(),
        "capabilities": capabilities,
        "isolated_feat_available": isolated_feat_available,
        "execution_capabilities": execution_capabilities,
        "execution_config_fingerprint": execution_config.fingerprint() if execution_config is not None else None,
        "execution_schema_fingerprint": schema_fingerprint(),
        "execution_schema_lock_valid": schema_lock_valid(),
        "external_manifest_count": len(external_manifests),
        "missing_requirements": missing,
    }
