# EvoForest Conformance Report

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Seed: `17`

Passed: `18/18`

| Requirement | Status | Evidence |
| --- | --- | --- |
| dag_nodes | PASS | `{"node_kinds": {"callable": 1, "fitting": 2, "input": 2, "intermediate": 3, "output": 1}, "nodes": 9}` |
| output_semantics | PASS | `{"n_features": 21, "n_output_alternatives": 3, "output_in_config_space": false}` |
| configuration_search | PASS | `{"auc": 0.9930555555555556, "capped": true, "evaluated": 16, "total": 48}` |
| ancestor_cache | PASS | `{"entries": 29, "hits": 111, "key": "ancestor_conditioned_subpath", "misses": 29, "shared_across_configurations": true}` |
| fitting_nodes | PASS | `{"ridge_g": {"alternative": "huber", "description": "Huber-style residual downweighting.", "irls": [{"final_alpha": 31.622776601683793, "...` |
| global_parameters | PASS | `{"referenced_globals": ["gate_scale", "projection_vector", "residual_huber_scale"], "trainable_globals": ["gate_scale", "projection_vecto...` |
| two_phase_evaluation | PASS | `{"accepted_updates": 0, "backend": "numpy_coordinate", "enabled": true, "fallback_reason": "", "final_loss": 0.056878406904416506, "initi...` |
| ridge_diagnostics | PASS | `{"linear_shap": {"basis": "standardized linear Ridge contribution z_j * coefficient_j", "cv_reconstruction_error": 2.220446049250313e-16,...` |
| alternative_history | PASS | `{"n_alternative_stats": 15, "sample": [{"age": 1, "alternative": "basic", "best_config_auc": 0.9930555555555556, "kind": "intermediate", ...` |
| mutation_yaml | PASS | `{"add": [{"alternative_id": "spectral_conformance", "description": "Conformance mutation alternative.", "global_refs": [], "kind": "add_a...` |
| maintenance | PASS | `{"collapsed_duplicates": [], "removed_alternatives": [], "removed_globals": [], "removed_nodes": []}` |
| failed_feedback_salvage_surface | PASS | `{"event_salvaged": [], "has_salvage_method": true}` |
| artifacts | PASS | `{"archive_rows": 1, "checkpoint_keys": ["diagnostics_toon", "feedback", "graph", "island", "result", "step"], "run_dir": "benchmark_repor...` |
| memorandum | PASS | `{"sections_present": ["[OUTCOME HISTORY]", "[STATE]", "[WHAT WORKS]", "[WHAT FAILED]", "[ERROR LOG]"]}` |
| task_context | PASS | `{"contains_scorer_mechanics": true, "contains_tensor_inventory": true}` |
| llm_two_stage | PASS | `{"engineer_temperature": 0.0, "island_2_temperature": 0.6, "scientist_island_temperatures": [0.35, 0.5, 0.6, 0.75]}` |
| source_mutation_gate | PASS | `{"allow_source_false": false, "allow_source_true": true}` |
| graph_maintenance_api | PASS | `{"maintenance_methods": ["_collapse_duplicate_alternatives", "_prune_unreachable_nodes", "_prune_unused_globals"]}` |
