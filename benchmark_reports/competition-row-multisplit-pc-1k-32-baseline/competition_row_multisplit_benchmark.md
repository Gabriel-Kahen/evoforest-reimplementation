# Competition Row Multi-Split Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-row` from `/home/gabe/evoforest-competition-data`, rows=`31729`, ids=`1000`, max_ids=`1000`, max_rows_per_id=`32`.

Config: `{'folds': 3, 'max_configurations': 16, 'split_seeds': [113, 127, 149], 'stability_weight': 0.5, 'objective_mode': 'auc', 'prune_tolerance': 0.001, 'include_builtin_templates': False, 'include_source_mutations': False, 'objective': 'mean_validation_auc + stability_weight * min_validation_auc'}`

Reduced test accessed: `False`.

Best graph by objective: `row_baseline_only_graph` with mean validation AUC `0.6078` and mean delta `0.0000`.

Internal non-selection test audit for pruned output-only suite: mean graph AUC `0.5657`, mean delta `0.0000`, minimum delta `0.0000`.

| Graph | Mean Val AUC | Mean Delta | Min Val AUC | Objective |
| --- | --- | --- | --- | --- |
| seed_graph | 0.6013 | -0.0065 | 0.5942 | 0.8985 |
| row_baseline_graph | 0.6052 | -0.0026 | 0.5808 | 0.8956 |
| row_baseline_only_graph | 0.6078 | 0.0000 | 0.5839 | 0.8998 |
| row_template_suite_graph | 0.6052 | -0.0026 | 0.5808 | 0.8956 |
| pruned_row_template_suite_graph | 0.6013 | -0.0065 | 0.5942 | 0.8985 |
| row_template_suite_output_only_graph | 0.6078 | 0.0000 | 0.5839 | 0.8998 |
| pruned_row_template_suite_output_only_graph | 0.6078 | 0.0000 | 0.5839 | 0.8998 |

| Pruned Alternative | Removed | Objective Before | Objective After | Delta |
| --- | --- | --- | --- | --- |
| adia_row_baseline_output | yes | 0.8956 | 0.8985 | 0.0028 |
