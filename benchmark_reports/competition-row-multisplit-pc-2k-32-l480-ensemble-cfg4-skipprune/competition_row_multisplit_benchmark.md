# Competition Row Multi-Split Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-row` from `/home/gabe/evoforest-competition-data`, rows=`63449`, ids=`2000`, max_ids=`2000`, max_rows_per_id=`32`.

Config: `{'folds': 3, 'max_configurations': 4, 'split_seeds': [113, 127, 149], 'stability_weight': 0.5, 'objective_mode': 'auc', 'prune_tolerance': 0.001, 'include_focused_row_templates': True, 'include_builtin_templates': False, 'include_source_mutations': False, 'skip_pruning': True, 'objective': 'mean_validation_auc + stability_weight * min_validation_auc'}`

Reduced test accessed: `False`.

Best graph by objective: `row_baseline_graph` with mean validation AUC `0.6839` and mean delta `0.0051`.

Internal non-selection test audit for pruned output-only suite: mean graph AUC `0.6501`, mean delta `0.0001`, minimum delta `-0.0065`.

Internal non-selection test audit for best objective graph `row_baseline_graph`: mean graph AUC `0.6438`.

Best archive/OOF ensemble `best_archive_member` mean validation AUC `0.6839`, mean delta `0.0051`.

Internal non-selection test audit for selected ensemble: mean AUC `0.6438`, mean delta `-0.0062`.

| Graph | Mean Val AUC | Mean Delta | Min Val AUC | Objective |
| --- | --- | --- | --- | --- |
| seed_graph | 0.6474 | -0.0314 | 0.6262 | 0.9605 |
| row_baseline_graph | 0.6839 | 0.0051 | 0.6670 | 1.0173 |
| row_baseline_only_graph | 0.6788 | 0.0000 | 0.6583 | 1.0079 |
| row_template_suite_graph | 0.6730 | -0.0058 | 0.6523 | 0.9992 |
| pruned_row_template_suite_graph | 0.6730 | -0.0058 | 0.6523 | 0.9992 |
| row_template_suite_output_only_graph | 0.6753 | -0.0035 | 0.6537 | 1.0022 |
| pruned_row_template_suite_output_only_graph | 0.6753 | -0.0035 | 0.6537 | 1.0022 |

| Ensemble | Members | Mean Val AUC | Mean Delta | Min Val AUC | Objective |
| --- | --- | --- | --- | --- | --- |
| best_archive_member | row_baseline_graph | 0.6839 | 0.0051 | 0.6670 | 1.0173 |
| top_2_archive | row_baseline_graph, row_baseline_only_graph | 0.6826 | 0.0038 | 0.6633 | 1.0142 |
| top_3_archive | row_baseline_graph, row_baseline_only_graph, row_template_suite_output_only_graph | 0.6844 | 0.0056 | 0.6628 | 1.0158 |
| top_5_archive | row_baseline_graph, row_baseline_only_graph, row_template_suite_output_only_graph, pruned_row_template_suite_output_only_graph, row_template_suite_graph | 0.6819 | 0.0031 | 0.6599 | 1.0119 |
| top_7_archive | row_baseline_graph, row_baseline_only_graph, row_template_suite_output_only_graph, pruned_row_template_suite_output_only_graph, row_template_suite_graph, pruned_row_template_suite_graph, seed_graph | 0.6819 | 0.0032 | 0.6605 | 1.0122 |
| baseline_plus_best_archive_blend | baseline, row_baseline_graph | 0.6826 | 0.0038 | 0.6633 | 1.0142 |
| greedy_oof_archive | row_baseline_graph | 0.6839 | 0.0051 | 0.6670 | 1.0173 |

| Pruned Alternative | Removed | Objective Before | Objective After | Delta |
| --- | --- | --- | --- | --- |
