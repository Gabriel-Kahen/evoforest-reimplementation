# Competition Row Multi-Split Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-row` from `/home/gabe/evoforest-competition-data`, rows=`63449`, ids=`2000`, max_ids=`2000`, max_rows_per_id=`32`.

Config: `{'folds': 3, 'max_configurations': 16, 'split_seeds': [113, 127, 149], 'stability_weight': 0.5, 'objective_mode': 'auc', 'prune_tolerance': 0.001, 'include_focused_row_templates': True, 'include_builtin_templates': False, 'include_source_mutations': False, 'skip_pruning': False, 'prune_scoring_mode': 'fixed-config', 'objective': 'mean_validation_auc + stability_weight * min_validation_auc'}`

Reduced test accessed: `False`.

Best graph by objective: `row_baseline_graph` with mean validation AUC `0.6817` and mean delta `0.0029`.

Internal non-selection test audit for pruned output-only suite: mean graph AUC `0.6501`, mean delta `0.0000`, minimum delta `0.0000`.

Internal non-selection test audit for best objective graph `row_baseline_graph`: mean graph AUC `0.6414`.

Best archive/OOF ensemble `top_7_archive` mean validation AUC `0.6845`, mean delta `0.0057`.

Internal non-selection test audit for selected ensemble: mean AUC `0.6522`, mean delta `0.0021`.

| Graph | Mean Val AUC | Mean Delta | Min Val AUC | Objective |
| --- | --- | --- | --- | --- |
| seed_graph | 0.6585 | -0.0203 | 0.6420 | 0.9795 |
| row_baseline_graph | 0.6817 | 0.0029 | 0.6697 | 1.0166 |
| row_baseline_only_graph | 0.6788 | 0.0000 | 0.6583 | 1.0079 |
| row_template_suite_graph | 0.6764 | -0.0024 | 0.6597 | 1.0063 |
| pruned_row_template_suite_graph | 0.6817 | 0.0029 | 0.6697 | 1.0166 |
| row_template_suite_output_only_graph | 0.6753 | -0.0035 | 0.6537 | 1.0022 |
| pruned_row_template_suite_output_only_graph | 0.6788 | 0.0000 | 0.6583 | 1.0079 |

| Ensemble | Members | Mean Val AUC | Mean Delta | Min Val AUC | Objective |
| --- | --- | --- | --- | --- | --- |
| best_archive_member | row_baseline_graph | 0.6817 | 0.0029 | 0.6697 | 1.0166 |
| top_2_archive | row_baseline_graph, pruned_row_template_suite_graph | 0.6817 | 0.0029 | 0.6697 | 1.0166 |
| top_3_archive | row_baseline_graph, pruned_row_template_suite_graph, row_baseline_only_graph | 0.6820 | 0.0032 | 0.6666 | 1.0153 |
| top_5_archive | row_baseline_graph, pruned_row_template_suite_graph, row_baseline_only_graph, pruned_row_template_suite_output_only_graph, row_template_suite_graph | 0.6834 | 0.0047 | 0.6658 | 1.0164 |
| top_7_archive | row_baseline_graph, pruned_row_template_suite_graph, row_baseline_only_graph, pruned_row_template_suite_output_only_graph, row_template_suite_graph, row_template_suite_output_only_graph, seed_graph | 0.6845 | 0.0057 | 0.6651 | 1.0170 |
| baseline_plus_best_archive_blend | baseline, row_baseline_graph | 0.6817 | 0.0029 | 0.6648 | 1.0141 |
| greedy_oof_archive | row_baseline_graph | 0.6817 | 0.0029 | 0.6697 | 1.0166 |

| Pruned Alternative | Removed | Mode | Objective Before | Objective After | Delta |
| --- | --- | --- | --- | --- | --- |
| row_multiscale_tail_output | yes | fixed-config | 1.0063 | 1.0171 | 0.0109 |
| row_time_basis_output | yes | fixed-config | 1.0171 | 1.0166 | -0.0005 |
| adia_row_baseline_output | no | fixed-config | 1.0166 | 0.9619 | -0.0547 |
