# Competition Source-Suite Structural-Break Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`4000`, max_samples=`4000`.

Config: `{'folds': 3, 'max_configurations': 32, 'split_seeds': [211, 223, 307], 'stability_weight': 0.5, 'objective_mode': 'auc', 'prune_tolerance': 0.001, 'screen_sources': False, 'include_builtin_templates': True, 'objective': 'mean_validation_auc + stability_weight * min_validation_auc'}`

Reduced test accessed: `False`.

Best graph by objective: `pruned_source_suite_graph` with mean validation AUC `0.5505` and mean delta `-0.0035`.

Internal non-selection test audit for pruned suite: mean graph AUC `0.5319`, mean delta `0.0056`, minimum delta `0.0030`.

| Graph | Mean Val AUC | Mean Delta | Objective |
| --- | --- | --- | --- |
| seed_graph | 0.5425 | -0.0115 | 0.7985 |
| source_suite_graph | 0.5412 | -0.0128 | 0.8047 |
| pruned_source_suite_graph | 0.5505 | -0.0035 | 0.8204 |

| Isolated Template | Node | Primitive | Source | Mean Val AUC | Mean Delta | Min Delta | Objective |
| --- | --- | --- | --- | --- | --- | --- | --- |

| Pruned Alternative | Removed | Objective Before | Objective After | Delta |
| --- | --- | --- | --- | --- |
| source_quantile_transport_shift | yes | 0.8047 | 0.8056 | 0.0009 |
| source_haar_scale_energy_shift | yes | 0.8056 | 0.8047 | -0.0010 |
| source_multiresolution_cusum_shift | yes | 0.8047 | 0.8048 | 0.0001 |
| source_moment_shape_shift | yes | 0.8048 | 0.8096 | 0.0047 |
| source_tail_extrema_shift | yes | 0.8096 | 0.8116 | 0.0020 |
| source_spectral_entropy_shift | no | 0.8116 | 0.8087 | -0.0029 |
| source_rank_shape_shift | yes | 0.8116 | 0.8121 | 0.0005 |
| source_multiscale_tail_shift | yes | 0.8121 | 0.8164 | 0.0043 |
| source_autocorr_volatility_shift | no | 0.8164 | 0.8117 | -0.0047 |
| source_distribution_shift | no | 0.8164 | 0.8113 | -0.0052 |
| huber_residual_mutation | yes | 0.8164 | 0.8164 | 0.0000 |
| late_energy_weight_mutation | no | 0.8164 | 0.8117 | -0.0048 |
| boundary_weight_mutation | yes | 0.8164 | 0.8164 | 0.0000 |
| post_concentration_mutation | yes | 0.8164 | 0.8176 | 0.0012 |
| drawdown_mutation | yes | 0.8176 | 0.8176 | 0.0000 |
| projection_output_mutation | yes | 0.8176 | 0.8179 | 0.0003 |
| row_context_output_mutation | yes | 0.8179 | 0.8179 | 0.0000 |
| row_local_output_mutation | yes | 0.8179 | 0.8191 | 0.0012 |
| interaction_output_mutation | no | 0.8191 | 0.8171 | -0.0019 |
| competition_event_output_mutation | no | 0.8191 | 0.8139 | -0.0051 |
| adia_baseline_output_mutation | no | 0.8191 | 0.8099 | -0.0091 |
| sigmoid_gate_mutation | yes | 0.8191 | 0.8203 | 0.0013 |
| row_cusum_local_mutation | no | 0.8203 | 0.8170 | -0.0034 |
| row_volatility_burst_mutation | no | 0.8203 | 0.8170 | -0.0034 |
| row_recent_change_mutation | no | 0.8203 | 0.8166 | -0.0037 |
| late_trend_mutation | yes | 0.8203 | 0.8203 | -0.0000 |
| late_shift_mutation | no | 0.8203 | 0.8170 | -0.0033 |
| robust_mutation | yes | 0.8203 | 0.8203 | 0.0000 |
| spectral_mutation | yes | 0.8203 | 0.8204 | 0.0001 |
