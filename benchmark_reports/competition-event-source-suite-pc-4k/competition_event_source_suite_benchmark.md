# Competition Source-Suite Structural-Break Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`4000`, max_samples=`4000`.

Config: `{'folds': 3, 'max_configurations': 32, 'split_seeds': [211, 223, 307], 'stability_weight': 0.5, 'objective_mode': 'delta', 'prune_tolerance': 0.001, 'screen_sources': True, 'objective': 'mean_delta_vs_baseline + stability_weight * min_delta_vs_baseline'}`

Reduced test accessed: `False`.

Best graph by objective: `pruned_source_suite_graph` with mean validation AUC `0.5492` and mean delta `-0.0048`.

Internal non-selection test audit for pruned suite: mean graph AUC `0.5426`, mean delta `0.0163`, minimum delta `0.0083`.

| Graph | Mean Val AUC | Mean Delta | Objective |
| --- | --- | --- | --- |
| seed_graph | 0.5425 | -0.0115 | -0.0200 |
| source_suite_graph | 0.5380 | -0.0160 | -0.0257 |
| pruned_source_suite_graph | 0.5492 | -0.0048 | -0.0087 |

| Isolated Source | Mean Val AUC | Mean Delta | Min Delta | Objective |
| --- | --- | --- | --- | --- |
| source_autocorr_volatility_shift | 0.5523 | -0.0017 | -0.0060 | -0.0047 |
| source_spectral_entropy_shift | 0.5470 | -0.0070 | -0.0128 | -0.0134 |
| source_distribution_shift | 0.5444 | -0.0096 | -0.0129 | -0.0160 |
| source_rank_shape_shift | 0.5445 | -0.0094 | -0.0143 | -0.0166 |
| source_quantile_transport_shift | 0.5428 | -0.0112 | -0.0155 | -0.0189 |
| source_tail_extrema_shift | 0.5395 | -0.0145 | -0.0164 | -0.0227 |
| source_multiresolution_cusum_shift | 0.5376 | -0.0164 | -0.0214 | -0.0271 |
| source_moment_shape_shift | 0.5377 | -0.0163 | -0.0225 | -0.0276 |
| source_haar_scale_energy_shift | 0.5345 | -0.0194 | -0.0286 | -0.0337 |
| source_multiscale_tail_shift | 0.5324 | -0.0216 | -0.0270 | -0.0351 |

| Pruned Alternative | Removed | Objective Before | Objective After | Delta |
| --- | --- | --- | --- | --- |
| source_quantile_transport_shift | yes | -0.0257 | -0.0257 | 0.0000 |
| source_haar_scale_energy_shift | yes | -0.0257 | -0.0233 | 0.0024 |
| source_multiresolution_cusum_shift | yes | -0.0233 | -0.0204 | 0.0029 |
| source_moment_shape_shift | yes | -0.0204 | -0.0147 | 0.0057 |
| source_tail_extrema_shift | no | -0.0147 | -0.0176 | -0.0029 |
| source_spectral_entropy_shift | no | -0.0147 | -0.0179 | -0.0032 |
| source_rank_shape_shift | yes | -0.0147 | -0.0151 | -0.0004 |
| source_multiscale_tail_shift | yes | -0.0151 | -0.0087 | 0.0064 |
| source_autocorr_volatility_shift | no | -0.0087 | -0.0165 | -0.0078 |
| source_distribution_shift | no | -0.0087 | -0.0122 | -0.0036 |
