# Competition Multi-Split Structural-Break Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`4000`, max_samples=`4000`.

Config: `{'steps': 48, 'folds': 3, 'max_configurations': 32, 'split_seeds': [211, 223, 307], 'include_source_mutations': True, 'min_objective_improvement': 0.0001, 'stability_weight': 0.5, 'prune_tolerance': 0.001, 'objective': 'mean_delta_vs_baseline + stability_weight * min_delta_vs_baseline'}`

Reduced test accessed: `False`.

Pruned consensus graph mean validation AUC `0.5611`, mean delta `0.0071`, minimum delta `0.0041`, objective `0.0092`.

Best archive/OOF ensemble `top_2_archive` mean validation AUC `0.5611`, mean delta `0.0071`, minimum delta `0.0041`.

Internal non-selection test audit mean graph AUC `0.5371`, mean delta `0.0108`, minimum delta `0.0081`.

| Split Seed | Train | Validation | Test | Baseline Val AUC |
| --- | --- | --- | --- | --- |
| 211 | 2560 | 640 | 800 | 0.5283 |
| 223 | 2560 | 640 | 800 | 0.5952 |
| 307 | 2560 | 640 | 800 | 0.5385 |

| Step | Accepted | Node | Primitive | Source | Mean Val AUC | Mean Delta | Min Delta | Objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | no | segment_stats | row_recent_change | no | 0.5420 | -0.0119 | -0.0171 | -0.0205 |
| 2 | no | segment_stats | segment_late_shift | no | 0.5278 | -0.0261 | -0.0450 | -0.0487 |
| 3 | yes | output | source | yes | 0.5523 | -0.0017 | -0.0060 | -0.0047 |
| 4 | no | output | source | yes | 0.5464 | -0.0076 | -0.0142 | -0.0146 |
| 5 | no | output | source | yes | 0.5429 | -0.0110 | -0.0150 | -0.0185 |
| 6 | no | output | source | yes | 0.5435 | -0.0105 | -0.0180 | -0.0195 |
| 7 | no | output | source | yes | 0.5427 | -0.0113 | -0.0191 | -0.0209 |
| 8 | no | output | source | yes | 0.5481 | -0.0059 | -0.0135 | -0.0126 |
| 9 | no | output | source | yes | 0.5521 | -0.0019 | -0.0066 | -0.0052 |
| 10 | no | output | source | yes | 0.5503 | -0.0037 | -0.0089 | -0.0082 |
| 11 | no | output | source | yes | 0.5461 | -0.0079 | -0.0133 | -0.0145 |
| 12 | no | output | adia_structural_break_baseline_outputs | no | 0.5501 | -0.0039 | -0.0081 | -0.0080 |
| 13 | no | output | competition_event_outputs | no | 0.5451 | -0.0089 | -0.0343 | -0.0260 |
| 14 | yes | output | interaction_outputs | no | 0.5571 | 0.0031 | -0.0001 | 0.0031 |
| 15 | no | output | row_local_outputs | no | 0.5506 | -0.0034 | -0.0061 | -0.0064 |
| 16 | no | trend_stats | trend_late_window | no | 0.5492 | -0.0047 | -0.0162 | -0.0129 |
| 17 | no | ridge_w | late_energy_weight | no | 0.5571 | 0.0031 | -0.0001 | 0.0031 |
| 18 | no | shape_stats | row_cusum_local | no | 0.5531 | -0.0009 | -0.0047 | -0.0032 |
| 19 | no | shape_stats | row_volatility_burst | no | 0.5561 | 0.0021 | -0.0001 | 0.0021 |
| 20 | no | shape_stats | shape_drawdown | no | 0.5441 | -0.0099 | -0.0316 | -0.0257 |
| 21 | no | shape_stats | shape_post_concentration | no | 0.5535 | -0.0005 | -0.0110 | -0.0060 |
| 22 | yes | output | source | yes | 0.5578 | 0.0039 | 0.0015 | 0.0046 |
| 23 | no | output | row_context_outputs | no | 0.5578 | 0.0039 | 0.0015 | 0.0046 |
| 24 | no | output | source | yes | 0.5478 | -0.0062 | -0.0174 | -0.0149 |
| 25 | no | output | source | yes | 0.5526 | -0.0014 | -0.0033 | -0.0031 |
| 26 | no | output | source | yes | 0.5489 | -0.0050 | -0.0116 | -0.0108 |
| 27 | no | output | source | yes | 0.5536 | -0.0003 | -0.0065 | -0.0036 |
| 28 | no | output | source | yes | 0.5553 | 0.0013 | -0.0058 | -0.0016 |
| 29 | yes | output | source | yes | 0.5611 | 0.0071 | 0.0041 | 0.0092 |
| 30 | no | output | competition_event_outputs | no | 0.5458 | -0.0082 | -0.0365 | -0.0264 |
| 31 | no | output | row_context_outputs | no | 0.5611 | 0.0071 | 0.0041 | 0.0092 |
| 32 | no | output | row_local_outputs | no | 0.5484 | -0.0055 | -0.0148 | -0.0129 |
| 33 | no | trend_stats | trend_late_window | no | 0.5549 | 0.0009 | -0.0067 | -0.0024 |
| 34 | no | ridge_w | late_energy_weight | no | 0.5611 | 0.0071 | 0.0041 | 0.0092 |
| 35 | no | shape_stats | row_cusum_local | no | 0.5611 | 0.0071 | 0.0041 | 0.0092 |
| 36 | no | shape_stats | row_volatility_burst | no | 0.5600 | 0.0060 | 0.0041 | 0.0081 |
| 37 | no | shape_stats | shape_drawdown | no | 0.5556 | 0.0016 | -0.0067 | -0.0018 |
| 38 | no | shape_stats | shape_post_concentration | no | 0.5534 | -0.0006 | -0.0173 | -0.0092 |
| 39 | no | segment_stats | row_recent_change | no | 0.5546 | 0.0007 | -0.0053 | -0.0020 |
| 40 | no | segment_stats | segment_late_shift | no | 0.5479 | -0.0060 | -0.0235 | -0.0178 |
| 41 | no | output | source | yes | 0.5532 | -0.0008 | -0.0070 | -0.0043 |
| 42 | no | output | source | yes | 0.5496 | -0.0044 | -0.0146 | -0.0117 |
| 43 | no | output | source | yes | 0.5560 | 0.0020 | -0.0014 | 0.0013 |
| 44 | no | output | source | yes | 0.5519 | -0.0020 | -0.0089 | -0.0065 |
| 45 | no | output | source | yes | 0.5557 | 0.0017 | -0.0038 | -0.0002 |
| 46 | no | output | source | yes | 0.5577 | 0.0037 | -0.0015 | 0.0030 |
| 47 | no | output | source | yes | 0.5520 | -0.0020 | -0.0078 | -0.0059 |
| 48 | no | output | adia_structural_break_baseline_outputs | no | 0.5499 | -0.0040 | -0.0109 | -0.0095 |
