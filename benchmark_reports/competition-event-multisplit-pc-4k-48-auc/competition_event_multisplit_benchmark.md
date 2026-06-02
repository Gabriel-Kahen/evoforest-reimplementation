# Competition Multi-Split Structural-Break Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`4000`, max_samples=`4000`.

Config: `{'steps': 48, 'folds': 3, 'max_configurations': 32, 'split_seeds': [211, 223, 307], 'include_source_mutations': True, 'min_objective_improvement': 0.0001, 'stability_weight': 0.5, 'objective_mode': 'auc', 'prune_tolerance': 0.001, 'objective': 'mean_validation_auc + stability_weight * min_validation_auc'}`

Reduced test accessed: `False`.

Pruned consensus graph mean validation AUC `0.5611`, mean delta `0.0071`, minimum delta `0.0041`, objective `0.8289`.

Best archive/OOF ensemble `top_2_archive` mean validation AUC `0.5611`, mean delta `0.0071`, minimum delta `0.0041`.

Internal non-selection test audit mean graph AUC `0.5371`, mean delta `0.0108`, minimum delta `0.0081`.

| Split Seed | Train | Validation | Test | Baseline Val AUC |
| --- | --- | --- | --- | --- |
| 211 | 2560 | 640 | 800 | 0.5283 |
| 223 | 2560 | 640 | 800 | 0.5952 |
| 307 | 2560 | 640 | 800 | 0.5385 |

| Step | Accepted | Node | Primitive | Source | Mean Val AUC | Mean Delta | Min Delta | Objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | no | segment_stats | row_recent_change | no | 0.5420 | -0.0119 | -0.0171 | 0.7980 |
| 2 | no | segment_stats | segment_late_shift | no | 0.5278 | -0.0261 | -0.0450 | 0.7838 |
| 3 | yes | output | source | yes | 0.5523 | -0.0017 | -0.0060 | 0.8134 |
| 4 | no | output | source | yes | 0.5464 | -0.0076 | -0.0142 | 0.8037 |
| 5 | no | output | source | yes | 0.5429 | -0.0110 | -0.0150 | 0.7996 |
| 6 | no | output | source | yes | 0.5435 | -0.0105 | -0.0180 | 0.8032 |
| 7 | no | output | source | yes | 0.5427 | -0.0113 | -0.0191 | 0.7995 |
| 8 | no | output | source | yes | 0.5481 | -0.0059 | -0.0135 | 0.8055 |
| 9 | no | output | source | yes | 0.5521 | -0.0019 | -0.0066 | 0.8129 |
| 10 | yes | output | source | yes | 0.5503 | -0.0037 | -0.0089 | 0.8145 |
| 11 | no | output | adia_structural_break_baseline_outputs | no | 0.5513 | -0.0027 | -0.0056 | 0.8126 |
| 12 | no | output | competition_event_outputs | no | 0.5467 | -0.0073 | -0.0334 | 0.8127 |
| 13 | yes | output | interaction_outputs | no | 0.5586 | 0.0046 | 0.0016 | 0.8263 |
| 14 | no | output | row_local_outputs | no | 0.5517 | -0.0022 | -0.0080 | 0.8163 |
| 15 | no | trend_stats | trend_late_window | no | 0.5538 | -0.0002 | -0.0082 | 0.8189 |
| 16 | no | ridge_w | late_energy_weight | no | 0.5586 | 0.0046 | 0.0016 | 0.8263 |
| 17 | yes | shape_stats | row_cusum_local | no | 0.5595 | 0.0055 | 0.0016 | 0.8272 |
| 18 | no | shape_stats | shape_drawdown | no | 0.5461 | -0.0079 | -0.0288 | 0.8138 |
| 19 | no | shape_stats | shape_post_concentration | no | 0.5419 | -0.0121 | -0.0304 | 0.8042 |
| 20 | yes | output | source | yes | 0.5611 | 0.0071 | 0.0041 | 0.8289 |
| 21 | no | output | source | yes | 0.5519 | -0.0020 | -0.0078 | 0.8122 |
| 22 | no | output | row_context_outputs | no | 0.5611 | 0.0071 | 0.0041 | 0.8289 |
| 23 | no | shape_stats | row_volatility_burst | no | 0.5601 | 0.0062 | 0.0012 | 0.8280 |
| 24 | no | output | source | yes | 0.5519 | -0.0020 | -0.0089 | 0.8116 |
| 25 | no | output | source | yes | 0.5557 | 0.0017 | -0.0038 | 0.8179 |
| 26 | no | output | source | yes | 0.5577 | 0.0037 | -0.0015 | 0.8211 |
| 27 | no | output | source | yes | 0.5519 | -0.0020 | -0.0078 | 0.8122 |
| 28 | no | output | adia_structural_break_baseline_outputs | no | 0.5499 | -0.0040 | -0.0109 | 0.8128 |
| 29 | no | output | competition_event_outputs | no | 0.5456 | -0.0084 | -0.0365 | 0.8116 |
| 30 | no | output | row_context_outputs | no | 0.5611 | 0.0071 | 0.0041 | 0.8289 |
| 31 | no | output | row_local_outputs | no | 0.5484 | -0.0055 | -0.0148 | 0.8125 |
| 32 | no | trend_stats | trend_late_window | no | 0.5528 | -0.0012 | -0.0092 | 0.8175 |
| 33 | no | ridge_w | late_energy_weight | no | 0.5611 | 0.0071 | 0.0041 | 0.8289 |
| 34 | no | shape_stats | row_volatility_burst | no | 0.5601 | 0.0062 | 0.0012 | 0.8280 |
| 35 | no | shape_stats | shape_drawdown | no | 0.5601 | 0.0062 | 0.0012 | 0.8280 |
| 36 | no | shape_stats | shape_post_concentration | no | 0.5502 | -0.0038 | -0.0241 | 0.8200 |
| 37 | no | segment_stats | row_recent_change | no | 0.5569 | 0.0029 | -0.0053 | 0.8184 |
| 38 | no | segment_stats | segment_late_shift | no | 0.5451 | -0.0089 | -0.0320 | 0.8099 |
| 39 | no | output | source | yes | 0.5531 | -0.0009 | -0.0075 | 0.8134 |
| 40 | no | output | source | yes | 0.5496 | -0.0044 | -0.0146 | 0.8064 |
| 41 | no | output | source | yes | 0.5553 | 0.0013 | -0.0014 | 0.8199 |
| 42 | no | output | source | yes | 0.5519 | -0.0020 | -0.0089 | 0.8116 |
| 43 | no | output | source | yes | 0.5557 | 0.0017 | -0.0038 | 0.8179 |
| 44 | no | output | source | yes | 0.5577 | 0.0037 | -0.0015 | 0.8211 |
| 45 | no | output | source | yes | 0.5519 | -0.0020 | -0.0078 | 0.8122 |
| 46 | no | output | adia_structural_break_baseline_outputs | no | 0.5499 | -0.0040 | -0.0109 | 0.8128 |
| 47 | no | output | competition_event_outputs | no | 0.5456 | -0.0084 | -0.0365 | 0.8116 |
| 48 | no | output | row_context_outputs | no | 0.5611 | 0.0071 | 0.0041 | 0.8289 |
