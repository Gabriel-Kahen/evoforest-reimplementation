# Competition Multi-Split Structural-Break Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`1000`, max_samples=`1000`.

Config: `{'steps': 96, 'folds': 3, 'max_configurations': 32, 'split_seeds': [211, 223, 307], 'include_source_mutations': True, 'min_objective_improvement': 0.0001, 'stability_weight': 0.5, 'prune_tolerance': 0.001, 'objective': 'mean_delta_vs_baseline + stability_weight * min_delta_vs_baseline'}`

Reduced test accessed: `False`.

Pruned consensus graph mean validation AUC `0.5323`, mean delta `0.0393`, minimum delta `0.0324`, objective `0.0555`.

Best archive/OOF ensemble `top_2_archive` mean validation AUC `0.5323`, mean delta `0.0393`, minimum delta `0.0324`.

Internal non-selection test audit mean graph AUC `0.5562`, mean delta `-0.0258`, minimum delta `-0.0387`.

| Split Seed | Train | Validation | Test | Baseline Val AUC |
| --- | --- | --- | --- | --- |
| 211 | 639 | 161 | 200 | 0.5432 |
| 223 | 639 | 161 | 200 | 0.4961 |
| 307 | 639 | 161 | 200 | 0.4396 |

| Step | Accepted | Node | Primitive | Source | Mean Val AUC | Mean Delta | Min Delta | Objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | yes | segment_stats | row_recent_change | no | 0.4990 | 0.0060 | 0.0036 | 0.0077 |
| 2 | no | output | source | yes | 0.4899 | -0.0031 | -0.0170 | -0.0116 |
| 3 | yes | output | source | yes | 0.5086 | 0.0157 | 0.0093 | 0.0203 |
| 4 | no | output | source | yes | 0.5031 | 0.0101 | 0.0070 | 0.0136 |
| 5 | yes | output | source | yes | 0.5117 | 0.0187 | 0.0131 | 0.0253 |
| 6 | no | output | source | yes | 0.5103 | 0.0174 | 0.0096 | 0.0221 |
| 7 | no | output | source | yes | 0.5091 | 0.0161 | 0.0097 | 0.0210 |
| 8 | no | output | source | yes | 0.5091 | 0.0161 | 0.0091 | 0.0207 |
| 9 | no | output | source | yes | 0.5072 | 0.0142 | 0.0040 | 0.0162 |
| 10 | no | output | competition_event_outputs | no | 0.4982 | 0.0053 | -0.0020 | 0.0042 |
| 11 | no | output | interaction_outputs | no | 0.5067 | 0.0137 | 0.0060 | 0.0167 |
| 12 | no | output | row_context_outputs | no | 0.5117 | 0.0187 | 0.0131 | 0.0253 |
| 13 | no | output | row_local_outputs | no | 0.5093 | 0.0163 | 0.0142 | 0.0234 |
| 14 | yes | trend_stats | trend_late_window | no | 0.5143 | 0.0213 | 0.0099 | 0.0262 |
| 15 | no | shape_stats | row_cusum_local | no | 0.5085 | 0.0155 | 0.0099 | 0.0204 |
| 16 | no | shape_stats | row_volatility_burst | no | 0.5085 | 0.0155 | 0.0099 | 0.0204 |
| 17 | no | shape_stats | shape_drawdown | no | 0.5085 | 0.0155 | 0.0099 | 0.0204 |
| 18 | no | shape_stats | shape_post_concentration | no | 0.5085 | 0.0155 | 0.0099 | 0.0204 |
| 19 | no | segment_stats | segment_late_shift | no | 0.5143 | 0.0213 | 0.0099 | 0.0262 |
| 20 | no | output | source | yes | 0.5084 | 0.0154 | 0.0093 | 0.0200 |
| 21 | no | output | source | yes | 0.5033 | 0.0103 | 0.0065 | 0.0135 |
| 22 | no | output | source | yes | 0.5071 | 0.0141 | -0.0015 | 0.0133 |
| 23 | yes | output | source | yes | 0.5147 | 0.0217 | 0.0110 | 0.0272 |
| 24 | no | output | source | yes | 0.5163 | 0.0233 | 0.0079 | 0.0273 |
| 25 | no | output | source | yes | 0.5162 | 0.0232 | 0.0015 | 0.0240 |
| 26 | no | output | competition_event_outputs | no | 0.5035 | 0.0105 | 0.0059 | 0.0134 |
| 27 | yes | output | interaction_outputs | no | 0.5192 | 0.0262 | 0.0148 | 0.0336 |
| 28 | no | ridge_w | late_energy_weight | no | 0.5122 | 0.0192 | 0.0096 | 0.0239 |
| 29 | yes | shape_stats | row_cusum_local | no | 0.5197 | 0.0267 | 0.0192 | 0.0363 |
| 30 | no | shape_stats | shape_post_concentration | no | 0.5165 | 0.0235 | 0.0096 | 0.0283 |
| 31 | yes | segment_stats | segment_late_shift | no | 0.5200 | 0.0270 | 0.0192 | 0.0366 |
| 32 | yes | output | source | yes | 0.5237 | 0.0307 | 0.0272 | 0.0443 |
| 33 | no | output | competition_event_outputs | no | 0.5075 | 0.0145 | 0.0009 | 0.0149 |
| 34 | no | output | row_context_outputs | no | 0.5237 | 0.0307 | 0.0272 | 0.0443 |
| 35 | no | output | row_local_outputs | no | 0.5231 | 0.0301 | 0.0244 | 0.0423 |
| 36 | yes | ridge_w | late_energy_weight | no | 0.5323 | 0.0393 | 0.0324 | 0.0555 |
| 37 | no | output | source | yes | 0.5274 | 0.0344 | 0.0258 | 0.0473 |
| 38 | no | output | source | yes | 0.5273 | 0.0343 | 0.0247 | 0.0466 |
| 39 | no | output | source | yes | 0.5239 | 0.0309 | 0.0216 | 0.0418 |
| 40 | no | output | source | yes | 0.5308 | 0.0378 | 0.0293 | 0.0525 |
| 41 | no | output | source | yes | 0.5301 | 0.0371 | 0.0318 | 0.0530 |
| 42 | no | output | source | yes | 0.5187 | 0.0257 | 0.0119 | 0.0316 |
| 43 | no | output | competition_event_outputs | no | 0.5075 | 0.0145 | 0.0009 | 0.0149 |
| 44 | no | output | row_context_outputs | no | 0.5323 | 0.0393 | 0.0324 | 0.0555 |
| 45 | no | output | row_local_outputs | no | 0.5224 | 0.0294 | 0.0256 | 0.0422 |
| 46 | no | shape_stats | row_volatility_burst | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 47 | no | shape_stats | shape_drawdown | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 48 | no | shape_stats | shape_post_concentration | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 49 | no | output | source | yes | 0.5274 | 0.0344 | 0.0258 | 0.0473 |
| 50 | no | output | source | yes | 0.5273 | 0.0343 | 0.0247 | 0.0466 |
| 51 | no | output | source | yes | 0.5239 | 0.0309 | 0.0216 | 0.0418 |
| 52 | no | output | source | yes | 0.5308 | 0.0378 | 0.0293 | 0.0525 |
| 53 | no | output | source | yes | 0.5301 | 0.0371 | 0.0318 | 0.0530 |
| 54 | no | output | source | yes | 0.5187 | 0.0257 | 0.0119 | 0.0316 |
| 55 | no | output | competition_event_outputs | no | 0.5075 | 0.0145 | 0.0009 | 0.0149 |
| 56 | no | output | row_context_outputs | no | 0.5323 | 0.0393 | 0.0324 | 0.0555 |
| 57 | no | output | row_local_outputs | no | 0.5224 | 0.0294 | 0.0256 | 0.0422 |
| 58 | no | shape_stats | row_volatility_burst | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 59 | no | shape_stats | shape_drawdown | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 60 | no | shape_stats | shape_post_concentration | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 61 | no | output | source | yes | 0.5274 | 0.0344 | 0.0258 | 0.0473 |
| 62 | no | output | source | yes | 0.5273 | 0.0343 | 0.0247 | 0.0466 |
| 63 | no | output | source | yes | 0.5239 | 0.0309 | 0.0216 | 0.0418 |
| 64 | no | output | source | yes | 0.5308 | 0.0378 | 0.0293 | 0.0525 |
| 65 | no | output | source | yes | 0.5301 | 0.0371 | 0.0318 | 0.0530 |
| 66 | no | output | source | yes | 0.5187 | 0.0257 | 0.0119 | 0.0316 |
| 67 | no | output | competition_event_outputs | no | 0.5075 | 0.0145 | 0.0009 | 0.0149 |
| 68 | no | output | row_context_outputs | no | 0.5323 | 0.0393 | 0.0324 | 0.0555 |
| 69 | no | output | row_local_outputs | no | 0.5224 | 0.0294 | 0.0256 | 0.0422 |
| 70 | no | shape_stats | row_volatility_burst | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 71 | no | shape_stats | shape_drawdown | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 72 | no | shape_stats | shape_post_concentration | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 73 | no | output | source | yes | 0.5274 | 0.0344 | 0.0258 | 0.0473 |
| 74 | no | output | source | yes | 0.5273 | 0.0343 | 0.0247 | 0.0466 |
| 75 | no | output | source | yes | 0.5239 | 0.0309 | 0.0216 | 0.0418 |
| 76 | no | output | source | yes | 0.5308 | 0.0378 | 0.0293 | 0.0525 |
| 77 | no | output | source | yes | 0.5301 | 0.0371 | 0.0318 | 0.0530 |
| 78 | no | output | source | yes | 0.5187 | 0.0257 | 0.0119 | 0.0316 |
| 79 | no | output | competition_event_outputs | no | 0.5075 | 0.0145 | 0.0009 | 0.0149 |
| 80 | no | output | row_context_outputs | no | 0.5323 | 0.0393 | 0.0324 | 0.0555 |
| 81 | no | output | row_local_outputs | no | 0.5224 | 0.0294 | 0.0256 | 0.0422 |
| 82 | no | shape_stats | row_volatility_burst | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 83 | no | shape_stats | shape_drawdown | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 84 | no | shape_stats | shape_post_concentration | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 85 | no | output | source | yes | 0.5274 | 0.0344 | 0.0258 | 0.0473 |
| 86 | no | output | source | yes | 0.5273 | 0.0343 | 0.0247 | 0.0466 |
| 87 | no | output | source | yes | 0.5239 | 0.0309 | 0.0216 | 0.0418 |
| 88 | no | output | source | yes | 0.5308 | 0.0378 | 0.0293 | 0.0525 |
| 89 | no | output | source | yes | 0.5301 | 0.0371 | 0.0318 | 0.0530 |
| 90 | no | output | source | yes | 0.5187 | 0.0257 | 0.0119 | 0.0316 |
| 91 | no | output | competition_event_outputs | no | 0.5075 | 0.0145 | 0.0009 | 0.0149 |
| 92 | no | output | row_context_outputs | no | 0.5323 | 0.0393 | 0.0324 | 0.0555 |
| 93 | no | output | row_local_outputs | no | 0.5224 | 0.0294 | 0.0256 | 0.0422 |
| 94 | no | shape_stats | row_volatility_burst | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 95 | no | shape_stats | shape_drawdown | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
| 96 | no | shape_stats | shape_post_concentration | no | 0.5231 | 0.0301 | 0.0222 | 0.0412 |
