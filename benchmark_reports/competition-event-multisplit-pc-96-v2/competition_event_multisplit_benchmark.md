# Competition Multi-Split Structural-Break Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`1000`, max_samples=`1000`.

Config: `{'steps': 96, 'folds': 3, 'max_configurations': 32, 'split_seeds': [211, 223, 307], 'include_source_mutations': True, 'min_objective_improvement': 0.0001, 'stability_weight': 0.5, 'prune_tolerance': 0.001, 'objective': 'mean_delta_vs_baseline + stability_weight * min_delta_vs_baseline'}`

Reduced test accessed: `False`.

Pruned consensus graph mean validation AUC `0.5323`, mean delta `0.0393`, minimum delta `0.0324`, objective `0.0555`.

Best archive/OOF ensemble `top_2_archive` mean validation AUC `0.5307`, mean delta `0.0377`, minimum delta `0.0323`.

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
| 10 | no | output | adia_structural_break_baseline_outputs | no | 0.5076 | 0.0146 | 0.0060 | 0.0176 |
| 11 | no | output | competition_event_outputs | no | 0.4982 | 0.0053 | -0.0020 | 0.0042 |
| 12 | no | output | interaction_outputs | no | 0.5067 | 0.0137 | 0.0060 | 0.0167 |
| 13 | no | output | row_context_outputs | no | 0.5117 | 0.0187 | 0.0131 | 0.0253 |
| 14 | no | output | row_local_outputs | no | 0.5093 | 0.0163 | 0.0142 | 0.0234 |
| 15 | yes | trend_stats | trend_late_window | no | 0.5143 | 0.0213 | 0.0099 | 0.0262 |
| 16 | no | shape_stats | row_cusum_local | no | 0.5085 | 0.0155 | 0.0099 | 0.0204 |
| 17 | no | shape_stats | row_volatility_burst | no | 0.5085 | 0.0155 | 0.0099 | 0.0204 |
| 18 | no | shape_stats | shape_drawdown | no | 0.5085 | 0.0155 | 0.0099 | 0.0204 |
| 19 | no | shape_stats | shape_post_concentration | no | 0.5085 | 0.0155 | 0.0099 | 0.0204 |
| 20 | no | segment_stats | segment_late_shift | no | 0.5143 | 0.0213 | 0.0099 | 0.0262 |
| 21 | no | output | source | yes | 0.5033 | 0.0103 | 0.0065 | 0.0135 |
| 22 | yes | output | source | yes | 0.5147 | 0.0217 | 0.0110 | 0.0272 |
| 23 | yes | ridge_w | late_energy_weight | no | 0.5149 | 0.0219 | 0.0110 | 0.0274 |
| 24 | yes | output | source | yes | 0.5177 | 0.0247 | 0.0120 | 0.0307 |
| 25 | no | output | competition_event_outputs | no | 0.5011 | 0.0081 | 0.0000 | 0.0081 |
| 26 | yes | output | interaction_outputs | no | 0.5160 | 0.0230 | 0.0207 | 0.0333 |
| 27 | no | shape_stats | row_cusum_local | no | 0.5139 | 0.0209 | 0.0100 | 0.0259 |
| 28 | no | shape_stats | row_volatility_burst | no | 0.5139 | 0.0209 | 0.0100 | 0.0259 |
| 29 | no | shape_stats | shape_drawdown | no | 0.5145 | 0.0215 | 0.0100 | 0.0265 |
| 30 | no | shape_stats | shape_post_concentration | no | 0.5172 | 0.0243 | 0.0100 | 0.0293 |
| 31 | yes | segment_stats | segment_late_shift | no | 0.5175 | 0.0245 | 0.0207 | 0.0348 |
| 32 | no | output | source | yes | 0.5143 | 0.0213 | 0.0185 | 0.0306 |
| 33 | no | output | source | yes | 0.5107 | 0.0177 | 0.0147 | 0.0250 |
| 34 | no | output | source | yes | 0.5074 | 0.0144 | 0.0036 | 0.0162 |
| 35 | no | output | adia_structural_break_baseline_outputs | no | 0.5070 | 0.0140 | 0.0091 | 0.0186 |
| 36 | no | output | competition_event_outputs | no | 0.4999 | 0.0069 | -0.0088 | 0.0025 |
| 37 | no | output | row_context_outputs | no | 0.5175 | 0.0245 | 0.0207 | 0.0348 |
| 38 | no | output | row_local_outputs | no | 0.5142 | 0.0212 | 0.0119 | 0.0271 |
| 39 | yes | shape_stats | row_cusum_local | no | 0.5218 | 0.0288 | 0.0221 | 0.0399 |
| 40 | no | output | source | yes | 0.5201 | 0.0271 | 0.0232 | 0.0387 |
| 41 | no | output | source | yes | 0.5072 | 0.0142 | -0.0025 | 0.0129 |
| 42 | no | output | source | yes | 0.5082 | 0.0152 | 0.0073 | 0.0189 |
| 43 | yes | output | source | yes | 0.5301 | 0.0371 | 0.0318 | 0.0530 |
| 44 | no | output | row_context_outputs | no | 0.5218 | 0.0288 | 0.0221 | 0.0399 |
| 45 | no | output | row_local_outputs | no | 0.5183 | 0.0253 | 0.0134 | 0.0320 |
| 46 | no | shape_stats | row_volatility_burst | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 47 | no | shape_stats | shape_drawdown | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 48 | no | shape_stats | shape_post_concentration | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 49 | no | output | source | yes | 0.5201 | 0.0271 | 0.0232 | 0.0387 |
| 50 | no | output | source | yes | 0.5072 | 0.0142 | -0.0025 | 0.0129 |
| 51 | no | output | source | yes | 0.5082 | 0.0152 | 0.0073 | 0.0189 |
| 52 | no | output | source | yes | 0.5191 | 0.0261 | 0.0181 | 0.0351 |
| 53 | no | output | source | yes | 0.5161 | 0.0231 | 0.0100 | 0.0281 |
| 54 | no | output | adia_structural_break_baseline_outputs | no | 0.5071 | 0.0141 | 0.0066 | 0.0174 |
| 55 | no | output | competition_event_outputs | no | 0.5028 | 0.0098 | -0.0080 | 0.0058 |
| 56 | no | output | row_context_outputs | no | 0.5218 | 0.0288 | 0.0221 | 0.0399 |
| 57 | no | output | row_local_outputs | no | 0.5183 | 0.0253 | 0.0134 | 0.0320 |
| 58 | no | shape_stats | row_volatility_burst | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 59 | no | shape_stats | shape_drawdown | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 60 | no | shape_stats | shape_post_concentration | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 61 | no | output | source | yes | 0.5201 | 0.0271 | 0.0232 | 0.0387 |
| 62 | no | output | source | yes | 0.5072 | 0.0142 | -0.0025 | 0.0129 |
| 63 | no | output | source | yes | 0.5082 | 0.0152 | 0.0073 | 0.0189 |
| 64 | no | output | source | yes | 0.5191 | 0.0261 | 0.0181 | 0.0351 |
| 65 | no | output | source | yes | 0.5161 | 0.0231 | 0.0100 | 0.0281 |
| 66 | no | output | adia_structural_break_baseline_outputs | no | 0.5071 | 0.0141 | 0.0066 | 0.0174 |
| 67 | no | output | competition_event_outputs | no | 0.5028 | 0.0098 | -0.0080 | 0.0058 |
| 68 | no | output | row_context_outputs | no | 0.5218 | 0.0288 | 0.0221 | 0.0399 |
| 69 | no | output | row_local_outputs | no | 0.5183 | 0.0253 | 0.0134 | 0.0320 |
| 70 | no | shape_stats | row_volatility_burst | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 71 | no | shape_stats | shape_drawdown | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 72 | no | shape_stats | shape_post_concentration | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 73 | no | output | source | yes | 0.5201 | 0.0271 | 0.0232 | 0.0387 |
| 74 | no | output | source | yes | 0.5072 | 0.0142 | -0.0025 | 0.0129 |
| 75 | no | output | source | yes | 0.5082 | 0.0152 | 0.0073 | 0.0189 |
| 76 | no | output | source | yes | 0.5191 | 0.0261 | 0.0181 | 0.0351 |
| 77 | no | output | source | yes | 0.5161 | 0.0231 | 0.0100 | 0.0281 |
| 78 | no | output | adia_structural_break_baseline_outputs | no | 0.5071 | 0.0141 | 0.0066 | 0.0174 |
| 79 | no | output | competition_event_outputs | no | 0.5028 | 0.0098 | -0.0080 | 0.0058 |
| 80 | no | output | row_context_outputs | no | 0.5218 | 0.0288 | 0.0221 | 0.0399 |
| 81 | no | output | row_local_outputs | no | 0.5183 | 0.0253 | 0.0134 | 0.0320 |
| 82 | no | shape_stats | row_volatility_burst | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 83 | no | shape_stats | shape_drawdown | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 84 | no | shape_stats | shape_post_concentration | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 85 | no | output | source | yes | 0.5201 | 0.0271 | 0.0232 | 0.0387 |
| 86 | no | output | source | yes | 0.5072 | 0.0142 | -0.0025 | 0.0129 |
| 87 | no | output | source | yes | 0.5082 | 0.0152 | 0.0073 | 0.0189 |
| 88 | no | output | source | yes | 0.5191 | 0.0261 | 0.0181 | 0.0351 |
| 89 | no | output | source | yes | 0.5161 | 0.0231 | 0.0100 | 0.0281 |
| 90 | no | output | adia_structural_break_baseline_outputs | no | 0.5071 | 0.0141 | 0.0066 | 0.0174 |
| 91 | no | output | competition_event_outputs | no | 0.5028 | 0.0098 | -0.0080 | 0.0058 |
| 92 | no | output | row_context_outputs | no | 0.5218 | 0.0288 | 0.0221 | 0.0399 |
| 93 | no | output | row_local_outputs | no | 0.5183 | 0.0253 | 0.0134 | 0.0320 |
| 94 | no | shape_stats | row_volatility_burst | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 95 | no | shape_stats | shape_drawdown | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
| 96 | no | shape_stats | shape_post_concentration | no | 0.5184 | 0.0254 | 0.0117 | 0.0313 |
