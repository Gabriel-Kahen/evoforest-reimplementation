# Competition Id-Level Structural-Break Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`2000`, max_samples=`2000`.

Split: `group_stratified_random` by `sample_id`, sizes={'train': 1200, 'validation': 400, 'test': 400}, group overlaps={'train_validation': 0, 'train_test': 0, 'validation_test': 0}.

Reduced test accessed: `False`.

Baseline validation AUC: `0.4706`; best evolved graph validation AUC: `0.4765`; delta: `0.0059`.

Best ensemble: `top_2_graph_archive` validation AUC `0.4765`, delta vs baseline `0.0059`.

| Step | Accepted | Node | Primitive | Source | CV Train AUC | Holdout Val AUC | Delta vs Baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CHECK | segment_stats | row_recent_change | no | 0.5386 | 0.4603 | -0.0104 |
| 2 | CHECK | segment_stats | segment_late_shift | no | 0.5386 | 0.4603 | -0.0104 |
| 3 | PASS | output | source | yes | 0.5562 | 0.4642 | -0.0065 |
| 4 | PASS | output | source | yes | 0.5553 | 0.4668 | -0.0038 |
| 5 | CHECK | output | interaction_outputs | no | 0.5495 | 0.4631 | -0.0076 |
| 6 | CHECK | output | row_context_outputs | no | 0.5553 | 0.4668 | -0.0038 |
| 7 | PASS | output | row_local_outputs | no | 0.5474 | 0.4673 | -0.0033 |
| 8 | CHECK | ridge_w | late_energy_weight | no | 0.5492 | 0.4673 | -0.0033 |
| 9 | CHECK | shape_stats | row_cusum_local | no | 0.5474 | 0.4673 | -0.0033 |
| 10 | PASS | shape_stats | row_volatility_burst | no | 0.5495 | 0.4731 | 0.0025 |
| 11 | CHECK | shape_stats | shape_post_concentration | no | 0.5495 | 0.4731 | 0.0025 |
| 12 | CHECK | segment_stats | row_recent_change | no | 0.5495 | 0.4731 | 0.0025 |
| 13 | CHECK | segment_stats | segment_late_shift | no | 0.5495 | 0.4731 | 0.0025 |
| 14 | PASS | output | source | yes | 0.5457 | 0.4765 | 0.0059 |
| 15 | CHECK | output | row_context_outputs | no | 0.5457 | 0.4765 | 0.0059 |
| 16 | CHECK | trend_stats | trend_late_window | no | 0.5481 | 0.4720 | 0.0014 |
| 17 | CHECK | ridge_w | late_energy_weight | no | 0.5476 | 0.4765 | 0.0059 |
| 18 | CHECK | shape_stats | row_cusum_local | no | 0.5457 | 0.4765 | 0.0059 |
| 19 | CHECK | shape_stats | shape_drawdown | no | 0.5457 | 0.4765 | 0.0059 |
| 20 | CHECK | shape_stats | shape_post_concentration | no | 0.5457 | 0.4765 | 0.0059 |
| 21 | CHECK | segment_stats | row_recent_change | no | 0.5457 | 0.4765 | 0.0059 |
| 22 | CHECK | segment_stats | segment_late_shift | no | 0.5457 | 0.4765 | 0.0059 |
| 23 | CHECK | output | competition_event_outputs | no | 0.5404 | 0.4757 | 0.0051 |
| 24 | CHECK | output | interaction_outputs | no | 0.5425 | 0.4744 | 0.0038 |
| 25 | CHECK | output | row_context_outputs | no | 0.5457 | 0.4765 | 0.0059 |
| 26 | CHECK | trend_stats | trend_late_window | no | 0.5481 | 0.4720 | 0.0014 |
| 27 | CHECK | ridge_w | late_energy_weight | no | 0.5476 | 0.4765 | 0.0059 |
| 28 | CHECK | shape_stats | row_cusum_local | no | 0.5457 | 0.4765 | 0.0059 |
| 29 | CHECK | shape_stats | shape_drawdown | no | 0.5457 | 0.4765 | 0.0059 |
| 30 | CHECK | shape_stats | shape_post_concentration | no | 0.5457 | 0.4765 | 0.0059 |
| 31 | CHECK | segment_stats | row_recent_change | no | 0.5457 | 0.4765 | 0.0059 |
| 32 | CHECK | segment_stats | segment_late_shift | no | 0.5457 | 0.4765 | 0.0059 |
| 33 | CHECK | output | competition_event_outputs | no | 0.5404 | 0.4757 | 0.0051 |
| 34 | CHECK | output | interaction_outputs | no | 0.5425 | 0.4744 | 0.0038 |
| 35 | CHECK | output | row_context_outputs | no | 0.5457 | 0.4765 | 0.0059 |
| 36 | CHECK | trend_stats | trend_late_window | no | 0.5481 | 0.4720 | 0.0014 |
| 37 | CHECK | ridge_w | late_energy_weight | no | 0.5476 | 0.4765 | 0.0059 |
| 38 | CHECK | shape_stats | row_cusum_local | no | 0.5457 | 0.4765 | 0.0059 |
| 39 | CHECK | shape_stats | shape_drawdown | no | 0.5457 | 0.4765 | 0.0059 |
| 40 | CHECK | shape_stats | shape_post_concentration | no | 0.5457 | 0.4765 | 0.0059 |
| 41 | CHECK | segment_stats | row_recent_change | no | 0.5457 | 0.4765 | 0.0059 |
| 42 | CHECK | segment_stats | segment_late_shift | no | 0.5457 | 0.4765 | 0.0059 |
| 43 | CHECK | output | competition_event_outputs | no | 0.5404 | 0.4757 | 0.0051 |
| 44 | CHECK | output | interaction_outputs | no | 0.5425 | 0.4744 | 0.0038 |
| 45 | CHECK | output | row_context_outputs | no | 0.5457 | 0.4765 | 0.0059 |
| 46 | CHECK | trend_stats | trend_late_window | no | 0.5481 | 0.4720 | 0.0014 |
| 47 | CHECK | ridge_w | late_energy_weight | no | 0.5476 | 0.4765 | 0.0059 |
| 48 | CHECK | shape_stats | row_cusum_local | no | 0.5457 | 0.4765 | 0.0059 |
