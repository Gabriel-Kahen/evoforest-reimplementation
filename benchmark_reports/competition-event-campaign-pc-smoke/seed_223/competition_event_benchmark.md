# Competition Id-Level Structural-Break Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`1000`, max_samples=`1000`.

Split: `group_stratified_random` by `sample_id`, sizes={'train': 600, 'validation': 200, 'test': 200}, group overlaps={'train_validation': 0, 'train_test': 0, 'validation_test': 0}.

Reduced test accessed: `False`.

Baseline validation AUC: `0.5865`; best evolved graph validation AUC: `0.5894`; delta: `0.0029`.

Best ensemble: `baseline_plus_best_graph_blend` validation AUC `0.5919`, delta vs baseline `0.0054`.

| Step | Accepted | Node | Primitive | Source | CV Train AUC | Holdout Val AUC | Delta vs Baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CHECK | segment_stats | row_recent_change | no | 0.5520 | 0.5568 | -0.0297 |
| 2 | CHECK | segment_stats | segment_late_shift | no | 0.5520 | 0.5568 | -0.0297 |
| 3 | PASS | output | source | yes | 0.5624 | 0.5835 | -0.0030 |
| 4 | CHECK | output | source | yes | 0.5690 | 0.5757 | -0.0108 |
| 5 | CHECK | output | competition_event_outputs | no | 0.5693 | 0.5458 | -0.0407 |
| 6 | CHECK | output | interaction_outputs | no | 0.5540 | 0.5784 | -0.0081 |
| 7 | CHECK | output | row_context_outputs | no | 0.5624 | 0.5835 | -0.0030 |
| 8 | CHECK | output | row_local_outputs | no | 0.5425 | 0.5779 | -0.0086 |
| 9 | CHECK | trend_stats | trend_late_window | no | 0.5624 | 0.5835 | -0.0030 |
| 10 | CHECK | ridge_w | late_energy_weight | no | 0.5624 | 0.5835 | -0.0030 |
| 11 | CHECK | shape_stats | row_cusum_local | no | 0.5624 | 0.5835 | -0.0030 |
| 12 | PASS | shape_stats | row_volatility_burst | no | 0.5649 | 0.5855 | -0.0010 |
| 13 | CHECK | shape_stats | shape_post_concentration | no | 0.5649 | 0.5855 | -0.0010 |
| 14 | CHECK | segment_stats | row_recent_change | no | 0.5624 | 0.5835 | -0.0030 |
| 15 | CHECK | segment_stats | segment_late_shift | no | 0.5624 | 0.5835 | -0.0030 |
| 16 | PASS | output | source | yes | 0.5685 | 0.5894 | 0.0029 |
| 17 | CHECK | output | interaction_outputs | no | 0.5615 | 0.5890 | 0.0025 |
| 18 | CHECK | output | row_context_outputs | no | 0.5685 | 0.5894 | 0.0029 |
| 19 | CHECK | output | row_local_outputs | no | 0.5547 | 0.5812 | -0.0053 |
| 20 | CHECK | trend_stats | trend_late_window | no | 0.5685 | 0.5894 | 0.0029 |
| 21 | CHECK | ridge_w | late_energy_weight | no | 0.5667 | 0.5893 | 0.0028 |
| 22 | CHECK | shape_stats | row_cusum_local | no | 0.5685 | 0.5894 | 0.0029 |
| 23 | CHECK | shape_stats | shape_drawdown | no | 0.5685 | 0.5894 | 0.0029 |
| 24 | CHECK | shape_stats | shape_post_concentration | no | 0.5685 | 0.5894 | 0.0029 |
