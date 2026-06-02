# Competition Row-Level Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-row` from `/home/gabe/evoforest-competition-data`, rows=`31729`, ids=`1000`, max_ids=`1000`, max_rows_per_id=`32`.

Split: `group_stratified_random` by `sample_id`, sizes={'train': 19032, 'validation': 6330, 'test': 6367}, group overlaps={'train_validation': 0, 'train_test': 0, 'validation_test': 0}.

Reduced test accessed: `False`.

Baseline validation AUC: `0.6124`; evolved validation AUC: `0.5872`; delta: `-0.0253`; meaningful margin met: `False`.

| Step | Accepted | Node | Primitive | CV Train AUC | Holdout Val AUC | Delta vs Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | CHECK | segment_stats | row_recent_change | 0.5815 | 0.5577 | -0.0547 |
| 2 | CHECK | segment_stats | segment_late_shift | 0.5815 | 0.5577 | -0.0547 |
| 3 | CHECK | output | competition_event_outputs | 0.5646 | 0.5531 | -0.0593 |
| 4 | CHECK | output | interaction_outputs | 0.5921 | 0.5519 | -0.0605 |
| 5 | PASS | output | row_context_outputs | 0.5739 | 0.5714 | -0.0410 |
| 6 | CHECK | trend_stats | trend_late_window | 0.5715 | 0.5714 | -0.0410 |
| 7 | CHECK | ridge_w | late_energy_weight | 0.5739 | 0.5714 | -0.0410 |
| 8 | CHECK | shape_stats | row_cusum_local | 0.5739 | 0.5714 | -0.0410 |
| 9 | PASS | shape_stats | row_volatility_burst | 0.5959 | 0.5872 | -0.0253 |
| 10 | CHECK | shape_stats | shape_post_concentration | 0.5946 | 0.5717 | -0.0407 |
| 11 | CHECK | segment_stats | row_recent_change | 0.5959 | 0.5872 | -0.0253 |
| 12 | CHECK | segment_stats | segment_late_shift | 0.5959 | 0.5872 | -0.0253 |
