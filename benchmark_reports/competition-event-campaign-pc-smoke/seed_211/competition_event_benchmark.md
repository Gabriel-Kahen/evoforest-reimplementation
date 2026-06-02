# Competition Id-Level Structural-Break Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`1000`, max_samples=`1000`.

Split: `group_stratified_random` by `sample_id`, sizes={'train': 600, 'validation': 200, 'test': 200}, group overlaps={'train_validation': 0, 'train_test': 0, 'validation_test': 0}.

Reduced test accessed: `False`.

Baseline validation AUC: `0.4914`; best evolved graph validation AUC: `0.5498`; delta: `0.0584`.

Best ensemble: `top_2_graph_archive` validation AUC `0.5498`, delta vs baseline `0.0584`.

| Step | Accepted | Node | Primitive | Source | CV Train AUC | Holdout Val AUC | Delta vs Baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CHECK | segment_stats | row_recent_change | no | 0.5638 | 0.5203 | 0.0288 |
| 2 | CHECK | segment_stats | segment_late_shift | no | 0.5638 | 0.5203 | 0.0288 |
| 3 | PASS | output | source | yes | 0.5636 | 0.5254 | 0.0339 |
| 4 | PASS | output | source | yes | 0.5541 | 0.5323 | 0.0408 |
| 5 | PASS | output | interaction_outputs | no | 0.5522 | 0.5348 | 0.0433 |
| 6 | PASS | output | row_local_outputs | no | 0.5515 | 0.5479 | 0.0565 |
| 7 | CHECK | ridge_w | late_energy_weight | no | 0.5526 | 0.5479 | 0.0565 |
| 8 | CHECK | shape_stats | row_cusum_local | no | 0.5515 | 0.5479 | 0.0565 |
| 9 | CHECK | shape_stats | row_volatility_burst | no | 0.5515 | 0.5479 | 0.0565 |
| 10 | CHECK | shape_stats | shape_drawdown | no | 0.5515 | 0.5479 | 0.0565 |
| 11 | CHECK | shape_stats | shape_post_concentration | no | 0.5515 | 0.5479 | 0.0565 |
| 12 | CHECK | segment_stats | row_recent_change | no | 0.5515 | 0.5479 | 0.0565 |
| 13 | CHECK | segment_stats | segment_late_shift | no | 0.5515 | 0.5479 | 0.0565 |
| 14 | CHECK | output | source | yes | 0.5589 | 0.5360 | 0.0445 |
| 15 | PASS | output | competition_event_outputs | no | 0.5407 | 0.5498 | 0.0584 |
| 16 | CHECK | ridge_w | late_energy_weight | no | 0.5379 | 0.5498 | 0.0584 |
| 17 | CHECK | shape_stats | row_cusum_local | no | 0.5407 | 0.5498 | 0.0584 |
| 18 | CHECK | shape_stats | row_volatility_burst | no | 0.5407 | 0.5498 | 0.0584 |
| 19 | CHECK | shape_stats | shape_drawdown | no | 0.5407 | 0.5498 | 0.0584 |
| 20 | CHECK | shape_stats | shape_post_concentration | no | 0.5407 | 0.5498 | 0.0584 |
| 21 | CHECK | segment_stats | row_recent_change | no | 0.5407 | 0.5498 | 0.0584 |
| 22 | CHECK | segment_stats | segment_late_shift | no | 0.5407 | 0.5498 | 0.0584 |
| 23 | CHECK | output | source | yes | 0.5476 | 0.5381 | 0.0466 |
| 24 | CHECK | output | row_context_outputs | no | 0.5407 | 0.5498 | 0.0584 |
