# Search Dynamics Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported task score.

Seed: `31`

Initial score: `0.6931`; final score: `0.7473`; global-best versions: `7`; run artifacts: `benchmark_reports/full_2026-06-01/artifacts/search_dynamics_run`

Final graph: `9` nodes, `21` alternatives, `6` output alternatives, `192` total configs.

| Step | Accepted | Score | Best | Added | Salvaged |
| --- | --- | --- | --- | --- | --- |
| 1 | PASS | 0.6931 | 0.6931 | segment_stats.row_recent_change_mutation_1 | none |
| 2 | PASS | 0.7030 | 0.7030 | output.event_detection_output_mutation_2 | none |
| 3 | PASS | 0.7473 | 0.7473 | output.row_baseline_output_mutation_3 | none |
| 4 | CHECK | 0.3666 | 0.7473 | output.row_multiscale_tail_output_mutation_4 | none |
| 5 | PASS | 0.7473 | 0.7473 | output.row_time_basis_output_mutation_5 | none |
| 6 | PASS | 0.7473 | 0.7473 | trend_stats.late_trend_mutation_6 | none |
| 7 | PASS | 0.7473 | 0.7473 | segment_stats.late_shift_mutation_7 | none |
| 8 | CHECK | 0.7416 | 0.7473 | shape_stats.row_volatility_burst_mutation_8 | none |
