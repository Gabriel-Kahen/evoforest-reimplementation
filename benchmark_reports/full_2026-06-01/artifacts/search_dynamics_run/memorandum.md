# Evolution Memorandum

[OUTCOME HISTORY]
- ACCEPTED: step=1 score=0.693111 best=0.693111
- ACCEPTED: step=2 score=0.703016 best=0.703016
- ACCEPTED: step=3 score=0.747283 best=0.747283
- REJECTED: step=4 score=0.366551 best=0.747283
- ACCEPTED: step=5 score=0.747283 best=0.747283
- ACCEPTED: step=6 score=0.747283 best=0.747283
- ACCEPTED: step=7 score=0.747283 best=0.747283
- REJECTED: step=8 score=0.741578 best=0.747283

[STATE]
- Best score: 0.747283; config={'activation': 'identity', 'ridge_g': 'identity', 'ridge_w': 'uniform', 'segment_stats': 'row_recent_change_mutation_1', 'shape_stats': 'spectral', 'trend_stats': 'linear'}.
- Configuration search: 16/192 evaluated, capped=True.
- Representation: effective_rank=19.0438, mean_max_corr=0.7377, global_ridge_score=0.892689.
- Cache: hits=186, entries=50, key=ancestor_conditioned_subpath.
- Dominant feature: output.row_baseline_output_mutation_3.last_value imp=0.1435, target_align=0.4898, resid=-0.0353.
- Dominant subnode: output.row_baseline_output_mutation_3 imp=0.4962, features=21.
- Dominant alternative: segment_stats.basic age=7, participations=3.

[WHAT WORKS]
- ACCEPTED: step=3 score=0.747283 best=0.747283
- ACCEPTED: step=5 score=0.747283 best=0.747283
- ACCEPTED: step=6 score=0.747283 best=0.747283
- ACCEPTED: step=7 score=0.747283 best=0.747283
- Productive substructure: output.row_baseline_output_mutation_3 aggregates imp=0.4962.
- Productive substructure: segment_stats.row_recent_change_mutation_1 aggregates imp=0.4112.
- Productive substructure: trend_stats.linear aggregates imp=0.4112.

[WHAT FAILED]
- REJECTED: step=4 score=0.366551 best=0.747283
- REJECTED: step=8 score=0.741578 best=0.747283
- Risky feature: output.row_baseline_output_mutation_3.last_value redundancy=0.9950, stability=2.8051.
- Risky feature: output.activated.identity_recent_tail_drift redundancy=1.0000, stability=22.6872.
- Risky feature: output.row_baseline_output_mutation_3.recent_tail_drift redundancy=1.0000, stability=22.6872.
- Risky feature: output.raw_concat.recent_tail_drift redundancy=1.0000, stability=22.6872.

[ERROR LOG]
- No runtime errors recorded.
