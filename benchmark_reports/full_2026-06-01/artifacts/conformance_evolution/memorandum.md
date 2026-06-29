# Evolution Memorandum

[OUTCOME HISTORY]
- ACCEPTED: step=1 score=0.747782 best=0.747782

[STATE]
- Best score: 0.747782; config={'activation': 'clipped_linear', 'ridge_g': 'huber', 'ridge_w': 'uniform', 'segment_stats': 'basic', 'shape_stats': 'cusum', 'trend_stats': 'linear'}.
- Configuration search: 4/72 evaluated, capped=True.
- Representation: effective_rank=9.1443, mean_max_corr=0.9248, global_ridge_score=0.799921.
- Cache: hits=28, entries=20, key=ancestor_conditioned_subpath.
- Dominant feature: output.raw_concat.std_log_ratio_abs imp=0.1224, target_align=0.4859, resid=0.0057.
- Dominant subnode: segment_stats.basic imp=1.0000, features=21.
- Dominant alternative: segment_stats.basic age=2, participations=2.

[WHAT WORKS]
- ACCEPTED: step=1 score=0.747782 best=0.747782
- Productive substructure: segment_stats.basic aggregates imp=1.0000.
- Productive substructure: trend_stats.linear aggregates imp=1.0000.
- Productive substructure: shape_stats.cusum aggregates imp=1.0000.

[WHAT FAILED]
- Risky feature: output.raw_concat.cusum_peak redundancy=0.9984, stability=4.0294.
- Risky feature: output.projection.global_projection redundancy=0.9984, stability=4.1372.
- Risky feature: output.raw_concat.post_slope redundancy=0.9987, stability=2.5597.
- Risky feature: output.activated.clipped_linear_pre_slope redundancy=1.0000, stability=17.0249.

[ERROR LOG]
- No runtime errors recorded.
