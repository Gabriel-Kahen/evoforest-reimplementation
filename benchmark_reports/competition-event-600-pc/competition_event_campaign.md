# Competition Id-Level Campaign

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Data dir: `/home/gabe/evoforest-competition-data`

Config: `{'folds': 3, 'include_source_mutations': True, 'max_configurations': 32, 'max_samples': 1000, 'resume_existing': True, 'series_length': 160, 'steps': 600}`

Best ensemble delta seed: `211`; validation AUC `0.5511`; delta vs baseline `0.0597`.

Mean evolved delta vs baseline: `0.0216`; mean ensemble delta vs baseline: `0.0236`; minimum ensemble delta: `-0.0016`.

Reduced test accessed: `False`.

| Seed | Baseline Val AUC | Evolved Val AUC | Evolved Delta | Best Ensemble | Ensemble Val AUC | Ensemble Delta | Accepted | Source Candidates | Resumed | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 211 | 0.4914 | 0.5498 | 0.0584 | top_2_graph_archive | 0.5511 | 0.0597 | 6 | 184 | yes | 0.005367800000385614 |
| 223 | 0.5865 | 0.5993 | 0.0128 | top_2_graph_archive | 0.5993 | 0.0128 | 2 | 178 | no | 1706.3223768000025 |
| 307 | 0.6039 | 0.5974 | -0.0065 | baseline_plus_best_graph_blend | 0.6023 | -0.0016 | 3 | 152 | no | 2473.7764398 |
