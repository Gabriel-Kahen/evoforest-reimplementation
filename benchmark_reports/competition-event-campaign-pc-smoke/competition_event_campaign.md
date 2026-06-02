# Competition Id-Level Campaign

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Data dir: `/home/gabe/evoforest-competition-data`

Config: `{'series_length': 160, 'max_samples': 1000, 'steps': 24, 'folds': 3, 'max_configurations': 32, 'include_source_mutations': True}`

Best ensemble seed: `223`; validation AUC `0.5919`; delta vs baseline `0.0054`.

Reduced test accessed: `False`.

| Seed | Baseline Val AUC | Evolved Val AUC | Evolved Delta | Best Ensemble | Ensemble Val AUC | Ensemble Delta | Accepted | Source Candidates | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 211 | 0.4914 | 0.5498 | 0.0584 | top_2_graph_archive | 0.5498 | 0.0584 | 5 | 4 | 140.8871861000007 |
| 223 | 0.5865 | 0.5894 | 0.0029 | baseline_plus_best_graph_blend | 0.5919 | 0.0054 | 3 | 3 | 71.02356590000272 |
