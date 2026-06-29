# Runtime Scaling Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported task score.

Seed: `37`

Repeats per scenario: `3`

| Family | Setting | Median Seconds | Score | Configs Eval | Configs Total | Features | Cache Hits | Cache Misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| configuration_cap | max_configurations=1 | 0.01018 | 0.7487 | 1 | 48 | 21 | 6 | 9 |
| configuration_cap | max_configurations=8 | 0.08385 | 0.7487 | 8 | 48 | 21 | 59 | 23 |
| configuration_cap | max_configurations=16 | 0.17033 | 0.7531 | 16 | 48 | 21 | 111 | 29 |
| configuration_cap | max_configurations=32 | 0.33621 | 0.7531 | 32 | 48 | 21 | 200 | 32 |
| dataset_scale | n_series=48, length=64 | 0.14899 | 0.5045 | 16 | 48 | 21 | 111 | 29 |
| dataset_scale | n_series=96, length=96 | 0.17143 | 0.7531 | 16 | 48 | 21 | 111 | 29 |
| dataset_scale | n_series=144, length=128 | 0.18349 | 0.8300 | 16 | 48 | 21 | 111 | 29 |
| output_feature_growth | extra_output_alternatives=0 | 0.17002 | 0.7531 | 16 | 48 | 21 | 111 | 29 |
| output_feature_growth | extra_output_alternatives=4 | 0.19065 | 0.7463 | 16 | 48 | 25 | 207 | 45 |
| output_feature_growth | extra_output_alternatives=8 | 0.21219 | 0.7377 | 16 | 48 | 29 | 303 | 61 |
| output_feature_growth | extra_output_alternatives=12 | 0.23742 | 0.7376 | 16 | 48 | 33 | 399 | 77 |
