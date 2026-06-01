# Runtime Scaling Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Seed: `37`

Repeats per scenario: `3`

| Family | Setting | Median Seconds | AUC | Configs Eval | Configs Total | Features | Cache Hits | Cache Misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| configuration_cap | max_configurations=1 | 0.01129 | 0.9957 | 1 | 48 | 21 | 6 | 9 |
| configuration_cap | max_configurations=8 | 0.11449 | 0.9957 | 8 | 48 | 21 | 59 | 23 |
| configuration_cap | max_configurations=16 | 0.27098 | 0.9991 | 16 | 48 | 21 | 111 | 29 |
| configuration_cap | max_configurations=32 | 0.65658 | 0.9991 | 32 | 48 | 21 | 200 | 32 |
| dataset_scale | n_series=48, length=64 | 0.29072 | 0.9618 | 16 | 48 | 21 | 111 | 29 |
| dataset_scale | n_series=96, length=96 | 0.40401 | 0.9991 | 16 | 48 | 21 | 111 | 29 |
| dataset_scale | n_series=144, length=128 | 0.32899 | 0.9992 | 16 | 48 | 21 | 111 | 29 |
| output_feature_growth | extra_output_alternatives=0 | 0.23903 | 0.9991 | 16 | 48 | 21 | 111 | 29 |
| output_feature_growth | extra_output_alternatives=4 | 0.25190 | 0.9991 | 16 | 48 | 25 | 207 | 45 |
| output_feature_growth | extra_output_alternatives=8 | 0.28886 | 0.9991 | 16 | 48 | 29 | 303 | 61 |
| output_feature_growth | extra_output_alternatives=12 | 0.29871 | 0.9991 | 16 | 48 | 33 | 399 | 77 |
