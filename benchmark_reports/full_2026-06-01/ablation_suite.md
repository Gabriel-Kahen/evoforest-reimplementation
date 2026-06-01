# Ablation Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Seed: `29`

Full graph AUC: `0.9969`

Passed: `7/7`

| Ablation | Status | AUC | Delta | Configs Eval | Configs Total | Features |
| --- | --- | --- | --- | --- | --- | --- |
| full | PASS | 0.9969 | 0.0000 | 24 | 48 | 21 |
| default_path_only | PASS | 0.9962 | -0.0007 | 1 | 1 | 21 |
| raw_output_only | PASS | 0.9976 | 0.0007 | 16 | 16 | 10 |
| no_callable_choice | PASS | 0.9969 | 0.0000 | 16 | 16 | 21 |
| no_fitting_choice | PASS | 0.9969 | 0.0000 | 12 | 12 | 21 |
| no_global_projection | PASS | 0.9973 | 0.0003 | 24 | 48 | 20 |
| no_spectral_shape | PASS | 0.9962 | -0.0007 | 24 | 24 | 21 |
