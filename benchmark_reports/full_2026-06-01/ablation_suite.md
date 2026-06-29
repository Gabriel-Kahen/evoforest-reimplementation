# Ablation Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported task score.

Seed: `29`

Full graph score: `0.7667`

Passed: `7/7`

| Ablation | Status | Score | Delta | Configs Eval | Configs Total | Features |
| --- | --- | --- | --- | --- | --- | --- |
| full | PASS | 0.7667 | 0.0000 | 24 | 48 | 21 |
| default_path_only | PASS | 0.7583 | -0.0084 | 1 | 1 | 21 |
| raw_output_only | PASS | 0.7636 | -0.0031 | 16 | 16 | 10 |
| no_callable_choice | PASS | 0.7667 | 0.0000 | 16 | 16 | 21 |
| no_fitting_choice | PASS | 0.7583 | -0.0084 | 12 | 12 | 21 |
| no_global_projection | PASS | 0.7666 | -0.0001 | 24 | 48 | 20 |
| no_spectral_shape | PASS | 0.7667 | 0.0000 | 24 | 24 | 21 |
