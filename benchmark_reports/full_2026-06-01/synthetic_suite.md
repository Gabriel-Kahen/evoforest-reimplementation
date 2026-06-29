# Synthetic Mechanism Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported task score.

Seed: `23`

Passed: `5/5`

| Case | Status | Mechanism | Score | Baseline Score | Delta |
| --- | --- | --- | --- | --- | --- |
| structural_break_full_search | PASS | capped configuration search over reusable graph computations | 0.8230 | n/a | n/a |
| sample_weight_boundary_energy | PASS | ridge_w nonuniform sample weights | 0.7970 | 0.7874 | 0.0096 |
| residual_huber_irls | PASS | ridge_g iterative residual reweighting on heavy-tailed data | 0.5421 | 0.5421 | 0.0000 |
| callable_sigmoid_gate | PASS | callable-node activation family selected by configuration | 0.7050 | 0.7874 | -0.0824 |
| global_projection_feature | PASS | output feature backed by persistent trainable globals | 0.7050 | n/a | n/a |
