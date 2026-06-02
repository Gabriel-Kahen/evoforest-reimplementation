# Competition Event Readout Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-event` from `/home/gabe/evoforest-competition-data`, ids=`4000`, max_samples=`4000`.

Graph path: `benchmark_reports/competition-event-multisplit-pc-4k-48-auc/pruned_consensus_graph.json`.

Reduced test accessed: `False`.

Best readout by validation AUC: `ridge`.

| Readout | Mean Val AUC | Mean Val Delta | Mean Internal Test AUC | Mean Test Delta |
| --- | --- | --- | --- | --- |
| ridge | 0.5611 | 0.0071 | 0.5371 | 0.0108 |
| rank_interaction | 0.5388 | -0.0152 | 0.5319 | 0.0056 |
| ridge_rank_blend | 0.5583 | 0.0043 | 0.5345 | 0.0082 |

| Seed | Baseline Val | Ridge Val | Rank Val | Blend Val | Baseline Test | Blend Test |
| --- | --- | --- | --- | --- | --- | --- |
| 211 | 0.5283 | 0.5357 | 0.5203 | 0.5322 | 0.5214 | 0.5310 |
| 223 | 0.5952 | 0.6050 | 0.5730 | 0.6000 | 0.5296 | 0.5356 |
| 307 | 0.5385 | 0.5426 | 0.5230 | 0.5426 | 0.5279 | 0.5370 |
