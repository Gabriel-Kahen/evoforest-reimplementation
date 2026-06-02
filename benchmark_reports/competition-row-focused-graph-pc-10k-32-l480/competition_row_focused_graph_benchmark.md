# Competition Row Focused Graph Benchmark

These benchmarks validate architecture-level behavior of this clean-room reimplementation. They do not claim to reproduce the authors' private evolved graph, private code-generation stack, competition pipeline, or reported ROC-AUC.

Dataset: `competition-parquet-row` from `/home/gabe/evoforest-competition-data`, rows=`317199`, ids=`10000`, max_ids=`10000`, max_rows_per_id=`32`.

Config: `{'split_seeds': [113, 127, 149], 'stability_weight': 0.5, 'graphs': {'row_baseline_only_graph': ['adia_row_baseline_outputs'], 'row_baseline_time_graph': ['adia_row_baseline_outputs', 'row_time_basis_outputs'], 'row_baseline_tail_graph': ['adia_row_baseline_outputs', 'row_multiscale_tail_outputs'], 'row_baseline_time_tail_graph': ['adia_row_baseline_outputs', 'row_time_basis_outputs', 'row_multiscale_tail_outputs']}, 'selection_policy': 'best graph by grouped multi-split validation objective; internal test is reported for audit only'}`

Reduced test accessed: `False`.

Best validation graph: `row_baseline_time_tail_graph` with mean validation AUC `0.6567` and internal test mean AUC `0.6420`.

| Graph | Mean Val AUC | Mean Val Delta | Min Val AUC | Mean Test AUC | Mean Test Delta | Objective |
| --- | --- | --- | --- | --- | --- | --- |
| row_baseline_only_graph | 0.6512 | 0.0000 | 0.6476 | 0.6334 | 0.0000 | 0.9750 |
| row_baseline_time_graph | 0.6523 | 0.0011 | 0.6487 | 0.6362 | 0.0028 | 0.9766 |
| row_baseline_tail_graph | 0.6558 | 0.0046 | 0.6518 | 0.6398 | 0.0064 | 0.9816 |
| row_baseline_time_tail_graph | 0.6567 | 0.0056 | 0.6528 | 0.6420 | 0.0086 | 0.9831 |
