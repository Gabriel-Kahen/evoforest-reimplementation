# Task Context Summary

Clean-room EvoForest task context generated from runtime inputs.

## Tensor Inventory
- boundary: scalar int64, value=48.
- series: numeric_tensor float64, shape=[108, 96], finite=1.000, mean=0.082357, std=0.449268, min=-2.471652, max=2.058866.

## Target
- rows=108, positives=54, negatives=54, positive_rate=0.500000.

## Scorer Mechanics
- Fitness is best configuration ROC-AUC from stratified 3-fold Ridge CV.
- Configuration enumeration is capped at 16 candidates per evaluation.
- Features are standardized inside each fold; Ridge is solved by closed-form SVD.
- Alpha is selected from 17 log-scale values using leave-one-out leverage MSE.
- ridge_g residual rules run IRLS for up to 2 residual-weighted refits.
- Global refinement enabled=False, backend=auto, steps=20.

## Implementation Constraints
- Intermediate, callable, and fitting nodes are selected by configuration.
- All output alternatives are evaluated and stacked as Ridge features for each configuration.
- Graph alternatives should be deterministic over parents, inputs, and fixed globals during one evaluator pass because subpaths are cached.
- Globals are persistent trainable parameters; new globals are append-only at mutation time and unused globals may be pruned.
- Mutation documents must preserve DAG validity and use known primitives unless trusted source-backed mutations are explicitly enabled.
