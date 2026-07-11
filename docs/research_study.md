# EvoForest Generalization Study

This research suite asks when the paper's LLM-guided computational-graph evolution
is useful and where the architecture fails.
It is not a reproduction of the paper's structural-break experiment.

## Phase order

1. Controlled compositional DAG tasks with known variables, motifs, noise, and
   interpolation/extrapolation holdouts.
2. Raw Ridge, random-feature Ridge, seed-graph EvoForest, and evolved EvoForest
   comparisons under train-fitted readouts.
3. Paper-style Gemini scientist, engineer, and memorandum evolution with the
   original four-island temperature schedule.
4. Frozen local SRBench, SRSD, PMLB, or OpenML-style numeric regression manifests.

The quick end-to-end smoke run is:

```bash
python -m benchmarks.research_suite.run_study --quick --env-file .env --output benchmark_reports/research-quick
```

The default run increases synthetic sizes, evolution steps, and configuration
budgets:

```bash
python -m benchmarks.research_suite.run_study --env-file .env --output benchmark_reports/research
```

External datasets are always local, hash-pinned when possible, and use committed
train/validation/test indices:

```bash
python -m benchmarks.research_suite.run_study \
  --manifest data/srbench/task-a.json \
  --manifest data/pmlb/task-b.json \
  --output benchmark_reports/research
```

Before a paid confirmatory run, copy and complete
`benchmarks/research_suite/specs/execution_config.example.json`. Freeze the exact
provider/model, prices, token budgets, and AIDE commands, then load it through the
secret-free execution-config validator. Credentials remain environment variables;
their values are never serialized.

## Evaluation boundary

Graph search and diagnostics see search-train only. Archive versions are scored on
selection-validation using a readout fitted on search-train. After selection, the
graph is frozen, its final Ridge readout is trained on search-train plus validation,
and each sealed holdout is evaluated once. Test labels are never returned by the
protocol object.

The suite reports candidate/evaluation use separately from wall time and future LLM
accounting. Production studies should run paired seeds and treat the task or formula
family—not folds or seeds—as the statistical unit.

## Current baseline boundary

The dependency-free suite includes raw Ridge and deterministic random nonlinear
features. FEAT, PySR/Operon, AutoFeat, boosted trees, CAAFE, and AIDE require their
own pinned environments and should write into the same `ExperimentResultRow`
schema. They are deliberately not silently installed by the benchmark runner.
