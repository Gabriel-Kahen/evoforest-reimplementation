# Benchmarks

This repository includes a benchmark suite for demonstrating architecture-level
faithfulness to the EvoForest paper:

<https://arxiv.org/abs/2604.19761>

The suite does not attempt to reproduce the authors' private evolved graph,
private implementation, competition data path, LLM stack, or reported task score. It
tests whether this reimplementation behaves like the software architecture
described in the paper: multi-alternative graph semantics, configuration search,
Ridge scoring, fitting hooks, trainable globals, diagnostics, mutation artifacts,
memoranda, island settings, archive artifacts, and scaling behavior.

## Run

After installing the repo dependencies, run the full benchmark bundle:

```bash
python -m benchmarks.run_all --output benchmark_reports
```

For a faster smoke run:

```bash
python -m benchmarks.run_all --quick --output benchmark_reports/quick
```

If installed in editable mode, the console entrypoint is also available:

```bash
evoforest-benchmarks --quick --output benchmark_reports/quick
```

Each command writes paired JSON and Markdown reports. The top-level
`benchmark_index.json` and `benchmark_index.md` link to the individual reports.

## Reports

- `conformance_report`
  - Checks the paper-derived architecture contract against concrete runtime and
    static evidence. It verifies DAG node types, output semantics, capped
    configuration search, ancestor-conditioned cache behavior, `ridge_w` and
    `ridge_g`, trainable globals, optional refinement, diagnostics, mutation YAML,
    maintenance, salvage surface, task context, memoranda, LLM temperature
    settings, source-mutation gating, and archive artifacts.
- `synthetic_suite`
  - Runs deterministic synthetic tasks that exercise specific mechanisms:
    structural-break search, nonuniform sample weighting, residual IRLS, callable
    activations, and persistent-global projection outputs.
- `ablation_suite`
  - Evaluates graph variants with mechanisms removed or collapsed. These are
    architecture-surface ablations: they show how configs, feature counts, and
    scores change when search axes are disabled.
- `search_dynamics`
  - Runs a short deterministic evolution loop and summarizes events, accepted
    mutations, salvage fields, global-best archive versions, final graph
    complexity, and artifact paths.
- `runtime_scaling`
  - Measures evaluation time while varying configuration caps, dataset size, and
    output-feature growth. Timing varies by machine; the report includes cache
    hits/misses, config counts, and feature counts so results are interpretable.

## Interpreting Results

A passing benchmark means the reimplementation exposes and exercises the relevant
architectural mechanism. It is evidence that the code is faithful at the software
architecture level.

A passing benchmark does not mean:

- the implementation is the authors' original code;
- the private prompts, private LLM repair stack, or generated graph were recovered;
- any private competition pipeline was reproduced;
- the paper's reported score should be expected.

Runtime rows should be compared on the same machine and Python environment. They
are useful for trends, not for cross-machine absolute claims.

## Useful Individual Commands

```bash
python -m benchmarks.conformance_report --output benchmark_reports
python -m benchmarks.synthetic_suite --output benchmark_reports
python -m benchmarks.ablation_suite --output benchmark_reports
python -m benchmarks.search_dynamics --output benchmark_reports
python -m benchmarks.runtime_scaling --output benchmark_reports
```

The default suite scripts accept `--seed`, `--output`, and `--quick`.
