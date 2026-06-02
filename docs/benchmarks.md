# Benchmarks

This repository includes a benchmark suite for demonstrating architecture-level
faithfulness to the EvoForest paper:

<https://arxiv.org/abs/2604.19761>

The suite does not attempt to reproduce the authors' private evolved graph,
private implementation, competition data path, LLM stack, or reported ROC-AUC. It
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
- `competition_row_benchmark`
  - Optional external-data benchmark for the Crunch parquet files. It loads
    `y_train.parquet` at row/time granularity, splits by id, fits a fixed
    non-evolved row-local baseline on train ids, and compares validation-only
    graph evolution on held-out ids. It is not part of `benchmarks.run_all`
    because it requires the downloaded parquet bundle.
- `competition_row_multisplit_benchmark`
  - Stronger row/time-level benchmark. It uses one fixed grouped outer test
    holdout, multiple grouped validation splits inside the development pool,
    embeds the deterministic row-local baseline as a graph output primitive, and
    reports both seed-output-plus-baseline and output-only graph variants.
- `competition_event_benchmark`
  - Optional external-data benchmark for the paper's likely id-level ADIA
    protocol. It loads one structural-break label per id from the index parquet,
    uses grouped id splits, compares a strong fixed structural-break baseline
    against source-backed EvoForest mutations, and reports validation-selected
    graph archive ensembles.
- `competition_event_campaign`
  - Multi-seed wrapper around `competition_event_benchmark`. Use this on the PC
    to scale toward the paper-style `600+` step campaign and aggregate per-seed
    baseline, evolved graph, ensemble, source-candidate, and leakage results.
- `competition_event_multisplit_benchmark`
  - Stronger id-level benchmark that uses one fixed grouped outer test holdout,
    evolves against multiple grouped validation splits inside the development
    pool, prunes accepted alternatives with the same robust objective, writes
    consensus graph artifacts, and reports archive/OOF validation ensembles.
- `competition_event_readout_benchmark`
  - Optional external-data audit for a serialized event graph. It keeps the same
    fixed grouped outer test protocol and compares the standard Ridge readout to
    a train-only selected rank/interactions readout plus an OOF-selected blend.
- `competition_event_source_suite_benchmark`
  - Optional external-data audit for graph assembly strategy. It adds every
    repair-checked trusted source mutation as a set, optionally adds built-in
    mutation templates with `--include-builtins`, then backward-prunes the added
    alternatives under grouped multi-split validation.

## Interpreting Results

A passing benchmark means the reimplementation exposes and exercises the relevant
architectural mechanism. It is evidence that the code is faithful at the software
architecture level.

A passing benchmark does not mean:

- the implementation is the authors' original code;
- the private prompts, private LLM repair stack, or generated graph were recovered;
- the ADIA competition pipeline was reproduced;
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
python -m benchmarks.competition_row_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --series-length 64 \
  --max-ids 120 \
  --max-rows-per-id 16 \
  --steps 8 \
  --output benchmark_reports/competition-row
python -m benchmarks.competition_event_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --series-length 160 \
  --steps 24 \
  --max-configurations 96 \
  --output benchmark_reports/competition-event
python -m benchmarks.competition_event_campaign \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --seeds 211,223,227 \
  --series-length 160 \
  --steps 96 \
  --max-configurations 64 \
  --resume-existing \
  --output benchmark_reports/competition-event-campaign
python -m benchmarks.competition_event_multisplit_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --split-seeds 211,223,307 \
  --series-length 160 \
  --steps 96 \
  --max-configurations 64 \
  --objective-mode auc \
  --output benchmark_reports/competition-event-multisplit
```

```bash
python -m benchmarks.competition_event_readout_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --graph benchmark_reports/competition-event-multisplit/pruned_consensus_graph.json \
  --split-seeds 211,223,307 \
  --series-length 160 \
  --max-configurations 64 \
  --output benchmark_reports/competition-event-readout
```

```bash
python -m benchmarks.competition_event_source_suite_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --split-seeds 211,223,307 \
  --series-length 160 \
  --max-configurations 64 \
  --objective-mode auc \
  --include-builtins \
  --disable-source-screen \
  --output benchmark_reports/competition-event-template-suite
```

```bash
python -m benchmarks.competition_row_multisplit_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --split-seeds 113,127,149 \
  --series-length 480 \
  --max-ids 2000 \
  --max-rows-per-id 32 \
  --max-configurations 16 \
  --objective-mode auc \
  --disable-builtins \
  --output benchmark_reports/competition-row-multisplit
```

The default suite scripts accept `--seed`, `--output`, and `--quick`. The
external parquet benchmark accepts `--seed` and `--output` plus data-capping
arguments such as `--max-ids` and `--max-rows-per-id`.
