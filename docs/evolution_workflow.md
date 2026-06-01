# Production Evolution Workflow

This repository has two evolution paths:

- `evoforest-arch demo` is a research demo for exercising the architecture.
- `evoforest-arch evolve` is the safer production path for longer graph search.

The production path is still a clean-room implementation. It does not recreate the
paper authors' private evolved graph, private prompts, or private competition stack.
It provides the operational controls needed before attempting serious graph search.

## What The Production Path Guarantees

Each new run writes:

- `run_manifest.json`: dataset configuration, dataset fingerprint, evaluator config,
  acceptance policy, git commit, and staged execution rules.
- `splits.json`: fixed train, validation, and test indices.
- `state.json`: resume step, archive version, best scores, best config, graph paths,
  RNG state, recent history, and test recheck count.
- `current_graph.json` and `best_graph.json`: reloadable graph artifacts for
  registry-backed alternatives.
- `archive/index.jsonl`: versioned best graph entries.
- `events.jsonl`: candidate outcomes.
- `validation_rechecks.jsonl`: explicit recheck records.

Archive promotion is stricter than the demo loop. A candidate must improve on the
train split and improve on the validation split. The candidate's graph configuration
is selected on train, then validation is rechecked with that config fixed.

## Dataset Requirement

The current production CLI is wired to the synthetic structural-break dataset:

```bash
evoforest-arch evolve --steps 4 --seed 17 --n-series 240 --length 160 --output runs/production-smoke
```

That is enough to test the workflow. It is not enough to claim a strong evolved
model on a real benchmark.

For a real claim, add a loader that returns:

- `inputs`: a dictionary of graph input tensors and scalar metadata.
- `y`: a 1-D target array aligned with the sample axis of the tensors.
- a stable dataset fingerprint derived from the exact input arrays and labels.

Then create one split manifest and reuse it for every run. Do not tune on test.

## Competition Parquet Loader

The repository also includes a Crunch-style parquet loader:

```bash
evoforest-arch data-summary --data-dir /Users/gabrielkahen/Downloads/data --max-samples 200
evoforest-arch evolve \
  --dataset competition-parquet-event \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --max-samples 200 \
  --steps 1 \
  --output runs/competition-smoke
```

The loader expects:

- `X_train.parquet`
- `y_train_index.parquet`
- optional reduced holdout files `X_test.reduced.parquet` and
  `y_test_index.reduced.parquet`

It maps each id to one sequence by resampling period 1 into the pre-boundary half
and period 2 into the post-boundary half. The target is `tau_index >= 0` from the
index file. This is an id-level event-detection surrogate for the current graph
interface. It is not the official row-level competition metric.

Default commands do not read the reduced labeled test files:

- `evolve` reads train parquet only.
- `inspect` reads run artifacts only.
- `data-summary` reads train parquet only unless `--include-reduced-test` is passed.
- `recheck` reads train parquet only unless `--include-test` is passed.

Use `recheck --include-test` only after the graph and search procedure are frozen.

Full-data parquet jobs should run over SSH on the PC, for example:

```bash
ssh gabe@gabepc 'cd /home/gabe/evoforest-reimplementation-run && .venv/bin/evoforest-arch evolve --dataset competition-parquet-event --data-dir /home/gabe/evoforest-crunch-open-benchmark/data --steps 1 --output runs/competition-full'
```

## Staged Execution Rules

1. Run `evolve` on synthetic data with source mutations disabled.
2. Run capped parquet smoke locally, using `--max-samples`, with reduced test files untouched.
3. Inspect artifacts with `inspect`, export the best graph with `export-best`, and
   recheck validation with `recheck`.
4. Add or adjust the real dataset loader and verify that the saved fingerprint rejects changed
   data.
5. Run short real-data calibration jobs on train/validation only, using SSH for full data.
6. Increase steps and islands only after resume, export, and validation rechecks are
   boringly reliable.
7. Use `recheck --include-test` only once the graph and search procedure are frozen.

## Useful Commands

Fresh run:

```bash
evoforest-arch evolve --steps 8 --output runs/evolve-smoke
```

Resume:

```bash
evoforest-arch evolve --resume --steps 8 --output runs/evolve-smoke
```

Inspect:

```bash
evoforest-arch inspect runs/evolve-smoke
```

Export best graph:

```bash
evoforest-arch export-best runs/evolve-smoke --output runs/evolve-smoke-best.json
```

Validation recheck:

```bash
evoforest-arch recheck runs/evolve-smoke
```

Explicit test recheck:

```bash
evoforest-arch recheck runs/evolve-smoke --include-test
```
