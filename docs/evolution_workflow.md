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

## Staged Execution Rules

1. Run `evolve` on synthetic data with source mutations disabled.
2. Inspect artifacts with `inspect`, export the best graph with `export-best`, and
   recheck validation with `recheck`.
3. Add the real dataset loader and verify that the saved fingerprint rejects changed
   data.
4. Run short real-data calibration jobs on train/validation only.
5. Increase steps and islands only after resume, export, and validation rechecks are
   boringly reliable.
6. Use `recheck --include-test` only once the graph and search procedure are frozen.

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
