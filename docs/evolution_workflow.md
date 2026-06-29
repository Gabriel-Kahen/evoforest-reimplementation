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

Production runs are island-native by default. Four persistent island process
actors are assigned dedicated devices `cuda:0,cuda:1,cuda:2,cuda:3`; each actor
owns its proposal, repair, evaluation, prompt records, memorandum, graph state,
candidate commits, migration target updates, and job artifacts. The root run
directory records the global-best frontier and migrations from actor
acknowledgements.
Production island runs additionally write:

- `jobs.jsonl`: submitted/completed/failed/stale/abandoned job lifecycle records,
  including island id, worker id, worker execution model, actor PID, dedicated
  device, base graph hash, and mutation path.
- `migrations.jsonl`: global-best transfers to weaker islands, including target
  actor PID and execution model.
- `islands/island_N/state.json`: island-local step, generation, RNG state, best
  scores, history, and errors.
- `islands/island_N/current_graph.json` and `best_graph.json`: island graph state.
- `islands/island_N/checkpoint.json`, `memorandum.md`, `events.jsonl`,
  `jobs.jsonl`, `migrations.jsonl`, `archive/index.jsonl`, `mutations/`, and
  `prompts/`.

Archive promotion is stricter than the demo loop under the default production
profile. A candidate must improve on the train split and improve on the
validation split. The candidate's graph configuration is selected on train, then
validation is rechecked with that config fixed.

The paper profile intentionally changes that contract. `--profile paper` promotes
island and global bests by train-split cross-validated task score, records validation
scores for reporting, and does not use validation as a promotion gate.

## Dataset Requirement

The production CLI includes two built-in smoke datasets. The structural-break
dataset exercises the original time-series specialization:

```bash
evoforest-arch evolve --steps 4 --seed 17 --n-series 240 --length 160 --output runs/production-smoke
```

The generic tabular dataset exercises the task-independent seed graph and
primitive registry:

```bash
evoforest-arch evolve --dataset synthetic-tabular --steps 4 --seed 17 --n-samples 240 --n-features 12 --output runs/production-tabular-smoke
```

These are enough to test the workflow. They are not enough to claim a strong
evolved model on a real benchmark.

LLM-backed production evolution is opt-in and can be configured through `.env`:

```dotenv
EVOFOREST_LLM_PROVIDER=openai
OPENAI_API_KEY=...
EVOFOREST_LLM_MODEL=...
```

Use `EVOFOREST_LLM_PROVIDER=claude` with `ANTHROPIC_API_KEY`, or
`EVOFOREST_LLM_PROVIDER=gemini` with `GEMINI_API_KEY`, to switch providers.

```bash
evoforest-arch evolve --steps 4 --llm-provider env --env-file .env --output runs/production-llm
```

When LLM mode is enabled, provider configuration and LLM outputs are fail-fast.
The runner does not fall back to deterministic mutation agents if the provider is
missing, the HTTP request fails, or the returned mutation document is invalid.

Production four-island mode is the default:

```bash
evoforest-arch evolve --steps 16 --migration-interval 10 --output runs/production-islands
```

For the paper long-run contract:

```bash
evoforest-arch evolve --profile paper --llm-provider env --env-file .env --output runs/paper-profile
```

`--profile paper` resolves to 600 target steps, four async islands, four workers,
dedicated `cuda:0,cuda:1,cuda:2,cuda:3` devices, 64 configurations, PyTorch
L-BFGS refinement, the fixed scientist temperature schedule
`0.35,0.5,0.6,0.75`, engineer temperature `0`, and CV task score promotion. For a
non-CUDA smoke test, override the device slots and disable refinement:

```bash
evoforest-arch evolve --profile paper --steps 0 --island-devices cpu:0,cpu:1,cpu:2,cpu:3 --no-refine-globals --output runs/paper-profile-smoke
```

On a machine without CUDA, use explicit CPU-like slots for smoke tests:

```bash
evoforest-arch evolve --steps 4 --island-devices cpu:0,cpu:1,cpu:2,cpu:3 --no-refine-globals --output runs/production-cpu-smoke
```

Resume restores the manifest's island settings and appends root and per-island
JSONL logs, so the resume command does not need to repeat the island flags:

```bash
evoforest-arch evolve --resume --steps 16 --output runs/production-islands
```

For a real claim, use a loader that returns:

- `inputs`: a dictionary of graph input tensors and scalar metadata.
- `y`: a 1-D target array aligned with the sample axis of the tensors.
- a `TaskSchema` or dataset-name mapping that selects the seed graph and
  primitive registry for those inputs.
- a stable dataset fingerprint derived from the exact input arrays and targets.

Then create one split manifest and reuse it for every run. Do not tune on test.

## External Dataset Loaders

The production CLI includes three external loader adapters:

- `external-npz`: loads a row-aligned `.npz` file with `--dataset-path`,
  `--target-key`, repeated `--input-key`, and optional `--task-schema-file`.
- `external-manifest`: loads a JSON manifest whose relative paths resolve from
  the manifest directory. The manifest can point at an `.npz` adapter or a Python
  module adapter.
- `python-module`: imports `--dataset-module` and calls `--dataset-function`
  (`load_dataset` by default). The function should return a `LoadedDataset`, a
  dict with `inputs` and `y`, or `(inputs, y[, metadata[, task_schema]])`.

For unit- or engine-level tasks, pass the same metadata key into both the split
manifest and cross-validation fold strategy:

```bash
evoforest-arch evolve \
  --dataset external-manifest \
  --dataset-manifest data/dataset_manifest.json \
  --split-group-key engine_id \
  --fold-strategy group_random \
  --group-key engine_id \
  --scorer rmse \
  --output runs/real-task
```

For ordered degradation or forecasting-style data, use `--fold-strategy
time_blocked --time-key cycle`. For target-distribution control on generic
regression tasks, use `--fold-strategy stratified --stratify-bins 5`.

Keep private adapters, targets, and contest submission code outside the public
repo, but commit the public manifest/schema and the resulting split manifest when
you need comparable runs.

## Staged Execution Rules

1. Run `evolve` on synthetic data with source mutations disabled.
2. Inspect artifacts with `inspect`, export the best graph with `export-best`, and
   recheck validation with `recheck`.
3. Add or adjust the real dataset loader and verify that the saved fingerprint rejects changed
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
