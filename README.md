# EvoForest Reimplementation

This repo recreates the software architecture described in
the EvoForest paper. It is not a reproduction of the authors' private evolved graph,
their 600-step run, or their reported score.

See the paper here: https://arxiv.org/abs/2604.19761

The implementation focuses on the reusable system design:

- A shared directed acyclic graph of computational nodes.
- Intermediate nodes with multiple alternative implementations.
- Callable-family nodes for reusable projections, gates, and activations.
- Persistent low-dimensional global parameters.
- A single output node whose alternatives are all evaluated as ensemble features.
- Fitting nodes (`ridge_w`, `ridge_g`) that alter sample weighting and iterative
  residual reweighting.
- A Ridge-based cross-validation evaluator with capped configuration search.
- Ancestor-conditioned subpath caching shared across evaluated configurations.
- A two-phase evaluation path: optional global refinement, then frozen Ridge scoring.
  The refinement phase can use optional PyTorch L-BFGS on differentiable graph paths
  or fall back to deterministic NumPy coordinate search.
- Structured diagnostics plus TOON-like reports with scoring context, feature
  correlations, residual signals, exact additive linear contribution summaries,
  a diagnostic global Ridge fit, effective rank, and fold stability.
- Persistent alternative-level age and quality summaries accumulated from the
  best evaluated configuration's feature/dependency diagnostics.
- Deterministic scientist/engineer agents that convert diagnostics into YAML-style
  mutation documents with hypotheses, removals, appended globals, and adds.
- Optional HTTP JSON LLM scientist/engineer agents that use paper-style prompt
  artifacts and emit the same structured mutation documents. In island mode the
  scientist agent defaults to the paper's fixed temperature schedule
  `(0.35, 0.5, 0.6, 0.75)`, while engineer synthesis defaults to temperature `0`.
- Cached task-context summaries with tensor inventory, target balance, scorer
  mechanics, and implementation constraints injected into LLM prompts.
- Node-level mutation support so a document can introduce a new intermediate,
  callable, output, or fitting node before adding alternatives to it.
- Optional trusted source-backed mutation alternatives that store and execute
  paper-style `lambda ctx, values: ...` implementations from mutation YAML.
- Graph maintenance for duplicate collapse, unreachable pruning, and unused globals.
- Failed generated candidates are rejected, logged into events/memoranda, and fed
  back to the next engineer prompt.
- Rejected but executable candidates still go through salvage for locally useful
  alternatives.
- A mutation/evolution loop with persistent JSON artifacts, sectioned
  hypothesis-free memoranda, a versioned global-best archive, sequential island
  mode, and asynchronous thread-backed island mode.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

For the optional PyTorch L-BFGS refinement backend:

```bash
python -m pip install -e '.[dev,torch]'
```

## Run Tests

```bash
pytest -q
```

## Run Demo

```bash
evoforest-arch demo --steps 12 --islands 2 --async-islands --refine-globals --refine-backend auto --output runs/demo
```

## Run Production Evolution Smoke

The production workflow is the safer path for serious graph search. It writes a
run manifest, dataset fingerprint, fixed train/validation/test split manifest,
resume state, reloadable graph artifacts, and a stricter archive that only promotes
graphs after both train improvement and validation recheck improvement.

```bash
evoforest-arch evolve --steps 4 --seed 17 --n-series 240 --length 160 --output runs/production-smoke
evoforest-arch inspect runs/production-smoke
evoforest-arch export-best runs/production-smoke --output runs/production-smoke-best.json
evoforest-arch recheck runs/production-smoke
```

To continue an interrupted run, pass `--resume`; `--steps` then means additional
steps:

```bash
evoforest-arch evolve --resume --steps 4 --output runs/production-smoke
```

The test split is not evaluated during `evolve` or default `recheck`. Consume it
only when you are ready to burn the holdout:

```bash
evoforest-arch recheck runs/production-smoke --include-test
```

This command currently wires the synthetic structural-break dataset. A real
strength claim needs a fixed external dataset loader plus a committed split
manifest; see [docs/evolution_workflow.md](docs/evolution_workflow.md).

The Crunch-style parquet bundle can be loaded as an id-level event-detection
surrogate for the current sequence graph interface:

```bash
evoforest-arch data-summary --data-dir /Users/gabrielkahen/Downloads/data --max-samples 200
evoforest-arch evolve \
  --dataset competition-parquet-event \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --max-samples 200 \
  --steps 1 \
  --output runs/competition-smoke
```

This maps each id to one fixed-length sequence and predicts whether
`tau_index >= 0`. It is useful for evolving this architecture against the parquet
bundle, but it is not the official row-level competition metric. The reduced
labeled test files are not read by `evolve`, `inspect`, `data-summary`, or default
`recheck`; they are read only when `recheck --include-test` is explicitly used.
Run full-data parquet jobs on the PC over SSH rather than on a MacBook Air.

For the row/time-level target table, use `competition-parquet-row`. This creates
one graph sample per `(id, time)` label row from `y_train.parquet`, keeps all
rows for an id in the same split, and represents each row as period-1 reference
features plus the causal period-2 prefix available at that target time:

```bash
evoforest-arch data-summary \
  --row-level \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --competition-series-length 64 \
  --max-ids 120 \
  --max-rows-per-id 16

evoforest-arch evolve \
  --dataset competition-parquet-row \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --competition-series-length 64 \
  --max-ids 120 \
  --max-rows-per-id 16 \
  --steps 8 \
  --output runs/competition-row-smoke
```

The row-level benchmark compares a fixed non-evolved baseline to validation-only
graph evolution on held-out ids:

```bash
python -m benchmarks.competition_row_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --series-length 64 \
  --max-ids 120 \
  --max-rows-per-id 16 \
  --steps 8 \
  --output benchmark_reports/competition-row
```

To check whether mutations are useful on capped parquet data without touching the
reduced test files:

```bash
python -m benchmarks.competition_mutation_usefulness \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --max-samples 1000 \
  --max-configurations 96 \
  --output benchmark_reports/competition
```

To target the paper's likely id-level ADIA protocol more directly, run the
structural-break event benchmark. It uses one label per id, a grouped id split,
strong fixed structural-break baselines, trusted source-backed graph mutations,
and validation-selected graph archive ensembles:

```bash
python -m benchmarks.competition_event_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --series-length 160 \
  --steps 24 \
  --max-configurations 96 \
  --output benchmark_reports/competition-event
```

For multi-seed campaigns, use the campaign runner. The paper-scale target is
`600+` steps per seed on the PC; start smaller to size runtime:

```bash
python -m benchmarks.competition_event_campaign \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --seeds 211,223,227 \
  --series-length 160 \
  --steps 96 \
  --max-configurations 64 \
  --resume-existing \
  --output benchmark_reports/competition-event-campaign
```

To reduce single-split overfitting, use the multi-split runner. It keeps one
fixed grouped outer test holdout, evolves against several grouped validation
splits inside the development pool, prunes accepted alternatives by the same
objective, and reports archive/OOF validation ensembles:

```bash
python -m benchmarks.competition_event_multisplit_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --split-seeds 211,223,307 \
  --series-length 160 \
  --steps 96 \
  --max-configurations 64 \
  --objective-mode auc \
  --output benchmark_reports/competition-event-multisplit
```

To audit whether the graph is bottlenecked by the final readout, compare the
standard Ridge readout to a train-only selected rank/interactions readout and an
OOF-selected blend:

```bash
python -m benchmarks.competition_event_readout_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --graph benchmark_reports/competition-event-multisplit/pruned_consensus_graph.json \
  --split-seeds 211,223,307 \
  --series-length 160 \
  --max-configurations 64 \
  --output benchmark_reports/competition-event-readout
```

To test whether sequential acceptance is rejecting feature families that are
useful only as a set, assemble all trusted source-backed alternatives first,
then backward-prune them under the same grouped multi-split objective. Add
`--include-builtins` to include the deterministic built-in mutation templates as
well:

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

For the row/time-level parquet labels, use the grouped row multi-split benchmark.
This embeds the deterministic row-local baseline as a graph output primitive and
can build an output-only graph by pruning the seed output alternatives:

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

For a faster full-data audit of the row-specific graph primitives, use the
focused graph benchmark. It compares output-only graph variants for the row
baseline, expanded target-time basis, and multiscale recent-tail features across
the same grouped validation/internal-test protocol:

```bash
python -m benchmarks.competition_row_focused_graph_benchmark \
  --data-dir /Users/gabrielkahen/Downloads/data \
  --split-seeds 113,127,149 \
  --series-length 480 \
  --max-ids 10000 \
  --max-rows-per-id 32 \
  --output benchmark_reports/competition-row-focused-graph
```

On the PC with the local parquet bundle, the best focused graph
(`row_baseline_time_tail_graph`) reached mean validation AUC `0.6567` and mean
internal-test AUC `0.6420` with no reduced-test access. That is a real full-data
improvement over the row baseline (`0.6334` internal-test AUC), but it is not yet
a reliable `0.65` internal-test result.

The archive/OOF row multi-split audit committed at
`benchmark_reports/competition-row-multisplit-pc-2k-32-l480-ensemble-cfg4-skipprune`
uses the validation-only archive selector on a 2k-id PC run with pruning skipped
for turnaround. It selected `best_archive_member` (`row_baseline_graph`) with
mean validation AUC `0.6839`, minimum split validation AUC `0.6670`, and no
reduced-test access. The selected member's non-selection internal-test mean/min
AUC was `0.6438`/`0.6416`, and the OOF archive selector did not improve beyond
that single graph. Treat this as evidence that grouped validation can clear
`0.65`, not as evidence that the selected graph/ensemble reliably clears `0.65`
on untouched internal tests.

For the 2026 real-time CrunchDAO contest interface, use the self-contained
notebook in `submissions/structural_break_real_time/`. It defines the required
`train()` and streaming `infer()` functions and adapts the same row/time/tail
feature family to per-online-step predictions.

## Run Benchmarks

The benchmark suite generates reproducible JSON and Markdown reports that show
architecture-level faithfulness to the paper without claiming to reproduce the
authors' private evolved graph or reported score:

```bash
python -m benchmarks.run_all --quick --output benchmark_reports/quick
```

See [docs/benchmarks.md](docs/benchmarks.md) for the full benchmark methodology.

Live LLM mutation synthesis is opt-in. Point the generic HTTP JSON client at a
chat-completions-compatible endpoint:

```bash
export EVOFOREST_LLM_URL="https://your-llm-endpoint.example/v1/chat/completions"
export EVOFOREST_LLM_API_KEY="..."
export EVOFOREST_LLM_MODEL="..."
evoforest-arch demo --steps 4 --llm-provider http-json --output runs/llm-demo
```

The paper-style island scientist schedule can be overridden with
`--llm-island-temperatures 0.35,0.5,0.6,0.75`. Use
`--llm-island-temperatures none` to reuse `--llm-scientist-temperature` for every
island.

By default, runs write a deterministic `task_context.md` from the runtime inputs
and evaluator settings. To inject a hand-authored or externally generated domain
brief into LLM prompts, pass `--task-context-file path/to/context.md`.

`ridge_g` residual rules run as iterative reweighted least squares. Use
`--irls-steps` to control the number of residual-weighted refits after the initial
Ridge fit.

To let trusted LLM-generated mutation documents include source-backed lambda
alternatives, also pass `--allow-source-mutations`. This executes local Python
source from mutation YAML and should only be used with trusted prompts and endpoints.

The demo generates synthetic structural-break data, builds a seed graph, evaluates
graph configurations with a Ridge readout, derives scientist/engineer mutation
documents from diagnostics, and writes mutation documents, events, checkpoints, TOON
diagnostics, versioned global-best archive entries, prompt records when LLM agents
are enabled, execution-error records, task-context summaries, and memoranda under
the chosen output directory. Memoranda use the paper-style sections
`[OUTCOME HISTORY]`, `[STATE]`, `[WHAT WORKS]`, `[WHAT FAILED]`, and
`[ERROR LOG]`.

## Scope

This repository intentionally excludes competition-specific code, hidden benchmark
logic, and local-label feature selection. It is meant as a clean architecture substrate
for experiments with open-ended computational graph evolution.

It also does not claim to recover the private 600-step evolved graph. The global
refinement phase supports a PyTorch L-BFGS path for differentiable primitives and
falls back to a deterministic NumPy surrogate when a graph path is not torch-enabled.
The asynchronous island mode uses local thread workers rather than dedicated GPU
islands. It preserves the paper's fixed four-temperature scientist schedule by
default, but the scientist/engineer loop can also run deterministically offline or
call an opt-in generic HTTP JSON LLM endpoint. It does not include the authors'
private model, exact prompts, or full private code-generation backend. The source-backed
mutation path recreates the paper's lambda-alternative representation for trusted
local experiments, but it is not a security sandbox. Alternative-level age and
quality history is implemented as rolling clean-room summaries over participating
alternatives in accepted stateful evaluations, not as the authors' private statistic
schema. Linear SHAP-style diagnostics are exact additive decompositions of this
repo's standardized Ridge readout (`z_j * coefficient_j`); they are not a claim to
match a private SHAP implementation byte-for-byte. Cross-configuration caching
assumes alternatives are deterministic functions of their parents, inputs, and
globals during one evaluator pass. The graph semantics, configuration scoring,
fitting-node hooks, diagnostics, mutation artifacts, graph maintenance,
failed-mutation feedback, salvage behavior, task-context summaries, prompt records,
global-best archive, and memoranda mirror the paper's software architecture.
Task-context summaries are
deterministic summaries of runtime inputs and scorer mechanics rather than private
LLM-authored domain briefs.
