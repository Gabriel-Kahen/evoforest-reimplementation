# EvoForest Reimplementation

This repo recreates the software architecture described in
the EvoForest paper. It is not a reproduction of the authors' private evolved graph,
their 600-step run, or their reported score.

See the paper here: https://arxiv.org/abs/2604.19761

The implementation focuses on the reusable system design:

- A shared directed acyclic graph of computational nodes.
- A task schema that declares row-aligned inputs and selects an appropriate seed
  graph/primitive registry for the current task.
- Intermediate nodes with multiple alternative implementations.
- Callable-family nodes for reusable projections, gates, and activations.
- Persistent low-dimensional global parameters.
- A single output node whose alternatives are all evaluated as ensemble features.
- Fitting nodes (`ridge_w`, `ridge_g`) that alter sample weighting and iterative
  residual reweighting.
- A Ridge-based cross-validation evaluator with capped configuration search.
- Ancestor-conditioned subpath caching shared across evaluated configurations.
- A paper-style two-phase evaluation path: PyTorch L-BFGS global refinement is
  attempted by default when trainable globals are active, then globals are frozen
  for Ridge scoring. A deterministic NumPy coordinate backend remains available
  only when explicitly requested for compatibility experiments.
- Structured diagnostics plus TOON-like reports with scoring context, feature
  correlations, residual signals, exact additive linear contribution summaries,
  a diagnostic global Ridge fit, effective rank, and fold stability.
- Persistent alternative-level age and quality summaries accumulated from the
  best evaluated configuration's feature/dependency diagnostics.
- Deterministic scientist/engineer agents that convert diagnostics into YAML-style
  mutation documents with hypotheses, removals, appended globals, and adds.
- Optional OpenAI, Claude, or Gemini LLM scientist, engineer, and memorandum
  agents that use paper-style prompt artifacts. LLM mutation synthesis is
  lambda-first and accepts paper-style node-keyed YAML such as `add: output:
  - "lambda ctx, values: ..."`. Shorthand lambdas infer parents from
  `values["parent"]`; the extended schema can also declare `parents`,
  `global_refs`, `node_kind`, `output_contract`, and `torch_source`. In island
  mode the scientist agent defaults to
  the paper's fixed temperature schedule `(0.35, 0.5, 0.6, 0.75)`, while
  engineer synthesis and memorandum updates default to temperature `0`.
- Cached task-context summaries with tensor inventory, target summary, scorer
  mechanics, and implementation constraints injected into LLM prompts.
- Node-level mutation support so a document can introduce a new intermediate,
  callable, output, or fitting node before adding alternatives to it.
- Optional sandboxed source-backed mutation alternatives that store and execute
  paper-style `lambda ctx, values: ...` implementations from mutation YAML with
  timeout, resource-limit, deterministic rerun, and output-contract checks.
- Graph maintenance for duplicate collapse, unreachable pruning, and unused globals.
- Failed generated candidates are rejected, logged into events/memoranda, and fed
  back to the next engineer prompt.
- Rejected but executable candidates still go through salvage for locally useful
  alternatives.
- A mutation/evolution loop with persistent JSON artifacts, sectioned
  hypothesis-free memoranda, a versioned global-best archive, sequential island
  mode, demo thread-backed islands, and production process-backed island actors.

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
evoforest-arch demo --steps 12 --islands 2 --async-islands --refine-backend auto --output runs/demo
```

## Run Production Evolution Smoke

The production workflow is the safer path for serious graph search. It writes a
run manifest, dataset fingerprint, fixed train/validation/test split manifest,
resume state, reloadable graph artifacts, and a stricter archive that only promotes
graphs after both train improvement and validation recheck improvement. By
default, production `evolve` uses the paper-native four-island asynchronous
topology with dedicated devices `cuda:0,cuda:1,cuda:2,cuda:3`.

```bash
evoforest-arch evolve --steps 4 --seed 17 --n-series 240 --length 160 --output runs/production-smoke
evoforest-arch evolve --dataset synthetic-tabular --steps 4 --seed 17 --n-samples 240 --n-features 12 --output runs/production-tabular-smoke
evoforest-arch inspect runs/production-smoke
evoforest-arch export-best runs/production-smoke --output runs/production-smoke-best.json
evoforest-arch recheck runs/production-smoke
```

Each production island runs as its own OS process actor and owns proposal,
repair, evaluation, prompt records, state, and memorandum updates. Candidate
commits and global-best migrations are sent to the target island as actor
messages, and the coordinator replaces its scheduling view from the actor's
returned snapshot. The root run directory is the global-best ledger. To run a
local smoke test without CUDA, pass explicit CPU-like device slots:

```bash
evoforest-arch evolve --steps 4 --island-devices cpu:0,cpu:1,cpu:2,cpu:3 --no-refine-globals --output runs/production-cpu-smoke
```

For the closest clean-room match to the paper's reported long run, use the paper
profile. It sets 600 target steps, four async islands, dedicated `cuda:0..3`
devices, 64 configurations, PyTorch L-BFGS refinement, scientist temperatures
`0.35,0.5,0.6,0.75`, engineer temperature `0`, and CV task score promotion without
the production validation gate:

```bash
evoforest-arch evolve --profile paper --llm-provider env --env-file .env --output runs/paper-profile
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

The built-in `synthetic-structural-break` and `synthetic-tabular` datasets are
workflow smoke tests. A real strength claim needs a fixed external dataset loader
plus a committed split manifest; see
[docs/evolution_workflow.md](docs/evolution_workflow.md). External data loaders,
contest submissions, and local benchmark artifacts are intentionally kept outside
this public reimplementation repo.

## Run Benchmarks

The benchmark suite generates reproducible JSON and Markdown reports that show
architecture-level faithfulness to the paper without claiming to reproduce the
authors' private evolved graph or reported score:

```bash
python -m benchmarks.run_all --quick --output benchmark_reports/quick
```

See [docs/benchmarks.md](docs/benchmarks.md) for the full benchmark methodology.

Live LLM mutation synthesis is opt-in. Start from the checked-in example and
choose exactly one provider block:

```bash
cp .env.example .env
```

OpenAI:

```dotenv
EVOFOREST_LLM_PROVIDER=openai
OPENAI_API_KEY=...
EVOFOREST_LLM_MODEL=...
```

Claude:

```dotenv
EVOFOREST_LLM_PROVIDER=claude
ANTHROPIC_API_KEY=...
EVOFOREST_LLM_MODEL=...
```

Gemini:

```dotenv
EVOFOREST_LLM_PROVIDER=gemini
GEMINI_API_KEY=...
EVOFOREST_LLM_MODEL=...
```

Optional shared settings:

```dotenv
EVOFOREST_LLM_TIMEOUT_SECONDS=120
EVOFOREST_LLM_MAX_TOKENS=4096
```

Then run with provider resolution from `.env`:

```bash
evoforest-arch demo --steps 4 --llm-provider env --env-file .env --output runs/llm-demo
```

You can also select the provider explicitly with `--llm-provider openai`,
`--llm-provider claude`, or `--llm-provider gemini`; credentials and model still
load from `.env`.

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

LLM-generated mutation documents may include source-backed lambda alternatives by
default, matching the paper's mutation representation. For deterministic
non-LLM runs, pass `--allow-source-mutations` to enable source mutation
documents. Source lambdas are AST-validated and evaluated in a subprocess
sandbox with timeout, memory limit where supported, deterministic repeated
evaluation, and output-contract checks. This is a practical local execution
boundary, not a replacement for an OS/container sandbox for hostile code.

LLM mode is fail-fast. If the configured provider is missing, the HTTP request
fails, the scientist response cannot be parsed into hypotheses, or the engineer
response cannot be parsed and validated as a mutation document, the run raises an
error instead of falling back to deterministic agents. Prompt artifacts written
before the error are kept under the run's `prompts/` directory.

The production workflow can use the same `.env` wiring:

```bash
evoforest-arch evolve --steps 4 --llm-provider env --env-file .env --output runs/production-llm
```

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
logic, and private target-specific feature selection. It is meant as a clean architecture substrate
for experiments with open-ended computational graph evolution.

It also does not claim to recover the private 600-step evolved graph. The global
refinement phase attempts PyTorch L-BFGS for differentiable primitives and
records an explicit skipped-refinement reason when the torch path is unavailable
or a gradient probe finds no active trainable global influence. The NumPy
coordinate refiner is retained as an explicit compatibility backend, not as the
paper-mode fallback.
Production `evolve` is island-native by default: four persistent island process
actors own their graph state and are assigned one dedicated device each
(`cuda:0,cuda:1,cuda:2,cuda:3` unless `--island-devices` overrides them). It
persists per-island state, graph artifacts, checkpoints, memoranda, job logs, and
migration records from inside the island actor, so resume restores the island
frontier instead of restarting from a seed graph. It preserves the paper's fixed
four-temperature scientist schedule by default. The `--profile paper` preset switches promotion from the
default held-out validation gate to the paper's CV task score frontier and records
that contract in the manifest. The scientist/engineer loop can run
deterministically offline or call an opt-in OpenAI, Claude, or Gemini provider.
It does not include the authors' private model, exact prompts, full private
code-generation backend, or cluster scheduler.
The source-backed mutation path recreates the paper's
lambda-alternative representation for trusted local experiments, but it is not a
security sandbox. Alternative-level age and
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
