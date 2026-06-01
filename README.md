# EvoForest Architecture

This is a clean-room repository that recreates the software architecture described in
the EvoForest paper. It is not a reproduction of the authors' private evolved graph,
their 600-step run, or their reported score.

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
