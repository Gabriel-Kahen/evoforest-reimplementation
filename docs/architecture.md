# Architecture Mapping

The paper describes EvoForest as a search-first machine-learning system that evolves
computational structure rather than only fitting weights. This repo maps that idea to
the following modules.

## Core Pieces

- `evoforest_arch.graph`
  - Graph IR, nodes, alternatives, evaluation context, feature blocks, and graph
    serialization metadata, including persistent per-alternative age and quality
    history plus ancestor-conditioned cache keys.
- `evoforest_arch.globals`
  - Persistent global parameters that alternatives can read and refine.
- `evoforest_arch.task`
  - Task schemas for row-aligned inputs, targets, default input names, and
    dataset-to-task selection.
- `evoforest_arch.datasets`
  - Dataset loader registry for synthetic smoke tasks, external `.npz` files,
    manifest-driven adapters, and Python module hooks. Loaders return inputs,
    target arrays, task schema, and metadata.
- `evoforest_arch.metrics`
  - Task scorer objects that preserve the raw metric name and natural direction
    while exposing a maximized optimization score, plus task-aware fold strategy
    objects for random, grouped, stratified, and time-blocked CV.
- `evoforest_arch.primitives`
  - A compact primitive library with task-selected tabular or structural-break
    feature primitives plus common callable gates, activations, output
    compositions, and Ridge fitting rules.
- `evoforest_arch.source`
  - Sandboxed source-backed lambda alternatives that mirror the paper's YAML
    mutation representation, with timeout/resource controls, deterministic
    validation, output contracts, callable/fitting-node specs, optional
    torch-source evaluators, and safe automatic torch-expression derivation for
    simple feature-producing NumPy lambdas.
- `evoforest_arch.evaluator`
  - Capped configuration enumeration, default paper-mode global refinement, Ridge
    cross-validation, fitting-node execution, diagnostic global Ridge fitting,
    and exact additive linear contribution diagnostics.
- `evoforest_arch.feedback`
  - Search feedback summaries and compact TOON-like diagnostic tables with
    scoring context, feature redundancy, residual signals, subnode aggregates,
    linear contribution summaries, and alternative-level history.
- `evoforest_arch.agents`
  - Deterministic scientist and engineer roles that turn diagnostics into
    structured mutation documents.
- `evoforest_arch.llm`
  - Paper-style global rules, scientist and engineer prompt builders, prompt
    records, deterministic test clients, and opt-in OpenAI, Claude, and Gemini
    clients.
- `evoforest_arch.task_context`
  - Deterministic cached task-context summaries with tensor inventory, target
    balance, scorer mechanics, and implementation constraints.
- `evoforest_arch.refinement`
  - PyTorch `nn.ParameterDict` plus L-BFGS refinement for differentiable fixed
    paths, with an explicit gradient-probe skip when trainable globals do not
    influence the selected path. A deterministic NumPy backend remains available
    only when explicitly requested.
- `evoforest_arch.mutations`
  - YAML-style mutation documents, paper-style node-keyed lambda additions,
    mutation specs, removals, appended globals, source-backed alternatives,
    explicit parent/global/node-kind/output-contract metadata, and a
    registry-driven deterministic compatibility mutation engine.
- `evoforest_arch.maintenance`
  - Graph cleanup for duplicate alternatives, unreachable nodes, empty nodes,
    unused globals, and DAG validity.
- `evoforest_arch.evolution`
  - Open-ended mutation loop, island loop, archive, events, checkpoints, and
    memoranda, sequential and asynchronous island execution, failed-candidate
    feedback, plus rejected-candidate salvage.
- `evoforest_arch.seed`
  - Task-selected seed graphs. The default is a generic tabular graph; the
    structural-break graph is an explicit specialization with the same
    intermediate/callable/output/fitting/global architecture.

## Paper-Specific Semantics

- `input` nodes read data supplied by the task.
- A `TaskSchema` defines the task's inputs and default input. The seed builder and
  primitive registry are selected from that schema, so structural-break inputs are
  no longer assumed by the architecture.
- `intermediate` nodes contain competing tensor-producing alternatives.
- `callable` nodes contain competing alternatives that return reusable functions.
- `output` alternatives are not selected by configuration. Every output alternative
  is evaluated for a fixed configuration and stacked into the Ridge feature matrix.
- `fitting` nodes are selected by configuration. `ridge_w` returns per-sample
  weights, and `ridge_g` returns a residual-to-weight rule used in iterative
  reweighted least squares.
- The evaluator enumerates reachable intermediate, callable, and fitting alternatives
  up to `max_configurations`; graph fitness is the best configured task score.
- Configuration search shares a cache across evaluated candidates. Cache keys encode
  the selected ancestor subpath for each alternative, so a reused intermediate is
  computed once when all of its ancestors agree, but recomputed when a parent
  alternative changes.
- The Ridge readout standardizes features, selects alpha from a log-scale grid using
  a leave-one-out leverage criterion inside each fold, scores each configuration by
  mean fold task score, records fold score dispersion, and summarizes the evaluated
  configuration score range.
- Scorers keep two values distinct: the raw task metric and the maximized
  optimization score. Lower-is-better losses such as RMSE/MAE are represented as
  negative raw values for selection while diagnostics retain the raw loss and its
  natural direction.
- Fold selection is task-aware. The evaluator can use random folds, grouped folds
  for unit/engine leakage control, stratified target folds, and time-blocked folds;
  the production split manifest can also keep train/validation/test unit groups
  disjoint.
- `ridge_g` residual rules are applied through an explicit IRLS loop: after the
  initial Ridge fit, residuals are converted to weights and Ridge is refit for the
  configured number of `irls_steps`, with per-fold iteration diagnostics recorded.
- After configuration scoring, the evaluator fits a diagnostic Ridge model on the
  valid feature pool accumulated across evaluated configurations. Feature
  diagnostics for the best configuration include exact additive linear contribution
  summaries for both the out-of-fold CV models and the best-configuration global
  diagnostic model. The contribution basis is the standardized Ridge term
  `z_j * coefficient_j`, which reconstructs each linear prediction with the model
  intercept up to numerical precision.
- For stateful evaluations, the best configuration updates persistent
  alternative-level statistics. Each alternative ages once per stateful evaluation,
  and participating alternatives accumulate rolling summaries of feature count,
  importance, standalone score, residual coverage, redundancy, fold-weight stability,
  and configuration score. These summaries are serialized in the graph and surfaced
  in feedback/TOON artifacts.
- Paper-mode global refinement runs before configuration scoring when trainable
  globals are active, then globals are frozen for Ridge evaluation. When PyTorch is
  installed and the selected path has torch evaluators, trainable globals are
  materialized as `nn.Parameter`s, checked by a short gradient probe, and optimized
  by L-BFGS together with an ephemeral linear readout.
- Mutation proposals follow a two-role pipeline: a scientist agent turns feature and
  subnode diagnostics into hypotheses, and an engineer agent emits structured
  YAML-style documents with rationale, new nodes, removals, appended globals, and
  additions. Additions can reference built-in primitives or, when source
  mutations are enabled, sandboxed `lambda ctx, values: ...` alternatives.
  Paper-style shorthand lambdas infer parents from `values["parent"]` and global
  references from `ctx.globals.get("name")`; the extended schema preserves
  explicit parents, global refs, output contracts, node kind, and optional
  `torch_source` for differentiable refinement. When a feature-producing NumPy
  lambda is simple enough, the source compiler derives a torch expression
  automatically. Callable and fitting source alternatives cross the subprocess
  boundary as declarative specs and are materialized into safe runtime objects in
  the parent process. LLM-backed engineer prompts are lambda-first and also
  accept the paper's node-keyed YAML form. The default agents are deterministic,
  while optional LLM-backed agents use the same document contract and persist
  their prompts and responses.
  LLM-backed agents are fail-fast: request failures, unparseable hypotheses, invalid
  mutation documents, malformed memorandum updates, and missing provider
  configuration raise errors instead of falling back to deterministic agents.
  In island mode, LLM scientist calls default to the paper's fixed schedule
  `(0.35, 0.5, 0.6, 0.75)` by island index, and LLM engineer calls default to
  zero temperature for mutation synthesis.
- At run startup, a task-context summary is written to `task_context.md` and injected
  into default LLM prompt builders. It records optional task-source excerpts,
  runtime tensor inventory, target summary, scorer mechanics, fold strategy, and
  implementation constraints so mutations are grounded in the actual task
  interface.
- After each mutation, maintenance validates the DAG, collapses exact duplicate
  alternatives, prunes unreachable structure, and removes unused globals.
- Generated candidates that fail during mutation application or evaluation are
  rejected without changing the active graph. The failure is written to events and
  memoranda, then passed into the next engineer prompt as recent execution errors.
- Rejected candidates are checked for locally beneficial added alternatives before
  they are discarded.
- Evolution writes mutation YAML, JSONL events, checkpoints, TOON diagnostics, and
  memoranda, including recent failure logs. It also writes a root-level
  `archive/index.jsonl` and versioned global-best snapshots, preserving the
  frontier checkpoints that correspond to the paper's global-best versions. When
  LLM-backed agents are enabled it also writes prompt/response records. Island
  mode keeps per-island artifacts and migrates the current global best to weaker
  islands. Demo async islands evaluate one candidate per island concurrently in
  local worker threads and process successes or failures as they arrive.
  Production evolve is island-native by default: four persistent OS process
  actors own proposal, repair, evaluation, prompt records, memoranda, graph
  state, job lifecycle logs, candidate commits, migration target application,
  and stale-completion detection, with one dedicated device per island and
  immediate persistence of migration targets so resume keeps the same island
  frontier. The coordinator keeps only a scheduling/global-best view and replaces
  that view from actor snapshots returned by commit and migration messages.
  LLM-backed island runs preserve the fixed scientist-temperature schedule
  independently of candidate evaluation completion order.
- Memoranda are sectioned into `[OUTCOME HISTORY]`, `[STATE]`, `[WHAT WORKS]`,
  `[WHAT FAILED]`, and `[ERROR LOG]`. In LLM mode a separate memorandum agent
  writes this paper-style hypothesis-free memory and must return all required
  sections. Offline deterministic runs use a compact non-TOON raw summary of
  observed outcomes and diagnostics.

## Design Principles

The code keeps the graph architecture explicit. Alternatives are first-class objects,
configuration is separate from the graph, and the evaluator produces structured
feedback rather than only a scalar score.

This is not intended to duplicate the authors' private implementation. It is a
faithful software substrate for the paper's architectural pattern.

Known approximations:

- The paper's private implementation is PyTorch-first. This repo now treats the
  PyTorch L-BFGS path as paper mode. Registry primitives need a `torch_fn`; source
  alternatives need a provided or safely auto-derived torch expression to
  participate in that path. Otherwise refinement records a skipped torch reason.
  The deterministic NumPy coordinate refiner is explicit compatibility behavior.
- The paper's long run used four asynchronous GPU islands; production `evolve`
  now defaults to four durable island-native process actors mapped to
  `cuda:0,cuda:1,cuda:2,cuda:3`, with the paper-style scientist temperature
  schedule persisted in the run manifest. The root coordinator still runs in one
  process, decides global-best promotion, and selects migration targets; it is not
  the authors' private cluster orchestration stack. Island candidate work, island
  commits, island memoranda, and migration target state changes are
  process-isolated per dedicated device.
- The default production profile remains stricter than the paper's reported
  global-best CV score frontier: a production candidate must improve both train CV
  score and a held-out validation recheck before it can become an island or
  global best. Use `--profile paper` to switch to the paper-style CV task score
  frontier without the validation gate.
- Cross-configuration caching assumes alternatives are deterministic over their
  parents, inputs, and fixed globals during one evaluator pass. Source-backed
  alternatives are rerun for deterministic validation inside the sandbox, but this
  is still a local execution boundary rather than a hardened container for hostile
  code.
- The LLM scientist/engineer/memorandum loop is represented by deterministic local
  agents by default, plus optional OpenAI, Claude, or Gemini LLM-backed agents
  configured from environment variables or a `.env` file. Source-backed
  alternatives recreate the paper's lambda-style graph edits with subprocess
  timeout/resource validation, but this does not include the authors' private
  model stack, exact prompt corpus, production sandbox, or distributed GPU
  scheduler. Task-context summaries are deterministic runtime/source summaries
  rather than private LLM-authored domain briefs.
- The diagnostic table is TOON-like and includes feature dependencies, subnode
  aggregates, importance, standalone score, max correlation, high-correlation counts,
  residual correlations, target-bin objective rows, group/unit objective rows when
  a group key is configured, exact additive Ridge contribution summaries,
  diagnostic global Ridge score, valid-feature-pool Ridge score, fold-weight
  stability, fold score dispersion, effective rank, and evaluated configuration
  score range. It is not the full private
  diagnostic schema. Alternative-level history is a clean-room rolling aggregate over
  the best stateful evaluation path rather than the authors' private statistic table,
  the SHAP-style fields are exact for this repo's standardized linear Ridge basis
  rather than a private implementation.
