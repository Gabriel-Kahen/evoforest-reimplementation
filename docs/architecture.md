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
- `evoforest_arch.primitives`
  - A compact primitive library for time-series change statistics, callable gates,
    activations, output compositions, and Ridge fitting rules.
- `evoforest_arch.source`
  - Trusted source-backed lambda alternatives that mirror the paper's YAML
    mutation representation.
- `evoforest_arch.evaluator`
  - Capped configuration enumeration, optional global refinement, Ridge
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
    records, deterministic test clients, and an opt-in generic HTTP JSON LLM
    client.
- `evoforest_arch.task_context`
  - Deterministic cached task-context summaries with tensor inventory, target
    balance, scorer mechanics, and implementation constraints.
- `evoforest_arch.refinement`
  - Optional PyTorch `nn.ParameterDict` plus L-BFGS refinement for differentiable
    fixed paths, with deterministic NumPy fallback for unsupported paths.
- `evoforest_arch.mutations`
  - YAML-style mutation documents, mutation specs, removals, appended globals,
    source-backed alternatives, and a registry-driven mutation engine.
- `evoforest_arch.maintenance`
  - Graph cleanup for duplicate alternatives, unreachable nodes, empty nodes,
    unused globals, and DAG validity.
- `evoforest_arch.evolution`
  - Open-ended mutation loop, island loop, archive, events, checkpoints, and
    memoranda, sequential and asynchronous island execution, failed-candidate
    feedback, plus rejected-candidate salvage.
- `evoforest_arch.seed`
  - A small seed graph that demonstrates intermediate alternatives, callable nodes,
    output alternatives, fitting nodes, and global parameters.

## Paper-Specific Semantics

- `input` nodes read data supplied by the task.
- `intermediate` nodes contain competing tensor-producing alternatives.
- `callable` nodes contain competing alternatives that return reusable functions.
- `output` alternatives are not selected by configuration. Every output alternative
  is evaluated for a fixed configuration and stacked into the Ridge feature matrix.
- `fitting` nodes are selected by configuration. `ridge_w` returns per-sample
  weights, and `ridge_g` returns a residual-to-weight rule used in iterative
  reweighted least squares.
- The evaluator enumerates reachable intermediate, callable, and fitting alternatives
  up to `max_configurations`; graph fitness is the best configuration ROC-AUC.
- Configuration search shares a cache across evaluated candidates. Cache keys encode
  the selected ancestor subpath for each alternative, so a reused intermediate is
  computed once when all of its ancestors agree, but recomputed when a parent
  alternative changes.
- The Ridge readout standardizes features, selects alpha from a log-scale grid using
  a leave-one-out leverage criterion inside each fold, records fold AUC dispersion,
  and summarizes the evaluated configuration AUC range.
- `ridge_g` residual rules are applied through an explicit IRLS loop: after the
  initial Ridge fit, residuals are converted to weights and Ridge is refit for the
  configured number of `irls_steps`, with per-fold iteration diagnostics recorded.
- After a configuration is scored, the evaluator fits a diagnostic global Ridge model
  on the best configuration's feature matrix. Feature diagnostics include exact
  additive linear contribution summaries for both the out-of-fold CV models and the
  global diagnostic model. The contribution basis is the standardized Ridge term
  `z_j * coefficient_j`, which reconstructs each linear prediction with the model
  intercept up to numerical precision.
- For stateful evaluations, the best configuration updates persistent
  alternative-level statistics. Each alternative ages once per stateful evaluation,
  and participating alternatives accumulate rolling summaries of feature count,
  importance, standalone AUC, residual coverage, redundancy, fold-weight stability,
  and configuration AUC. These summaries are serialized in the graph and surfaced
  in feedback/TOON artifacts.
- Optional global refinement runs before configuration scoring, then globals are
  frozen for Ridge evaluation. When PyTorch is installed and the selected path has
  torch evaluators, trainable globals are materialized as `nn.Parameter`s and
  optimized by L-BFGS together with an ephemeral linear readout.
- Mutation proposals follow a two-role pipeline: a scientist agent turns feature and
  subnode diagnostics into hypotheses, and an engineer agent emits structured
  YAML-style documents with rationale, new nodes, removals, appended globals, and
  additions. Additions can reference built-in primitives or, when trusted source
  mutations are explicitly enabled, source-backed `lambda ctx, values: ...`
  alternatives. The default agents are deterministic, while optional LLM-backed
  agents use the same document contract and persist their prompts and responses.
  In island mode, LLM scientist calls default to the paper's fixed schedule
  `(0.35, 0.5, 0.6, 0.75)` by island index, and LLM engineer calls default to
  zero temperature for mutation synthesis.
- At run startup, a task-context summary is written to `task_context.md` and injected
  into default LLM prompt builders. It records the runtime tensor inventory, target
  balance, scorer mechanics, and implementation constraints so mutations are
  grounded in the actual task interface.
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
  islands. The async island runner
  evaluates one candidate per island concurrently in local worker threads and
  processes successes or failures as they arrive. LLM-backed island runs preserve
  the fixed scientist-temperature schedule independently of candidate evaluation
  completion order.
- Memoranda are sectioned into `[OUTCOME HISTORY]`, `[STATE]`, `[WHAT WORKS]`,
  `[WHAT FAILED]`, and `[ERROR LOG]`. The clean-room writer is deterministic and
  hypothesis-free: it summarizes observed outcomes and diagnostics without adding
  new recommendations.

## Design Principles

The code keeps the graph architecture explicit. Alternatives are first-class objects,
configuration is separate from the graph, and the evaluator produces structured
feedback rather than only a scalar score.

This is not intended to duplicate the authors' private implementation. It is a
faithful software substrate for the paper's architectural pattern.

Known approximations:

- The paper's private implementation is PyTorch-first. This repo supports a PyTorch
  L-BFGS path for the included differentiable primitives, but arbitrary new
  clean-room primitives need a `torch_fn` to participate in that path. Unsupported
  paths fall back to a deterministic NumPy coordinate refiner.
- The paper's long run used asynchronous GPU islands; this repo includes
  asynchronous local worker islands with the paper-style scientist temperature
  schedule, but not a multi-GPU distributed scheduler.
- Cross-configuration caching assumes alternatives are deterministic over their
  parents, inputs, and fixed globals during one evaluator pass. Trusted source-backed
  alternatives that deliberately perform side effects are outside that assumption.
- The LLM scientist/engineer loop is represented by deterministic local agents by
  default, plus optional HTTP JSON LLM-backed agents. Source-backed alternatives
  recreate the paper's lambda-style graph edits for trusted local use, but this
  does not include the authors' private model stack, exact prompt corpus, or
  production sandbox/repair system. Memorandum updates are deterministic summaries
  of observed events and diagnostics rather than private LLM-authored prose.
  Task-context summaries are deterministic runtime summaries rather than private
  LLM-authored domain briefs.
- The diagnostic table is TOON-like and includes feature dependencies, subnode
  aggregates, importance, standalone AUC, max correlation, high-correlation counts,
  residual correlations, class effect size, exact additive Ridge contribution
  summaries, diagnostic global Ridge AUC, fold-weight stability, fold AUC dispersion,
  effective rank, and evaluated configuration AUC range. It is not the full private
  diagnostic schema. Alternative-level history is a clean-room rolling aggregate over
  the best stateful evaluation path rather than the authors' private statistic table,
  the SHAP-style fields are exact for this repo's standardized linear Ridge basis
  rather than a private implementation, and `n_features_global` is a clean-room
  evaluated configuration feature-column count rather than the authors' exact private
  feature pool accounting.
