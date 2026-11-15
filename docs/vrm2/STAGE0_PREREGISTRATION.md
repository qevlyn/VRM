# VRM v2 Stage-0 Preregistration

Status: frozen before neural-model implementation. This document specifies a falsifiable microworld experiment; it reports no model or reasoning result.

## 1. Research question

Can a task-specific executable program induced only from three support input/output pairs improve exact-match generalization to unseen ordered compositions over a compute-matched forward-only model, while causally controlling the query output?

## 2. Hypotheses

- **H1 — composition generalization:** across three preregistered model seeds, the executable-program model will exceed the forward-only twin by at least 5.0 percentage points in mean sealed exact-match accuracy, and the gain will be positive in all three seeds.
- **H2 — causal specificity:** deranged program swaps between tasks will reduce sealed exact-match accuracy by at least 10.0 points and will change the predicted grid on at least 80% of otherwise correctly predicted queries.
- **H3 — executable fidelity:** a later oracle-program interpreter will obtain at least 95% exact match on 2,000 newly generated sealed scenes.
- **H4 — hard execution:** hard execution will finish within 10.0 points of the best preregistered soft relaxation.

Failure to meet a threshold is evidence against the corresponding Stage-0 hypothesis, not evidence that a broader class of reasoning systems is impossible.

## 3. Non-goals

Stage 0 does not implement or evaluate a neural architecture, natural-language verbalizer, ARC-native training, variable output shape, long program search, handwritten solver modules, human-like reasoning, or natural-language explanations. The generator operations are data-generating mechanisms and are not proposed solver modules.

## 4. Frozen Stage-0 scope

```yaml
grid_size: 8
output_shape: fixed
support_pairs_per_task: 3
query_pairs_per_task: 1
objects_per_scene: [1, 3]
train_program_lengths: [1, 2]
dev_program_lengths: [2]
sealed_program_lengths: [2, 3]
train_seed_count: configurable
dev_seed_count: configurable
sealed_seed_count: configurable
model_sees_rule_metadata: false
variable_shape: false
natural_language_verbalizer: false
arc_native_data: false
```

Only generator, split, serialization, validation, documentation, tests, and smoke-CLI work is in scope. Any change to grid size, support count, operation registry, split policy, or leakage boundary requires a new version and a written amendment before model scoring.

## 5. Data-generating process

Each task samples one hidden ordered program and its parameters, then independently samples three support scenes and one query scene. Scenes contain one to three non-background, uniformly colored, four-connected objects on an 8×8 grid. Colors are integers 0–9 and 0 is background. Objects do not overlap or share an edge at scene creation.

The generator-only operation registry `vrm2-operations-v1` contains six operations:

1. translate the uniquely leftmost object by one sampled cardinal direction;
2. translate the uniquely topmost object by one sampled cardinal direction;
3. recolor the uniquely leftmost object to a sampled non-background color;
4. recolor the uniquely topmost object to a sampled non-background color;
5. reflect all content across the horizontal axis of its content-derived bounding box;
6. reflect all content across the vertical axis of its content-derived bounding box.

Translations are rejected if they leave the grid, overlap another object, or merge objects by edge contact. Selector ties, no-effect steps, irrelevant sampled colors, exact two-step cancellation, identical support inputs or outputs, query/support duplicates, and support sets explained by any one constituent step alone are deterministically rejected and resampled. These checks do not prove semantic identifiability: a finite support set can still admit an unenumerated program, a commuting alternative order, or a more complex extensional shortcut. Those residual ambiguities must be measured in later diagnostics.

The registry was chosen because leftmost/topmost selectors can change after translation or reflection, so ordered programs can be non-commutative, while every operation remains replayable from scene content. Reflections and translations preserve color histograms, and recoloring changes only a relation-selected object; therefore exact output location cannot be inferred from a single global color-frequency rule. Some operation pairs commute on some scenes, which is an explicitly retained source of semantic ambiguity rather than an alignment success criterion.

Parameters are derived from `(generator_version, composition_signature, episode_seed)` without the split label. Every scene uses an explicit local pseudorandom generator. No module-global random state is consumed.

## 6. Composition split

Composition signatures contain ordered operation names but not parameters. Valid signatures use no operation more than once: six length-1 signatures, 30 length-2 permutations, and 120 length-3 permutations.

- **Train:** all six atomic signatures plus the first 18 length-2 signatures after a deterministic versioned shuffle.
- **Dev:** the next six shuffled length-2 signatures.
- **Sealed:** the final six shuffled length-2 signatures plus all 120 length-3 signatures.

The shuffle seed is `20260713` under split version `vrm2-composition-split-v1`. Exact signatures are pairwise disjoint. Every atomic operation is known from train, while dev and sealed orders are unseen. Generation schedules eligible lengths round-robin and shuffles signatures locally within each length. The generator-only split manifest is machine-readable and is forbidden as model input.

## 7. Future model interfaces

The future model input may contain only the three support input/output grids and the query input grid. The query output is a supervised target, not an input. The interface must not accept episode seed, split, task ID, operation name, parameter, composition signature, generator version, split manifest, retry count, or any other generator diagnostic. The executable-program model and forward-only twin must consume byte-identical model-facing records.

## 8. Baselines

The mandatory primary baseline is a forward-only twin matched within 5% for trainable parameters, training updates, training examples, and measured query-time multiply-adds. Both systems receive identical serialized grids and seeds. Secondary baselines are identity output, support-output copying, nearest-support retrieval by input Hamming distance, and a query-only forward model. None may receive generator metadata.

## 9. Primary and secondary metrics

Primary metrics, fixed before model work, are:

1. sealed whole-grid exact-match accuracy over 2,000 tasks per seed;
2. executable-model minus compute-matched-twin exact-match difference in percentage points, reported per seed and as the three-seed mean;
3. program-swap specificity: exact-match drop and fraction of predicted grids changed after a deterministic deranged swap.

Secondary metrics are dev/train exact match, per-program-length exact match, per-cell accuracy, ground-truth-operation/code alignment, support reconstruction, invalid-output rate, and 95% Wilson confidence intervals. Alignment alone cannot establish success.

## 10. No-bypass constraints

Model-facing serialization must contain no hidden program, operation label, composition label, task ID, split label, episode seed, file-order surrogate, or generator-only metadata. Train/dev/sealed files must use the same schema. Query targets must be separated before `forward`. Generated record order must not encode composition identity: training loaders must reshuffle with a declared local seed. Source inspection and recursive forbidden-key tests are required before any model run.

## 11. Compute budget

Stage-0 data certification is limited to one CPU process, 2 GB of newly written data, and 60 minutes for a full 14,000-task train/dev/sealed generation on the reference machine. Unit and smoke defaults remain 24/8/8 tasks. A future model comparison is limited to three seeds, at most 100,000 optimizer updates per model/seed, and at most 24 GPU-hours per model/seed. The forward-only twin must match the executable model within 5% on parameters, examples, updates, and measured inference multiply-adds. Later search may use at most 3.0× the measured forward-only query-time compute unless it gains at least 5.0 exact-match points.

## 12. Seed policy

The split seed is `20260713`; default dataset seeds are train `110000`, dev `220000`, and sealed `330000`. Future model seeds are exactly `20260713`, `20260714`, and `20260715` for the primary report. Scientific task counts are configurable but must be fixed before generation and identical across compared models. A task is reproducible from generator version, split version, composition signature, and episode seed. Seed replacement after viewing sealed scores is prohibited.

## 13. Kill criteria

The Stage-0 data contract is killed immediately if either current gate fails:

1. Ground-truth generator execution is not 100% exact and deterministic.
2. Train/dev/sealed composition sets are not provably disjoint.

The following are **preregistered future gates, not results of this task**:

3. Kill if a later oracle-program interpreter is below 95% exact match on 2,000 unseen sealed scenes.
4. Kill refinement if it improves mean support reconstruction by at least 5.0 points but improves query exact match by less than 1.0 point.
5. Kill the causal-program claim if swapping programs lowers exact match by less than 10.0 points or leaves more than 20% of otherwise-correct predicted grids unchanged.
6. Kill the executable-program hypothesis if its three-seed mean does not exceed the compute-matched forward-only twin by at least 5.0 points, or if any seed has a non-positive gain.
7. Kill hard execution if it loses more than 10.0 exact-match points versus the best frozen soft relaxation.
8. Kill search if it costs more than 3.0× the forward-only baseline without at least a 5.0-point exact-match gain.
9. Kill any claimed gain that appears in only one random seed.

## 14. Expected artifacts

Stage 0 produces this preregistration, one versioned configuration, source under `vrm2/data/`, tests under `tests/vrm2/`, a model-facing JSONL dataset containing only grids, a separate generator-only episode manifest, a separate split manifest, deterministic SHA-256 digests and summary statistics, and captured test/smoke commands in the completion report. No checkpoint or neural-model artifact is expected.

## 15. Conditions for opening Stage 1

Stage 1 may open only after: all Stage-0 tests pass; two independent runs produce byte-identical datasets and manifests; all grid/program replays are 100% exact; composition intersections are empty; a 2,000-task-per-split certification run terminates within retry and compute limits; recursive leakage audit finds zero forbidden fields; and the later oracle-interpreter diagnostic reaches at least 95% sealed exact match. Meeting these conditions permits model design; it does not establish reasoning. Stage 1 is not opened by this document or by the Stage-0 implementation.

### Oracle typed-program diagnostic addendum

The oracle diagnostic uses typed schema `vrm2-typed-program-v1`: integer factors for operator, selector, argument type, and argument value. This is generator/diagnostic metadata and is never added to normal Stage-0 model-facing JSONL. The diagnostic trains only on registered train signatures and certifies on a pinned, disjoint corpus of 1,000 sealed length-2 plus 1,000 sealed length-3 scenes. Each of the three preregistered seeds must independently reach the existing 95% oracle threshold for length-2, length-3, and aggregate exact match. This addendum does not change any preregistered threshold or open Stage 1.
