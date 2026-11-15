# SupervisedTypedProgramInductorV0 frozen one-seed protocol

## Question and boundary

This bounded diagnostic tests whether two or three support input/output pairs identify a hard `vrm2-typed-program-v1` program which the immutable certified `OracleInterpreterV1` can execute on a fourth scene. The inductor receives only support grid pairs. Its API has no query input. The support-free execution process receives only a query grid and a hard integer program; it has no support input and never imports the episode support loader.

This is a supervised upper bound, not Stage 1, program search, unsupervised discovery, native ARC evaluation, or differentiable execution. Generator program factors are targets only. No gradient crosses the hard typed-program boundary, and the certified interpreter is always in evaluation mode with every parameter frozen.

## Frozen data and episode construction

The frozen sources are Stage-0 SHA-256 `d5bf7360ccb287bb5b6a97fb78bd64d4916b2950b7da38ba48e790c976372208`, train/train-eval/dev seeds `610000`/`710000`/`720000`, and the 2,000-example sealed corpus at seed `640000` with SHA-256 `b91381281584011b4521cbf2479a3ba6d8ed6c2477dae80506b904e9c5352ba9`. Assembly seeds are train `811000`, train-eval `812000`, dev `813000`, and sealed `814000`.

Examples are grouped by exact active typed-program signature, groups and examples are canonically sorted, each group receives its own deterministic seeded shuffle, and the result is chunked without reuse into three supports plus one query. Duplicate input grids and remainders below four are discarded. Manifests freeze counts, per-signature counts, support/query example and input fingerprints, artifact hashes, and cross-split overlap.

The V0 prediction vocabulary is explicitly lengths 2 and 3. Therefore train, train-evaluation, and dev episode assembly admits only length-2 examples; frozen length-1 source examples are retained unchanged but reported as structurally ineligible discards. Sealed L2 and L3 are assembled separately. Consequently length 3 has no direct supervised training examples and remains a strict zero-shot length/composition diagnostic. Support count is fixed at three.

## Model and objective

`SupervisedTypedProgramInductorV0` uses a shared joint grid-pair encoder with a 32-dimensional color embedding, hidden width 128, two spatial residual blocks, two four-head transformer blocks, and four learned pair-summary slots. A shared pair projection followed by masked mean aggregation is permutation invariant and uses no support-index embedding. A task MLP feeds separate length, operator, selector, argument-type, and argument-value heads for each of three instruction positions. It is not a classifier over complete signatures.

The trainable parameter ceiling is 2,000,000. Hard materialization uses categorical argmax and schema-only masks. Active instructions permit only registered operator-specific selector, argument-type, and value IDs; a length-2 prediction has a canonical all-NULL third instruction. The interpreter receives `torch.long` IDs, never logits, probabilities, or inductor states.

Every training episode produces the three fixed leave-one-support-out folds. The omitted pair and query are absent from the inductor input. The same target program supervises each fold with unit-weight cross-entropy for length and each active instruction field; inactive padded fields are masked. There is no execution loss or class reweighting.

## Frozen optimization and selection

The only model seed is `20260713`. The pilot uses AdamW, batch size 64 episodes (192 LOO views), learning rate `3e-4`, betas `(0.9, 0.95)`, weight decay `0.01`, 500-step warmup, cosine decay to 0.1 of the initial rate, deterministic BF16, gradient clipping 1.0, evaluations every 1,000 steps, minimum 6,000 and maximum 10,000 steps, three non-improving dev evaluations for early stopping, and a four-hour wall-clock limit. The smoke uses the same scientific architecture, optimizer, loss, data, and gates but runs 100 steps with its declared bounded cadence.

Checkpoint selection is the lexicographic maximum of dev query exact, dev behavioral-program exact, dev typed-program exact, train-evaluation query exact, and negative step so ties select the earlier checkpoint. Sealed data is not loaded by training or selection. The full ranking table is persisted.

## Evaluation and controls

Stage A loads supports and the selected inductor checkpoint and writes only episode ID, hard length and padded typed IDs, per-field confidence, and checkpoint hash. Stage B is a fresh support-free process that loads query-only records, hard programs, and certified V1. It reports deterministic semantic results separately from nondeterministic timing records.

Metrics include query and oracle exactness, exact typed program, length, per-step instruction, all four factor fields, behavioral support execution, and three-fold LOO program/execution transfer. Controls are correct supports, support permutation, support-set swap across target signatures, within-episode output shuffle, null support, the training-only most-frequent complete-program prior, oracle program, and post-induction program swap.

## Frozen executor and source freeze

The selected checkpoint is `artifacts/vrm2/oracle/runtime/v1/certification/oracle_interpreter_seed_20260713.pt`, step 7,000, 741,930 parameters, with model-state SHA-256 `4cf0b68d675c87fcbff9ac3613801017e9cba47acebe9e86f77137037ff6aae0`. Ordered parameter and buffer hashes, archive hash, frozen flags, and evaluation mode are audited before and after the pilot.

After implementation, assembly, smoke, and all tests, all inductor scientific source/config/protocol/test files and all episode artifacts are hashed. The one-seed pilot then runs without source or config changes. Any change yields `INDUCTOR_V0_INVALID_SOURCE_CHANGED`.

## Frozen gates and verdicts

Structural validity requires all tests, episode/no-bypass integrity, unchanged source/episodes/interpreter, exact support-permutation program and prediction identity, schema-valid predictions, and byte-identical repeated induction and execution semantic artifacts.

Hard-stop floors are sealed L2 0.70, L3 0.50, aggregate 0.60, and improvement over the most-frequent prior 0.10. Continuation requires L2 0.90, L3 0.85, aggregate 0.875, prior improvement 0.30, query-exact drops of 0.30 for support swap/output shuffle/null support, and output-change rates of 0.50 for the same three controls. Full-precision values are used.

The frozen verdict vocabulary is `INDUCTOR_V0_PILOT_CONTINUATION_GATE_PASSED`, `INDUCTOR_V0_REDENIGN_REQUIRED` (the task-specified spelling), `INDUCTOR_V0_INVALID`, and `INDUCTOR_V0_INVALID_SOURCE_CHANGED`. No second seed or follow-on experiment is run in this task.
