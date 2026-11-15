# SequentialResidualInductorV1 preregistered pilot protocol

## Scope and frozen boundaries

This is a one-seed architectural pilot, not certification. The only model seed is
`20260713`. The certified `OracleInterpreterV1`, typed schema
`vrm2-typed-program-v1`, Stage-0 corpus, composition split, operation registry,
scene generator, serializers, frozen V0 checkpoint, and prior evaluation corpora
are immutable. The interpreter checkpoint is evaluated before and after major
runs by archive, model-state, parameter-state, and buffer-state hashes; it is put
in evaluation mode and all parameters are frozen.

The inductor API accepts three support input/output grid pairs and their masks.
It receives no query, target output, program metadata, signature, split, task ID,
or generator metadata. Evaluation has a hard `torch.int64` serialization boundary:
Stage A maps support pairs to a typed program, and a fresh Stage B process receives
only query grids, hard programs, and the frozen interpreter checkpoint.

## Architecture

The V0 shared pair encoder and masked, permutation-invariant set aggregator are
retained. The V0 length head and position-specific heads are replaced by one
shared GRU decoder cell, one shared residual update, one shared head for each
typed field, a residual initializer, typed-instruction embeddings, and a learned
begin-program embedding. There are no position embeddings, separate length head,
special step-3 branch, operation-specific module, or selector-specific module.
The total trainable parameter budget is fixed at 1,350,000.

NULL is the exclusive termination mechanism. The first NULL determines program
length and forces a canonical NULL suffix. Training teacher-forces all three
positions, including NULL suffix positions, with the unweighted sum of equal
operator, selector, argument-type, and argument-value cross-entropies.
Evaluation is autoregressive; only autoregressive hard programs are executed.

## Data, optimization, and selection

Training combines deterministic length-1 episodes from the frozen Stage-0 source
with byte-identical V0 length-2 episodes. Length-3 targets are prohibited. Every
three-support episode creates the same three leave-one-support-out folds as V0;
the omitted support and query do not contribute to induction input or loss.

The frozen main schedule is AdamW, batch size 64, learning rate `3e-4`, betas
`[0.9, 0.95]`, weight decay `0.01`, cosine decay to 0.1 of the initial rate,
500 warm-up steps, gradient norm clipping at 1.0, deterministic BF16 on CUDA,
10,000 maximum steps, 6,000 minimum steps, evaluation every 1,000 steps, and
dev-only patience of three evaluations. Checkpoints rank lexicographically by:

1. dev L2 query exact;
2. dev autoregressive behavioral-program exact;
3. dev autoregressive typed-program exact;
4. train-evaluation query exact;
5. earlier step.

No sealed corpus participates in selection. The main, no-residual, and shuffled-
instruction-order runs use the same seed and schedule. The no-residual run keeps
the shared decoder but returns the unchanged initial residual after every step.
The shuffled run reverses the two active instructions in every L2 training target
and leaves L1 targets unchanged. These are the only training ablations.

## Coverage-qualified L3 corpus

Before training, seed `830000` deterministically generates exactly 512 evaluation-
only L3 episodes, each with three supports and one query sharing one exact program.
Eligibility requires frozen held-out L3 membership, instruction/category coverage
in L1/L2 training, a represented L2 ordered subsequence, absence of the complete
L3 signature from training, and deterministic generation of four distinct valid
scenes. Usage is balanced across eligible composition signatures. Fingerprints
must not overlap train, train-evaluation, dev, or original sealed data. A byte-
identical second generation, eligibility audit, corpus hash, rejection counts,
and access audit are required. Failure to construct 512 episodes is
`INDUCTOR_V1_INVALID_L3_CORPUS`.

The coverage corpus remains sealed throughout training, loss, selection, stopping,
and all Phase A evaluation. It can be opened only after a persisted passing Phase A
verdict. A failed Phase A ends the experiment without inspecting coverage content.

## Phase A gates

The main, no-residual, and shuffled-order checkpoints are evaluated on train-
evaluation, dev, frozen sealed L2, and the underpowered original seven L3 episodes.
The main model passes only when all conditions hold:

- sealed L2 query exact is at least 0.70;
- improvement over frozen V0 sealed L2 is at least 0.15;
- its margin over no-residual sealed L2 is at least 0.05;
- shuffled-order sealed L2 is at least 0.20 below main;
- support-permutation hard-program identity is exactly 1.0;
- every structural validity check passes.

Failure returns `INDUCTOR_V1_REDESIGN_REQUIRED` and prohibits Phase B.

## Phase B gates

Only after Phase A passes, evaluate frozen V0, V1 main, and V1 no-residual on the
512-episode L3 corpus and run main-model controls. Passing requires main query exact
at least 0.50, improvement over V0 at least 0.25, oracle exact at least 0.95,
support-swap/output-shuffle/null-support exact drops each at least 0.30, their
output-change rates each at least 0.50, and support-permutation hard-program
identity exactly 1.0. Passing returns
`INDUCTOR_V1_PILOT_CONTINUATION_GATE_PASSED`; otherwise it returns
`INDUCTOR_V1_L3_EXTRAPOLATION_FAILED`. No second seed is authorized.

## Freeze and invalidation

Implementation, corpus generation, configs, this protocol, smoke, and the full
test suite are completed before source freeze. Scientific source, configs, tests,
training episode artifacts, coverage artifacts, Oracle V1, and frozen V0 are
hashed. Any scientific source/config change after the pilot starts returns
`INDUCTOR_V1_INVALID_SOURCE_CHANGED`; any hard-boundary, executor-freeze, data-
isolation, parameter-budget, single-seed, or structural violation returns
`INDUCTOR_V1_INVALID`.
