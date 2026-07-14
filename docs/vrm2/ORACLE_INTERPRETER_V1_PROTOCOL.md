# OracleInterpreterV1 Rollout-State Canonicalization Protocol

## Motivation and bounded question

The frozen one-seed `OracleInterpreterV0` pilot reached 0.994 train new-scene exactness and
0.978 dev length-2 exactness, but only 0.901 sealed length-2 and 0.623 sealed length-3 exactness.
On sealed length-3 cases its intermediate exactness was 1.000, 0.963, and 0.623 after steps 1,
2, and 3. Program interventions remained strongly causal (correct 0.7540 versus random 0.0030,
swap 0.0030, reverse 0.0038, and null 0.0000). The preregistered V1 hypothesis is that V0's
continuously carried spatial and workspace activations drift away from a representation determined
only by the decoded grid.

This diagnostic changes one architectural principle: V1 discards all per-example execution
activations at every instruction boundary and reconstructs them from a hard canonical predicted
grid and the next typed instruction. It does not add a program inductor, operation-specific module,
teacher-forced main path, new task, new loss, or Stage 1 component.

## Canonicalization invariant

The public model inputs remain exactly `input_grid` and `typed_program_ids` under typed schema
`vrm2-typed-program-v1`. Integer input grids are converted to color-last one-hot tensors with ten
fixed colors. After a predicted step, evaluation uses `argmax` over color logits followed by hard
one-hot encoding. Non-winning logits and probabilities are unavailable to the next transition.

Training uses the fixed straight-through categorical estimator:

```python
soft = softmax(logits, dim=color_dim)
hard = one_hot(argmax(soft))
canonical_st = hard + soft - soft.detach()
```

The forward value is hard one-hot (up to floating-point cancellation tolerance), while later-step
losses receive a softmax surrogate gradient through earlier logits. Normal training and evaluation
never replace a prediction with a ground-truth intermediate grid.

For every instruction, the shared color embedding/projection and row/column embeddings rebuild 64
spatial tokens from the canonical grid. Workspace activations are freshly computed from shared
learned seed parameters plus projections of the current grid summary and current typed instruction.
The same `SharedExecutionBlock` object is then applied. Previous spatial tokens, convolutional
residuals, attention activations, logits, soft probabilities, and workspace activations are neither
arguments to nor retained by the next transition. The learned seed parameters are shared, but their
per-example activations are recomputed. There are no learned absolute execution-step embeddings.
This interface leaves no workspace or hidden-state bypass around the canonical grid.

V1 has 741,930 trainable parameters, compared with 735,787 for V0. The 6,143-parameter net increase
comes from removing V0's soft-feedback path and adding explicit fresh grid- and instruction-based
workspace initialization. Both remain below the frozen 5,000,000-parameter ceiling.

## Shared execution and losses

One instruction encoder, one grid encoder, one output head, and one structurally identical
`SharedExecutionBlock` instance are reused at all instruction positions. There is no operator,
selector, REFLECT, program-length, final-position, or third-step branch. The model predicts a grid
after every instruction. The V0 objective is unchanged: final flattened cell cross-entropy plus
`0.5` times flattened intermediate cell cross-entropy.

## Frozen assets and policy

- Stage-0 20-task SHA-256: `d5bf7360ccb287bb5b6a97fb78bd64d4916b2950b7da38ba48e790c976372208`
- Typed schema: `vrm2-typed-program-v1`, with all operator, selector, argument-type, and value IDs unchanged
- Sealed corpus seed/count: 640000 / 2,000 (1,000 length-2 and 1,000 length-3)
- Sealed SHA-256: `b91381281584011b4521cbf2479a3ba6d8ed6c2477dae80506b904e9c5352ba9`
- Training/train-evaluation/dev/model seeds: 610000 / 710000 / 720000 / 20260713
- Hard-stop floors: sealed length-2 0.70 and sealed length-3 0.50
- One-seed continuation threshold: sealed length-3 0.85
- Later certification target: 0.95 (not run in this task)

The pilot retains V0's corpus sizes and distributions, batch size 256, AdamW optimizer and betas,
learning-rate schedule and warmup, deterministic BF16 policy, evaluation every 1,000 steps,
dev/train-signature checkpoint ordering, early-stopping rule, and 12,000-step maximum. Sealed
performance is never used for checkpoint selection. The only pilot/model seed is 20260713.

## Reproducible workflow

The smoke configuration runs 200 updates using the V1 model path and its separate runtime namespace.
It is used only to verify execution, finite gradients, checkpointing, and falling loss. The one-seed
pilot uses `configs/vrm2/oracle_interpreter_v1_pilot.yaml` and writes under
`artifacts/vrm2/oracle/runtime/v1/pilot_seed_20260713/`. V0 artifacts are not overwritten.

The reproducible checks are:

1. Run `python -m pytest tests/vrm2 -q`.
2. Regenerate the pinned sealed corpus through `generate-sealed` using the certification config and
   verify the exact frozen SHA-256.
3. Run one V1 smoke with seed 20260713.
4. Run exactly one V1 pilot with seed 20260713.
5. Run `evaluate`, `controls`, and `rollout-diagnostics` on the selected V1 checkpoint.
6. Do not invoke `certify`.

## Diagnostic modes

Free rollout is the primary result. `rollout-diagnostics` additionally reports, on all sealed
length-3 examples: per-step free exactness; evaluation-only oracle reset from the correct current
grid at each transition; explicit integer prediction-reset equivalence to the normal boundary path;
and mean cell entropy and winning-class probability by step. Oracle reset is diagnostic only and
cannot affect training, checkpoint selection, or the continuation verdict. Existing correct,
random, swap, reverse, and null program controls retain their frozen sample and eligibility rules.

## Results and verdict

The 200-step smoke ran the V1 path with seed 20260713. Total loss decreased from 3.211958 at step 1
to 0.464561 at step 200 (final 2.170879 to 0.321217; intermediate 2.082158 to 0.286687), and the
step-200 V1 checkpoint reloaded successfully.

The one-seed pilot stopped under the unchanged early-stopping rule at step 10,000; its best
train/dev-selected checkpoint was step 7,000. It produced:

| Metric | V0 | V1 | Delta |
| --- | ---: | ---: | ---: |
| Train new-scene signatures | 0.994 | 0.995 | +0.001 |
| Dev length-2 | 0.978 | 0.994 | +0.016 |
| Sealed length-2 | 0.901 | 0.996 | +0.095 |
| Sealed length-3 | 0.623 | 0.993 | +0.370 |
| Sealed aggregate | 0.762 | 0.9945 | +0.2325 |

Sealed length-3 intermediate exactness was 1.000, 0.997, and 0.993. Oracle-reset exactness was
1.000, 0.997, and 0.996 (0.997667 over all transitions), leaving only a 0.003 step-3 gap from free
rollout. Explicit integer prediction reset matched the normal V1 transition exactly for all 2,000
boundaries, including a maximum absolute logit difference of zero. Mean cell entropy by step was
0.000010385, 0.000015796, and 0.000041844; mean winning probability was 0.99999625, 0.99999576,
and 0.99997855.

On the frozen 1,000-example control sample, exactness was 0.991 correct, 0.003 random, 0.005 swap,
0.000 reverse (261 eligible), and 0.000 null. Output-change rates versus the correct prediction were
0.997, 0.995, 1.000, and 1.000 for random, swap, reverse, and null.

Data preparation took 10.365904 seconds; training took 237.281071 seconds; total wall time was
249.117340 seconds at 10,535.826 training examples/second. Peak allocated VRAM was 553,631,744
bytes, and the checkpoint was 3,000,965 bytes. Sealed aggregate inference was 0.052806 ms/grid.
FLOPs were not reliably measured.

There were no deviations from the frozen V0 data, seed, optimizer, schedule, precision, loss,
evaluation cadence, checkpoint selection, maximum-step, control, or decision policy. The intended
V1 architecture and its resulting 6,143-parameter increase were the only experimental change.
Because free-running sealed length-3 exactness was 0.993, the preregistered result is
`PILOT_CONTINUATION_GATE_PASSED`. Three-seed certification was not run.
