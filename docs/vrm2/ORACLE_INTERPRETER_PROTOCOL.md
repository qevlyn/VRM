# VRM v2 Oracle Typed-Program Interpreter Protocol

Status: frozen executor diagnostic. This protocol does not implement a program inductor or open Stage 1.

## 1. Research question

Given the correct hard typed generator program, can one shared neural interpreter execute it on unseen 8×8 grids and generalize to ordered compositions excluded from training?

## 2. Isolated variable

The diagnostic supplies the ground-truth program and therefore removes program induction from the experiment. The model receives only an input grid and a sequence of four-integer typed instructions. Support grids, targets, task identifiers, split labels, composition signatures, operation strings, manifests, and intermediate target grids are not model inputs.

## 3. Non-claims

A pass establishes only executor competence in this finite microworld. It does not demonstrate reasoning, ARC competence, natural-language explanation, program induction, search, or human-like abstraction. A failure blocks work on the future inductor.

## 4. Typed schema

Schema version: `vrm2-typed-program-v1`.

Each generator-only `OperationStep` now carries a factorized `ProgramInstruction` while retaining its readable name and parameters for generator replay and diagnostics. Normal Stage-0 serialization is unchanged. Because generator semantics, sampling, model-facing records, generator manifests, and split assignments remain byte-compatible, the operation-registry and split versions are not incremented.

The neural interface is:

```text
[operator_id, selector_id, argument_type, argument_value]
```

| Factor | Integer IDs |
|---|---|
| operator | `0=NULL`, `1=TRANSLATE`, `2=RECOLOR`, `3=REFLECT` |
| selector | `0=NULL`, `1=LEFTMOST_OBJECT`, `2=TOPMOST_OBJECT`, `3=ALL_CONTENT` |
| argument type | `0=NULL`, `1=CARDINAL_DIRECTION`, `2=TARGET_COLOR`, `3=AXIS` |
| direction value | `0=NULL`, `1=UP`, `2=DOWN`, `3=LEFT`, `4=RIGHT` |
| color value | integers `1..9` |
| axis value | `0=NULL`, `1=ROW`, `2=COLUMN` |

Readable operation names occur only in generator and diagnostic records. Oracle batches expose exactly `input_grid` and `typed_program_ids` to `forward`.

## 5. Architecture

`OracleInterpreterV0` embeds 64 grid cells, learned row/column positions, and four workspace tokens at hidden width 96. Four factor embeddings are fused into one instruction vector. A two-layer instruction-conditioned block combines global multi-head attention, local 3×3 convolutions, and an MLP. FiLM-style conditioning is learned from the instruction vector.

The same `SharedExecutionBlock` instance is called for every step:

```text
state_(t+1) = F_theta(state_t, instruction_t)
```

The block has no operation-specific module or Python branch. A shared head predicts 10-color logits after each step. Hidden spatial/workspace state and a learned feedback projection of the model's own predicted color distribution carry forward. No target or intermediate target is fed back. Programs are hard integer tensors; no Gumbel-Softmax or program mixture is used.

Default parameter count is 735,787, below the 5,000,000 limit.

## 6. Data policy

Training uses only query scenes generated from registered Stage-0 train signatures: all atomic programs and registered train length-2 programs. Train-signature evaluation uses new seeds. Dev uses registered held-out length-2 signatures.

The pinned sealed corpus contains exactly 2,000 never-trained examples generated with seed `640000`: 1,000 registered sealed length-2 compositions and 1,000 registered sealed length-3 compositions, balanced across signatures as evenly as integer counts permit. Training begins at seed `610000`, new-scene train evaluation at `710000`, and dev evaluation at `720000`; their allocated ranges do not overlap. The canonical JSONL and separate manifest record generator/registry/split/schema versions, Python/PyTorch/CUDA runtimes, seed, SHA-256, byte count, signature counts, and length counts. Exact example fingerprints must have empty intersection with training data.

Oracle JSONL may contain targets and diagnostic metadata. `OracleBatch.model_inputs()` exposes only the input grid and numeric typed program. Intermediate grids are target tensors only.

## 7. Objective

For a program of length `T`:

```text
loss = CE(final_logits, final_target)
     + lambda_intermediate * mean_t CE(step_logits_t, intermediate_target_t)
```

The primary configuration freezes `lambda_intermediate = 0.5`. The registered ablation config sets it to `0.0`. Mixed precision, AdamW, cosine decay with warmup, gradient clipping at 1.0, deterministic seeds, and bounded early stopping are used. Intermediate targets never enter `forward`.

## 8. Controls

- Correct: original input plus correct typed program.
- Random: a length-matched program sampled from a different registered signature.
- Swap: a deterministic within-length derangement between different signatures.
- Reverse: length-2/3 instructions reversed only when generator replay is valid and produces a non-equivalent target.
- Null: an equal-length sequence of four zero IDs.
- Intermediate: exact match after each execution step.

Controls report exact match and output-change rate relative to correct-program predictions. Certification operationalizes "substantial" as at least a 20-point exact-match drop and at least a 50% output-change rate for every wrong-program control in every seed.

## 9. Pass/fail gate

Every one of seeds `20260713`, `20260714`, and `20260715` must independently achieve:

```text
sealed_len2_exact >= 0.95
sealed_len3_exact >= 0.95
sealed_all_exact >= 0.95
```

The implementation audit must also pass shared-weight, no-generator-import, no-teacher-forcing, order, data-disjointness, and control gates. Mean, population standard deviation, and minimum are reported across seeds. A pass is not inferred from one seed.

## 10. Compute budget and staged run policy

Level A unit and smoke runs are unconditional. Level B is one seed with at most 735,787 parameters, mixed precision, one RTX 4070 Ti SUPER, and four wall-clock hours. It hard-stops certification below 0.70 sealed length-2 or 0.50 sealed length-3. To make "plausibly approaching" operational before observing results, certification requires pilot train-signature exact at least 0.95, sealed length-2 at least 0.90, and sealed length-3 at least 0.85.

Level C uses one frozen certification config, the same pinned sealed corpus, and all three preregistered seeds without per-seed tuning. Search, inductor gradients, and generator execution are absent.

## 11. Reproducibility and commands

Generate the pinned sealed corpus:

```bash
python -m vrm2.oracle.cli generate-sealed --config configs/vrm2/oracle_interpreter_cert.yaml --output artifacts/vrm2/oracle/runtime/sealed_corpus.jsonl
```

Train one seed:

```bash
python -m vrm2.oracle.cli train --config configs/vrm2/oracle_interpreter_pilot.yaml --seed 20260713 --sealed-corpus artifacts/vrm2/oracle/runtime/sealed_corpus.jsonl --checkpoint artifacts/vrm2/oracle/runtime/pilot/oracle_interpreter_seed_20260713.pt --report artifacts/vrm2/oracle/runtime/pilot/oracle_interpreter_seed_20260713.json
```

Evaluate and run controls:

```bash
python -m vrm2.oracle.cli evaluate --config configs/vrm2/oracle_interpreter_pilot.yaml --checkpoint artifacts/vrm2/oracle/runtime/pilot/oracle_interpreter_seed_20260713.pt --sealed-corpus artifacts/vrm2/oracle/runtime/sealed_corpus.jsonl --output artifacts/vrm2/oracle/runtime/pilot/evaluation.json
python -m vrm2.oracle.cli controls --config configs/vrm2/oracle_interpreter_pilot.yaml --checkpoint artifacts/vrm2/oracle/runtime/pilot/oracle_interpreter_seed_20260713.pt --sealed-corpus artifacts/vrm2/oracle/runtime/sealed_corpus.jsonl --seed 20260713 --output artifacts/vrm2/oracle/runtime/pilot/controls.json
```

Conditionally certify:

```bash
python -m vrm2.oracle.cli certify --config configs/vrm2/oracle_interpreter_cert.yaml --sealed-corpus artifacts/vrm2/oracle/runtime/sealed_corpus.jsonl --pilot-report artifacts/vrm2/oracle/runtime/pilot/oracle_interpreter_seed_20260713.json --output-dir artifacts/vrm2/oracle/runtime/certification
```

The certification command reads the pilot report and refuses to train if the frozen continuation thresholds are not met.

## 12. Condition for implementing the future inductor

The program inductor remains forbidden unless the final verdict is `ORACLE_INTERPRETER_PASS`, all three seeds pass every 0.95 gate, the controls demonstrate causal program dependence, the no-bypass audits pass, and train/sealed overlap is zero. Any other verdict requires executor or data redesign first.
