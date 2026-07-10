# Pair Key-Value Reader R1

Pair Key-Value Reader R1 is a learned associative-memory unit test for the frozen Twin-Task dataset. It is not a VRM V1.1 modification, not an ARC solver, not test-time training, and not a task-adaptation loop.

In this experiment:

- Query `Q` is encoded as the lookup query.
- Each support input `X_i` is encoded as a key `K_i`.
- Each paired support output `Y_i` remains a raw value grid `V_i`.
- The model computes attention over support pairs and retrieves `sum_i alpha_i * one_hot(Y_i)`.

The query encoder and support-input encoder are the same module with shared weights. Support outputs are never encoded into a global value vector; they stay aligned to the same support-pair axis as the keys, so selecting key `K_i` routes value grid `V_i`.

Pair position is intentionally excluded. R1 should select by query-to-support-input correspondence, not by a memorized tensor slot. Training and evaluation may permute complete `(X_i, Y_i)` pairs, but the input-output relation inside each pair must move as a unit.

The matcher uses foreground-only tokenization: non-background cells are represented by color, row, and column embeddings, then read by a small shared self-attention encoder. This avoids letting the mostly black 7x7 background dominate the matching signal.

The retrieval loss is Twin-only supervision for this synthetic unit test. It teaches the reader which support pair is query-relevant, but it is not a general ARC training objective and should not be treated as evidence that ARC labels expose such pair indices.

If R1 passes, the only supported conclusion is:

> The standalone learned reader can solve the marker Twin correspondence unit test.

It does not prove VRM is repaired, ARC is solved, raw support-output copying is sufficient for ARC, or that R1 should be inserted into V1.1. A later R2 integration test is still required to test whether a correspondence reader can be connected to VRM without breaking the broader architecture and evaluation contract.

