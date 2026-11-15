# Pair Key-Value Reader R1B

Pair Key-Value Reader R1B is an isolated confirmation experiment for the frozen Twin-Task dataset. It keeps the same functional reader architecture as R1, but removes explicit pair-index supervision during training.

R1B trains only with output supervision:

```text
L_total = L_output
```

`L_output` is the negative log likelihood of the target output grid under the retrieved color probabilities. The trainer does not construct, log, or consume a query-relevant pair index. Candidate grids and labels remain evaluation-only.

This confirmation is needed before R2 because R1 used `L_retrieval`, which directly told the reader which support pair was correct. R1B asks whether the output signal alone is enough for the shared matcher to put mass on the query-matching support input.

The reader route is:

```text
Q -> shared matcher -> q
X_i -> same shared matcher -> k_i
alpha = softmax(q dot k_i / temperature)
retrieved_probs = sum_i alpha_i * one_hot(Y_i)
```

There is no canvas, paint head, query-copy head, pair-position embedding, pair-ID embedding, or bypass around `alpha`. The raw support output grid `Y_i` is the only value path.

All six support-pair permutations are tested because a correct correspondence reader must be invariant to tensor slot order after alpha is mapped back to logical support identity. Complete `(X_i, Y_i)` pairs move together.

The forced-value intervention keeps `q`, keys, logits, and alpha fixed while permuting the value grids. If output changes with the forced value, the value route is causally active. The forced-alpha intervention replaces alpha with exact one-hot choices and verifies that each selected support output grid is routed exactly.

Passing R1B does not mean ARC is solved, VRM is repaired, or raw copying is sufficient for ARC. The narrow conclusion is only that output-only supervision can train this standalone reader to solve the marker Twin correspondence unit test across seeds. That result permits moving to a separate R2 integration test; it does not replace one.

