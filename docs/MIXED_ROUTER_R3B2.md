# R3B-2 Frozen Expert Mixed Router

R3B-2 differs from R3A by replacing the transform oracle with the frozen R3B-0 neural transform expert. The router now selects between two real frozen neural branches: the R1B retrieval reader and the R3B-0 support-conditioned transform pointer.

Both experts are frozen so the experiment isolates boundary routing rather than expert improvement. Reader and transform parameters stay in eval mode with `requires_grad = False`; only the new evidence gate is optimized.

Training uses a soft probability mixture:

```text
soft_probs = g * reader_probs + (1 - g) * transform_probs
```

Pass gates use hard routing, where `g >= 0.5` selects the reader and `g < 0.5` selects the transform. This separates differentiable optimization from the actual discrete route decision being audited.

Both branches are evaluated for every training episode. The unselected branch still contributes to the soft-mixture loss, avoiding the hard top-k failure mode where an unselected expert receives no signal about whether the gate should have chosen it.

The gate receives only evidence summaries from the frozen branches: sorted retrieval logits, sorted alpha weights, match margins, q-to-key distances, confidence/entropy summaries, transform context statistics, pointer confidence, and transform entropy summaries. It does not receive targets, family labels, transform labels, raw support outputs, raw expert output grids, query-type descriptors, split IDs, episode IDs, or pair positions.

Branch confidence is diagnostic evidence, not proof of branch correctness. R3B-2 therefore reports route-utility agreement and routing regret after inference using evaluation-only branch NLLs. Those values are never used as training labels or auxiliary losses.

Hard-negative examples matter because they give the reader tempting near matches without an exact `X_i = Q` retrieval solution. Passing them means the gate learned to reject plausible retrieval shortcuts when the support evidence implies a geometric transform.

Same-query support counterfactuals matter because they keep the query fixed while changing only the support evidence. Passing them shows routing is support-sensitive rather than a query-type shortcut.

Boundary examples are audit-only. They are generated separately, partitioned only after frozen branch inference, and never contribute examples, labels, thresholds, losses, or auxiliary signals to gate training.

`BOTH_GOOD` and `BOTH_BAD` boundary partitions do not have unique route labels. For `BOTH_GOOD`, either route can produce the right output; for `BOTH_BAD`, neither branch solves the target. These partitions report gate behavior and regret rather than route correctness.

A pass proves that, on this synthetic 7x7 diagnostic, a learned gate can select a frozen neural retrieval reader when paired-output retrieval is useful and a frozen neural transform expert when support-derived geometric evidence is useful. It also proves the hard-negative retrieval temptation was rejected under the tested families and held-out visual query types.

It does not prove ARC solving, multi-step composition, MLC-style abstraction, latent operator learning, RIM behavior, or test-time training. R3B-2 remains a controlled routing diagnostic, not a multi-step ARC composition benchmark.
