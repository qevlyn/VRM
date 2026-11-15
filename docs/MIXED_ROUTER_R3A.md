# MIXED ROUTER R3A

R3A is an oracle mixed-route router benchmark. It follows R2A, which showed
that a frozen Pair Reader can be connected beside a legacy decoder route, but
did not show selective routing. R2A could pass by always copying reader output.

R3A adds three families:

- `RETRIEVAL_ELIGIBLE`: exactly one support input is identical to the query, so
  retrieving the paired support output is correct.
- `TRANSFORM_ONLY`: no support input matches the query, and the correct output
  is a deterministic transform of the query.
- `HARD_NEGATIVE_NO_MATCH`: support inputs are near matches, but none is an
  exact match, so the router must reject retrieval.

The transform expert is an oracle in R3A so the experiment isolates routing. The
reader and transform route are independently certified before interpreting the
gate. The gate trains only from final output loss; it receives reader-side match
evidence such as retrieval logits, alpha entropy, score margin, and q-key
distance features. It never receives target grids, route labels, transform
metadata, support output grids, candidates, query type metadata, or group IDs.

Alpha entropy alone is not enough for no-match detection because near-match
support sets can still produce a sharp top support. R3A therefore includes raw
score, margin, and distance features. No balancing, noisy gate, entropy,
sparsity, or expert-utilization objective is used initially, so success means
the output loss alone can induce the route decision.

Held-out `query_type` combinations test factor generalization rather than random
episode memorization. Same-query counterfactual groups are the strongest router
test: the query grid is identical, while support evidence changes the correct
route.

If R3A passes, it shows this benchmark can learn an evidence-sensitive gate
between a competent retrieval expert and an oracle transform expert. It does not
show ARC is solved, does not prove test-time adaptation, and does not prove a
learned transform route. R3B should replace the transform oracle with a learned
transform expert.

