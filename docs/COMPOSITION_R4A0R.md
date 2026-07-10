# R4A-0R v2

R4A-0R v2 is the corrected direct support-conditioned ordered-composition baseline. It preserves the algebraic audit from the failed v1 preflight, but fixes the query-type split to match the declared catalog.

The declared query type is:

```text
(primary_shape, primary_color, anchor_region, distractor_count)
```

with:

```text
4 shapes * 4 colors * 4 anchors * 2 distractor counts = 128 query types
```

The corrected disjoint split is:

```text
train 80 + dev 24 + sealed 24 = 128 query types
```

The previous `96 + 24 + 24 = 144` request is retained only as a tested legacy-infeasible configuration and is not used.

The algebra audit uses:

- `C`: `ROT90_CW`, `C(x,y) = (6-y, x)`
- `F`: `HFLIP`, `F(x,y) = (6-x, y)`
- `T`: `SHIFT_RIGHT_1`, `T(x,y) = (x+1, y)` when `x <= 5`

Over the common safe domain `{(x,y): x,y in {1,2,3,4,5}}`, all six ordered two-step sequence signatures are unique. Held-out two-step functions are distinct from seen two-step and single-step functions.

The model is a direct support-conditioned query-copy pointer. It derives a target-position to source-position pointer from complete support-pair correspondence, aggregates the three support-pair relations by mean, and copies query colors through pointer attention. It receives no sequence labels, operation labels, query-type metadata, split IDs, pair IDs, pair positions, targets, or intermediate outputs.

Training uses only output cross entropy against the final target grid. Evaluation reports visible single-step and seen two-step competence, held-out ordered two-step generalization, pair-permutation invariance, zero-support and deranged-pairing causal use, and same-query order counterfactuals.
