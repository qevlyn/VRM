# R5A-0 - Direct Factorized Transport-Recolor Baseline

R5A-0 is intended as the first heterogeneous-composition benchmark after the R4 spatial-pointer probes. R4 showed that a direct support-conditioned pointer can infer color-preserving spatial transport. That does not establish recolor competence, because a pure color-copy pointer never has to decide that a transported foreground cell should become a different color.

The proposed R5A-0 model factorizes the output distribution as:

```text
output_probs[u,c] = sum_v A[u,v] * B[u,v,c]
```

where `A` is color-blind transport and `B` is a generic color-transition head. The transport computation is intentionally prevented from reading query source color, while the transition computation sees source color plus source and target coordinates. This keeps the representation fair for geometry-then-recolor and recolor-then-geometry cases. Background preservation is a grammar-level occupancy constraint, not an explicit operation label or tool.

This experiment is lambda = 0. It does not test BIMT, connection cost, sparsity, modularity loss, iterative execution, latent operator supervision, or test-time training. Those belong only in a later controlled ablation after the baseline grammar is valid.

## Function-Audit Result

The exact operation-function audit over the required domain `(x,y,c)->(x_final,y_final,c_final)` detects collisions in the requested primitive split:

```text
F -> U == U -> F
L -> C == C -> U
```

These are mathematical consequences of the definitions:

- `F` changes `x` but leaves `y` unchanged, while `U` recolors by `y <= 3`, so their order commutes.
- `C` maps `(x,y)` to `(6-y,x)`, so applying `L` before `C` recolors exactly the cells that applying `U` after `C` would recolor.

Per the requested protocol, any function collision requires stopping before dataset generation, model construction, training, checkpointing, sealed evaluation, and head recombination.

## Claim Boundary

Because the declared grammar collides, no R5A-0 competence, held-out order-generalization, factor-use, or recombination claim can be made from this run. A later R5 variant needs a non-colliding heterogeneous split before BIMT-lite or connection-cost ablations are scientifically meaningful.
