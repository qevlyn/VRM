# Transform R3B0

R3B-0 is a narrow baseline before MLC-style expansion. It asks whether a learned transform expert can infer one of three single-step geometric rules from support input-output pairs, then apply that rule to a new query grid.

The active rules are HFLIP, VFLIP, and ROT180. The rule changes per episode, and each query type is exactly balanced across the three rules, so the query appearance alone cannot identify the active rule.

The dataset is separate from R3A. It uses the same small 7x7 marker grammar where practical, but every query and support input is checked to be non-degenerate: HFLIP(Q), VFLIP(Q), and ROT180(Q) must be three different grids. The target is also required to differ from every support output, so raw support-output copying cannot solve the task.

The model is a support-conditioned query-copy pointer. Each support pair is tokenized as input cells and output cells with color, row, column, and role embeddings. A shared relation encoder creates one representation per pair, and the task context is a permutation-invariant mean aggregation over the three pair representations. The output path is only a learned pointer distribution over query source cells, so output colors are copied from Q rather than painted from a free canvas.

The model forward pass does not receive transform labels, query types, split IDs, targets, episode IDs, counterfactual group IDs, candidates, or route labels. The target is used only by the output cross-entropy loss.

Zero-support and deranged-pairing ablations are required. Zero-support removes relation evidence after pair encoding, before task-context aggregation. Deranged pairing preserves the support input bag and output bag while breaking X_i -> Y_i correspondence. These tests distinguish real support-conditioned rule inference from query-only shortcuts or output-bag shortcuts.

The same-query support-rule counterfactual suite is the strongest test: the query and support inputs stay fixed, while paired support outputs switch among HFLIP, VFLIP, and ROT180 contexts. Passing requires the output to switch correctly solely because the demonstrated support relation changed.

Passing R3B-0 does not prove ARC reasoning, multi-step composition, a toolbox of transforms, or integration with the reader-router system. If R3B-0 fails on composition-style extensions, R3B-1 should add MLC-oriented structure. If R3B-0 passes, R3B-2 is the natural integration step: connect the learned transform expert to the retrieval/router path established by R3A.
