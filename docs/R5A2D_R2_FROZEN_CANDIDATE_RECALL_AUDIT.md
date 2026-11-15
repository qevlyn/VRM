# R5A-2D-R2 Frozen Candidate-Recall Audit

This audit measures deterministic candidate recall from pre-existing frozen
candidate catalogs on the already-inspected ARC-AGI-2 corpus manifest.

It does not measure:
- stochastic sampling;
- learned candidate generation;
- ARC generality;
- a deployable final-answer solver;
- learned routing;
- learned verification or ranking;
- test-time training;
- new primitives;
- expanded V2 classes;
- expanded topology hypotheses;
- performance on a fresh held-out corpus.

Because ARC-AGI-2 was previously used for the R5A-2D-R1 diagnostic,
this candidate-recall audit is exploratory on a development-burned corpus.
It may reveal whether frozen candidate diversity exists, but it is not
confirmatory evidence for future library design changes.

## Frozen Candidate Surfaces

- V2 candidate IDs are exactly the frozen `transport_recolor_r5a0v2_oracles.SEALED_CLASSES`.
- Topology candidate IDs are exactly the immutable tuples from `compiled_topology_r5a2c_r2_data.candidate_hypotheses()`.

No V2 class, topology hypothesis, primitive, adapter, router, verifier, ranker, cropper, or preprocessing rule is added.

## Metrics

The audit reports both:

- `EXACT_SUPPORT_CANDIDATE_RECALL`
- `RANKED_ALL_CATALOG_CANDIDATE_RECALL`

Fixed budgets are exactly `[1, 4, 16, 64, 128]`. Candidate budgets count deduplicated output grids, not raw candidate programs.

Run:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.run_r5a2d_r2_candidate_recall
pytest -q
```
