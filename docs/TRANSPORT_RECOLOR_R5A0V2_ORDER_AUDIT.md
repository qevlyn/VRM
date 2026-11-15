# R5A-0 V2 Order Audit

This audit checks the existing `transport_recolor_r5a0_v2` benchmark without changing its grammar, generator, model, losses, training code, checkpoints, split definitions, or prior verdict.

The checker in `vrm/twin_tasks/transport_recolor_r5a0v2_order_audit.py` independently implements the primitive execution semantics for:

- `C`, `H`, `V`
- `L`, `K`
- all current mixed programs and their reverses

It computes behavior fingerprints over the exhaustive pointwise domain:

```text
x,y in 0..6
color in 0..8
```

This is exhaustive for the current grammar because every operation is pointwise over cells, every geometry primitive is a coordinate bijection, and every recolor primitive depends only on current coordinate and color.

The report now makes that claim conditional on an explicit `cellwise_lift_validity` gate. The gate checks:

- every primitive is total on all `7 x 7 x 9` cell states;
- every output coordinate is in bounds;
- foreground coordinate maps are bijective, with no dropped, duplicated, colliding, or overwritten destinations;
- production primitive source does not read neighboring cells, object identity, occupancy, global color counts, or grid-level state;
- production whole-grid behavior agrees with the independent checker on deterministic multi-cell fixtures.

If this gate fails, `probe_domain_exhaustive` is false and the result is only a deterministic single-cell / branch-complete probe-domain audit.

The audit writes:

```text
/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/order_audit_report.json
/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/order_audit_samples.json
```

The pass verdict is:

```text
R5A0V2_ORDER_AUDIT_PASS
```

The failure verdict is:

```text
R5A0V2_ORDER_CLAIM_NOT_ESTABLISHED
```

This audit is scoped to semantic order distinguishability and held-out-vs-train function overlap. It does not establish ARC competence, future R5A/R5B design validity, or any property of a changed grammar.
