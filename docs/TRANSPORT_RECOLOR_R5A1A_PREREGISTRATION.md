# R5A-1A Shared-Renderer Control Preregistration

R5A-1A is a control-design and preflight task only. It exists to make future outcomes interpretable. It does not favor either the explicit factorized R5A-0 V2 model or a generic shared renderer, and it does not state or assume that the shared renderer is expected to pass or fail.

This preregistration does not change the R5A-0 V2 grammar, generator, split, model, loss, optimizer, checkpoints, or historical verdict.

Machine-readable report:

```text
/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/r5a1a_preregistration_report.json
```

## Shared Renderer Envelope

Allowed:

- same raw grid tokenization and positional encoding as V2;
- generic support/query encoding;
- generic learned output-cell query tokens;
- generic cross-attention from output cells to support/query tokens;
- generic MLPs and final per-cell color logits;
- generic output-size prediction based only on allowed raw support/query information.

Generic attention is allowed. It must not be classified as forbidden merely because it could internally learn correspondence. The prohibition is against a hand-specified semantic transport/color interface.

Forbidden:

- explicit source-coordinate prediction;
- explicit pointer distribution or pointer supervision;
- copy gate;
- explicit transport representation;
- explicit color-transition representation;
- factor-labelled latent channels;
- auxiliary geometry/recolor losses;
- oracle/program/factor metadata or derived equivalent.

## Pre-Registered Outcomes

`CONTROL_UNDERFIT_OR_INVALID`

Use if any shared variant fails the train-fit adequacy gate or violates interface/sealed-data discipline. Held-out mixed failure is not evidence for factorization in this outcome.

`PRACTICAL_FACTORIZATION_ADVANTAGE_SUPPORTED`

Use only if `shared_matched` and `shared_large` both pass train-fit adequacy and both underperform V2 on held-out mixed exact by the predeclared effect floor across all three seeds. This supports only a practical advantage for the explicit V2 factorization on this fixed grammar and training regime.

`EXPLICIT_FACTORIZATION_NOT_NECESSARY_ON_CURRENT_GRAMMAR`

Use if a train-adequate shared variant matches V2 within the predeclared equivalence margin across all three seeds. This means this grammar can be solved without the explicit transport/color factorization at the tested scale.

`CAPACITY_SENSITIVE_OR_INCONCLUSIVE`

Use if results differ materially between `shared_matched` and `shared_large`, vary substantially across seeds, or miss the effect/equivalence criteria.

## Thresholds

Train-fit adequacy gate:

- train visible overall exact at least `0.995`;
- train visible seen-mixed macro exact at least `0.995`;
- every train visible operation exact at least `0.99`.

Held-out mixed practical effect floor:

- absolute exact gap versus V2 at least `0.03`.

Equivalence margin:

- absolute exact gap versus V2 at most `0.01`.

Seed consistency:

- within-variant max-minus-min held-out mixed exact at most `0.02`.

These thresholds are fixed before R5A-1A training and must not be tuned after observing R5A-1A results.

## Sealed Discipline

The sealed split may be evaluated only for the fixed pre-registered final comparison. Sealed outcomes must not be used to resize a variant, add decoder components, change optimizer or schedule, change thresholds/margins, or select between variants.

Any post-sealed modification requires a new experiment version and a new sealed evaluation surface.
