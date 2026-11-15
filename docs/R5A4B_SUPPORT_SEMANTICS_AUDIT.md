# R5A-4B ARC Support Semantics and Non-Copy Audit

This is an audit of the existing frozen neural VRM line. No training, model change, TTT, adapters, synthetic data, or ARC evaluation/test outputs were used.

- frozen input verification: `FROZEN_INPUT_INTEGRITY_PASS`
- query-only dummy invariance: `QUERY_ONLY_BYPASS_VALID`
- baseline replay: `R5A4A_FULL_SUPPORT_REPLAY_PASS`
- exact verdict: `SUPPORT_DEPENDENT_BUT_SEMANTICALLY_UNRESOLVED`
- mandatory next action: `C. ARCHIVE_AS_UNRESOLVED`

## Dense Scoring

Dense target-coordinate color scoring measures color quality at target positions after inference. It does not repair a wrong predicted shape and is not a substitute for exact-grid accuracy.

## Limitations

Changed-cell color metrics are evaluated at target coordinates after inference and therefore isolate cell-color quality from shape prediction. They must be read together with exact shape accuracy and exact full-grid accuracy; they are not a substitute for either.

## Interpretation

A low macro changed-cell score, even with a favorable pooled score, indicates that useful transformed-cell prediction is concentrated in a small subset of folds rather than broadly distributed across the sentinel.
