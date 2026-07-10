# R5A-4B Pre-Run Patch

No model weights were trained or modified. The evaluator replays the frozen R5A-4A initial and selected checkpoints on the frozen sentinel folds.

## Limitations

Changed-cell color metrics are evaluated at target coordinates after inference and therefore isolate cell-color quality from shape prediction. They must be read together with exact shape accuracy and exact full-grid accuracy; they are not a substitute for either.

## Interpretation

A low macro changed-cell score, even with a favorable pooled score, indicates that useful transformed-cell prediction is concentrated in a small subset of folds rather than broadly distributed across the sentinel.

## Artifacts

- shape_regime_metrics_path: `/mnt/d/VRM/artifacts/r5a4b_neural_vrm_prerun_patch/shape_regime_metrics.json`
- changed_cell_pooled_and_macro_metrics_path: `/mnt/d/VRM/artifacts/r5a4b_neural_vrm_prerun_patch/changed_cell_pooled_and_macro_metrics.json`
- dense_prediction_scoring_ledger_path: `/mnt/d/VRM/artifacts/r5a4b_neural_vrm_prerun_patch/dense_prediction_scoring_ledger.json`
- pairing_rotation_integrity_ledger_path: `/mnt/d/VRM/artifacts/r5a4b_neural_vrm_prerun_patch/pairing_rotation_integrity_ledger.json`
- target_changed_region_difficulty_diagnostics_path: `/mnt/d/VRM/artifacts/r5a4b_neural_vrm_prerun_patch/target_changed_region_difficulty_diagnostics.json`

## Conditions

- `FRESH_INIT`
- `TRAINED_FULL_SUPPORT_REPLAY`
- `TRAINED_ZEROED_SUPPORT_OUTPUTS`
- `TRAINED_OUTPUT_PAIRING_BROKEN`
- `TRAINED_SUPPORT_PAIR_ORDER_PERMUTED`
- `TRAINED_QUERY_ONLY`
