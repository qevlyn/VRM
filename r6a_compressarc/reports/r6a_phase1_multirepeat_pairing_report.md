# R6A Phase1 Multi-Repeat Pairing Audit

- source commit: 83a22218024d46273eb32b769a906340202ffb4d
- environment label: ENVIRONMENT_DIFFERS_BUT_COMPATIBLE
- active dependency versions: {"matplotlib": "3.11.0", "numpy": "2.5.0", "torch": "2.12.1+cu130", "tqdm": "4.68.3"}
- Python executable: /home/quyen/.venvs/vrm/bin/python
- CUDA available: True
- firewall preflight passed: True
- protected VRM/reference integrity: PASS (701 files)
- all frozen runs completed: True (81/81)
- runtime seconds: 177399.775911634
- runtime hours: 49.278
- runtime note: 16.7 hours is only a lower-bound estimate based on the 12:22 smoke runtime; actual runtime may be approximately 25-27 hours because evaluation tasks may be larger or slower.
- final verdict: PAIRING_SENSITIVE_QUERY_PILOT

## Firewall Proof

```json
{
  "cpu_dummy_target_negative_control": {
    "byte_identical_probability_tensors": true,
    "byte_identical_solver_traces": true,
    "compared_fields": [
      "trajectory",
      "predictions",
      "actual_step_count",
      "structured_log_hash_excluding_wall_clock",
      "final_prediction_hash"
    ],
    "evaluator_metrics": {
      "dummy_a": {
        "all_cell_accuracy": 0.0,
        "changed_cell_macro_accuracy": 0.0,
        "changed_cell_pooled_accuracy": 0.0,
        "exact_grid_match": false,
        "output_shape_exact_match": true,
        "probability_checkpoint_count": 20,
        "probability_note": "For cells lying inside the evaluator-known target shape but outside a model-active or semantically meaningful padded region, the stored padded probability may not carry an interpretable spatial meaning. This does not alter the paired comparison because F1, S, and F2 use the identical canonical probability representation and evaluator rule.",
        "rows": [
          {
            "all_cell_accuracy": 0.0,
            "changed_cell_accuracy": 0.0,
            "changed_cell_count": 256,
            "changed_cell_entries": [
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 1,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 0,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 8,
                "query_color": 7,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 1,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 2,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 8,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 1,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 3,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 7,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 4,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 4,
                "query_color": 4,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 7,
                "query_color": 4,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 5,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 8,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 6,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 7,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 8,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 4,
                "query_color": 4,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 4,
                "query_color": 4,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 10,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 8,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 11,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 8,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 12,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 13,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 14,
                "target_color": 0
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 7,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 15,
                "target_color": 0
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 1,
                "query_color": 1,
                "row": 15,
                "target_color": 0
              }
            ],
            "changed_cell_loss_or_nll": null,
            "exact_grid_match": false,
            "output_shape_exact_match": true,
            "predicted_grid": [
              [
                9,
                8,
                2,
                8,
                9,
                7,
                9,
                7,
                7,
                9,
                7,
                9,
                8,
                2,
                8,
                1
              ],
              [
                8,
                8,
                1,
                1,
                7,
                9,
                1,
                1,
                1,
                1,
                9,
                8,
                1,
                1,
                8,
                8
              ],
              [
                2,
                1,
                1,
                2,
                9,
                1,
                9,
                9,
                9,
                9,
                1,
                9,
                2,
                1,
                1,
                2
              ],
              [
                9,
                3,
                2,
                8,
                7,
                1,
                9,
                9,
                9,
                9,
                1,
                7,
                8,
                2,
                1,
                8
              ],
              [
                9,
                7,
                9,
                7,
                8,
                6,
                8,
                6,
                6,
                8,
                6,
                8,
                8,
                9,
                7,
                9
              ],
              [
                7,
                3,
                3,
                3,
                6,
                4,
                6,
                8,
                8,
                6,
                7,
                6,
                1,
                1,
                9,
                7
              ],
              [
                9,
                3,
                3,
                3,
                8,
                6,
                6,
                6,
                6,
                6,
                6,
                9,
                9,
                9,
                1,
                9
              ],
              [
                7,
                3,
                3,
                3,
                6,
                8,
                6,
                8,
                8,
                6,
                8,
                6,
                9,
                9,
                1,
                7
              ],
              [
                7,
                3,
                3,
                3,
                6,
                8,
                6,
                8,
                8,
                6,
                8,
                6,
                9,
                9,
                1,
                7
              ],
              [
                9,
                1,
                9,
                9,
                8,
                6,
                6,
                6,
                6,
                6,
                6,
                8,
                9,
                9,
                1,
                9
              ],
              [
                7,
                3,
                3,
                3,
                3,
                4,
                6,
                8,
                8,
                6,
                4,
                6,
                1,
                1,
                9,
                7
              ],
              [
                9,
                3,
                3,
                3,
                3,
                6,
                8,
                6,
                6,
                8,
                6,
                6,
                7,
                9,
                7,
                9
              ],
              [
                9,
                1,
                2,
                8,
                7,
                1,
                9,
                9,
                9,
                9,
                1,
                7,
                8,
                2,
                1,
                8
              ],
              [
                2,
                1,
                1,
                2,
                9,
                1,
                9,
                9,
                9,
                9,
                1,
                9,
                2,
                1,
                1,
                2
              ],
              [
                8,
                8,
                1,
                1,
                7,
                9,
                1,
                1,
                1,
                1,
                9,
                7,
                1,
                1,
                8,
                8
              ],
              [
                1,
                8,
                2,
                8,
                9,
                7,
                9,
                7,
                6,
                9,
                7,
                9,
                8,
                2,
                8,
                1
              ]
            ],
            "target_grid": [
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ],
              [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
              ]
            ],
            "test_index": 0
          }
        ],
        "target_cell_nll_from_final100_mean_probability": 16.477752248455214,
        "total_changed_cells": 256
      },
      "dummy_b": {
        "all_cell_accuracy": 0.18359375,
        "changed_cell_macro_accuracy": 0.0,
        "changed_cell_pooled_accuracy": 0.0,
        "exact_grid_match": false,
        "output_shape_exact_match": true,
        "probability_checkpoint_count": 20,
        "probability_note": "For cells lying inside the evaluator-known target shape but outside a model-active or semantically meaningful padded region, the stored padded probability may not carry an interpretable spatial meaning. This does not alter the paired comparison because F1, S, and F2 use the identical canonical probability representation and evaluator rule.",
        "rows": [
          {
            "all_cell_accuracy": 0.18359375,
            "changed_cell_accuracy": 0.0,
            "changed_cell_count": 207,
            "changed_cell_entries": [
              {
                "col": 1,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 0,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 1,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 1,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 1,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 1,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 1,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 8,
                "query_color": 7,
                "row": 1,
                "target_color": 1
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 1,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 1,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 2,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 8,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 3,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 7,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 4,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 4,
                "query_color": 4,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 7,
                "query_color": 4,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 5,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 8,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 6,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 7,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 8,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 9,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 4,
                "query_color": 4,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 4,
                "query_color": 4,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 10,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 3,
                "query_color": 3,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 6,
                "query_color": 6,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 6,
                "query_color": 8,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 11,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 9,
                "query_color": 8,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 12,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 13,
                "target_color": 1
              },
              {
                "col": 0,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 14,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 14,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 14,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 14,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 14,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 14,
                "target_color": 1
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 14,
                "target_color": 1
              },
              {
                "col": 15,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 14,
                "target_color": 1
              },
              {
                "col": 1,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 2,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 3,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 4,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 5,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 6,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 7,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 8,
                "correct": false,
                "predicted_color": 6,
                "query_color": 7,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 9,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 10,
                "correct": false,
                "predicted_color": 7,
                "query_color": 7,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 11,
                "correct": false,
                "predicted_color": 9,
                "query_color": 9,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 12,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 13,
                "correct": false,
                "predicted_color": 2,
                "query_color": 2,
                "row": 15,
                "target_color": 1
              },
              {
                "col": 14,
                "correct": false,
                "predicted_color": 8,
                "query_color": 8,
                "row": 15,
                "target_color": 1
              }
            ],
            "changed_cell_loss_or_nll": null,
            "exact_grid_match": false,
            "output_shape_exact_match": true,
            "predicted_grid": [
              [
                9,
                8,
                2,
                8,
                9,
                7,
                9,
                7,
                7,
                9,
                7,
                9,
                8,
                2,
                8,
                1
              ],
              [
                8,
                8,
                1,
                1,
                7,
                9,
                1,
                1,
                1,
                1,
                9,
                8,
                1,
                1,
                8,
                8
              ],
              [
                2,
                1,
                1,
                2,
                9,
                1,
                9,
                9,
                9,
                9,
                1,
                9,
                2,
                1,
                1,
                2
              ],
              [
                9,
                3,
                2,
                8,
                7,
                1,
                9,
                9,
                9,
                9,
                1,
                7,
                8,
                2,
                1,
                8
              ],
              [
                9,
                7,
                9,
                7,
                8,
                6,
                8,
                6,
                6,
                8,
                6,
                8,
                8,
                9,
                7,
                9
              ],
              [
                7,
                3,
                3,
                3,
                6,
                4,
                6,
                8,
                8,
                6,
                7,
                6,
                1,
                1,
                9,
                7
              ],
              [
                9,
                3,
                3,
                3,
                8,
                6,
                6,
                6,
                6,
                6,
                6,
                9,
                9,
                9,
                1,
                9
              ],
              [
                7,
                3,
                3,
                3,
                6,
                8,
                6,
                8,
                8,
                6,
                8,
                6,
                9,
                9,
                1,
                7
              ],
              [
                7,
                3,
                3,
                3,
                6,
                8,
                6,
                8,
                8,
                6,
                8,
                6,
                9,
                9,
                1,
                7
              ],
              [
                9,
                1,
                9,
                9,
                8,
                6,
                6,
                6,
                6,
                6,
                6,
                8,
                9,
                9,
                1,
                9
              ],
              [
                7,
                3,
                3,
                3,
                3,
                4,
                6,
                8,
                8,
                6,
                4,
                6,
                1,
                1,
                9,
                7
              ],
              [
                9,
                3,
                3,
                3,
                3,
                6,
                8,
                6,
                6,
                8,
                6,
                6,
                7,
                9,
                7,
                9
              ],
              [
                9,
                1,
                2,
                8,
                7,
                1,
                9,
                9,
                9,
                9,
                1,
                7,
                8,
                2,
                1,
                8
              ],
              [
                2,
                1,
                1,
                2,
                9,
                1,
                9,
                9,
                9,
                9,
                1,
                9,
                2,
                1,
                1,
                2
              ],
              [
                8,
                8,
                1,
                1,
                7,
                9,
                1,
                1,
                1,
                1,
                9,
                7,
                1,
                1,
                8,
                8
              ],
              [
                1,
                8,
                2,
                8,
                9,
                7,
                9,
                7,
                6,
                9,
                7,
                9,
                8,
                2,
                8,
                1
              ]
            ],
            "target_grid": [
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ],
              [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
              ]
            ],
            "test_index": 0
          }
        ],
        "target_cell_nll_from_final100_mean_probability": 2.23914804189382,
        "total_changed_cells": 207
      }
    },
    "evaluator_metrics_differ": true,
    "fixed_prefix_steps": 20,
    "runs": {
      "dummy_a": {
        "cmd": [
          "/home/quyen/.venvs/vrm/bin/python",
          "/mnt/d/VRM/r6a_compressarc/scripts/phase1_condition_solver.py",
          "--compressarc-root",
          "/mnt/d/VRM/third_party/compressarc_reference",
          "--challenge-json",
          "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
          "--initialization-artifact",
          "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt",
          "--task-id",
          "903d1b4a",
          "--stratum",
          "FIREWALL_CPU_NEGATIVE",
          "--condition",
          "FIREWALL_CPU_PREFIX",
          "--steps",
          "20",
          "--output-json",
          "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_a_solver.json",
          "--probability-npz",
          "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_a_probability.npz",
          "--device",
          "cpu"
        ],
        "returncode": 0,
        "runtime_seconds": 0.0,
        "stdout": "reused"
      },
      "dummy_b": {
        "cmd": [
          "/home/quyen/.venvs/vrm/bin/python",
          "/mnt/d/VRM/r6a_compressarc/scripts/phase1_condition_solver.py",
          "--compressarc-root",
          "/mnt/d/VRM/third_party/compressarc_reference",
          "--challenge-json",
          "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
          "--initialization-artifact",
          "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt",
          "--task-id",
          "903d1b4a",
          "--stratum",
          "FIREWALL_CPU_NEGATIVE",
          "--condition",
          "FIREWALL_CPU_PREFIX",
          "--steps",
          "20",
          "--output-json",
          "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_b_solver.json",
          "--probability-npz",
          "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_b_probability.npz",
          "--device",
          "cpu"
        ],
        "returncode": 0,
        "runtime_seconds": 0.0,
        "stdout": "reused"
      }
    },
    "solver_did_not_receive_dummy_targets": true
  },
  "firewall_preflight_passed": true,
  "static_data_flow_isolation": {
    "data_path_environment_variables": {
      "XDG_DATA_DIRS": "/usr/local/share:/usr/share:/var/lib/snapd/desktop"
    },
    "evaluator_only_solution_paths": [
      "/mnt/d/VRM/third_party/compressarc_reference/dataset/arc-agi_evaluation_solutions.json"
    ],
    "mounted_or_copied_files": [
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt"
    ],
    "solution_json_path_absent_from_data_path_env_vars": true,
    "solution_json_path_absent_from_solver_commands": true,
    "solution_json_path_absent_from_solver_inputs": true,
    "solver_commands": [
      [
        "/home/quyen/.venvs/vrm/bin/python",
        "/mnt/d/VRM/r6a_compressarc/scripts/phase1_condition_solver.py",
        "--compressarc-root",
        "/mnt/d/VRM/third_party/compressarc_reference",
        "--challenge-json",
        "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
        "--initialization-artifact",
        "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt",
        "--task-id",
        "903d1b4a",
        "--stratum",
        "FIREWALL_CPU_NEGATIVE",
        "--condition",
        "FIREWALL_CPU_PREFIX",
        "--steps",
        "20",
        "--output-json",
        "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_a_solver.json",
        "--probability-npz",
        "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_a_probability.npz",
        "--device",
        "cpu"
      ],
      [
        "/home/quyen/.venvs/vrm/bin/python",
        "/mnt/d/VRM/r6a_compressarc/scripts/phase1_condition_solver.py",
        "--compressarc-root",
        "/mnt/d/VRM/third_party/compressarc_reference",
        "--challenge-json",
        "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
        "--initialization-artifact",
        "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt",
        "--task-id",
        "903d1b4a",
        "--stratum",
        "FIREWALL_CPU_NEGATIVE",
        "--condition",
        "FIREWALL_CPU_PREFIX",
        "--steps",
        "20",
        "--output-json",
        "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_b_solver.json",
        "--probability-npz",
        "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/dummy_b_probability.npz",
        "--device",
        "cpu"
      ]
    ],
    "solver_input_paths": [
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/challenge.json",
      "/mnt/d/VRM/r6a_compressarc/artifacts/phase1_firewall_cpu_negative/initialization.pt"
    ],
    "solver_working_directory": "/mnt/d/VRM"
  }
}
```

## Frozen Manifest

- manifest path: /mnt/d/VRM/r6a_compressarc/manifests/r6a_phase1_multirepeat_frozen_manifest.json
- seeds: [20260702, 20260703, 20260704]
- optimization budget: 1500
- run order count: 81
- permutation authority all Stratum A match: True

| task | stratum | eligible | old permutation | recomputed permutation | score | authority |
|---|---:|---:|---|---|---:|---:|
| 604001fa | A | True | [1, 0, 3, 2] | [1, 0, 3, 2] | 650 | True |
| 79fb03f4 | A | True | [5, 3, 4, 1, 2, 0] | [5, 3, 4, 1, 2, 0] | 1236 | True |
| 981571dc | A | True | [2, 3, 0, 1] | [2, 3, 0, 1] | 3360 | True |
| 3ed85e70 | A | True | [1, 2, 0] | [1, 2, 0] | 1283 | True |
| aa4ec2a5 | A | True | [1, 2, 0] | [1, 2, 0] | 1419 | True |
| 929ab4e9 | A | True | [3, 2, 1, 0] | [3, 2, 1, 0] | 2088 | True |
| 0dfd9992 | B | True | None | [1, 2, 0] | 998 | None |
| 1c786137 | B | True | None | [1, 2, 0] | 192 | None |
| 1f876c06 | B | True | None | [1, 2, 0] | 88 | None |

## Eligibility And Thresholds

```json
{
  "stratum_a": {
    "cells": 18,
    "delta_gt_nu_threshold": 7,
    "eligible_tasks": 6,
    "positive_delta_threshold": 13,
    "positive_task_mean_threshold": 5
  },
  "stratum_b": {
    "cells": 9,
    "delta_gt_nu_threshold": 4,
    "eligible_tasks": 3,
    "positive_control_valid": true,
    "positive_delta_threshold": 7,
    "positive_task_mean_threshold": 3
  }
}
```

## Initialization Artifacts

- artifact count: 27
- all artifacts exist: True

| task | stratum | seed | model hash | optimizer hash | rng hash |
|---|---|---:|---|---|---|
| 604001fa | A | 20260702 | 6abf283f29d376d9... | 303dfc88384552ee... | 65e54a566008dd1c... |
| 604001fa | A | 20260703 | 36ac8a1afa4ddee7... | 303dfc88384552ee... | 324360b1958aa83a... |
| 604001fa | A | 20260704 | 15411152edab70fa... | 303dfc88384552ee... | 516358d41e7d9dd2... |
| 79fb03f4 | A | 20260702 | c6f3f272651889db... | 303dfc88384552ee... | 9a25c4f083f6e21b... |
| 79fb03f4 | A | 20260703 | ef3697eb3f79de99... | 303dfc88384552ee... | 1981d97b6f92a41d... |
| 79fb03f4 | A | 20260704 | 0de051e6adf350c2... | 303dfc88384552ee... | 07ab14b0ce47affc... |
| 981571dc | A | 20260702 | 198dfcc64ea09284... | 303dfc88384552ee... | 78e15ecefaa2fd8d... |
| 981571dc | A | 20260703 | 48a9a60d322412ff... | 303dfc88384552ee... | de39aa50c153b87e... |
| 981571dc | A | 20260704 | 094a3e8603691eca... | 303dfc88384552ee... | fa0d2d40f68e238c... |
| 3ed85e70 | A | 20260702 | 220323dec0e6507f... | 303dfc88384552ee... | 9a25c4f083f6e21b... |
| 3ed85e70 | A | 20260703 | 6a7333523599f127... | 303dfc88384552ee... | 1981d97b6f92a41d... |
| 3ed85e70 | A | 20260704 | 5970288821a836ba... | 303dfc88384552ee... | 07ab14b0ce47affc... |
| aa4ec2a5 | A | 20260702 | eaf6ae0f2b854772... | 303dfc88384552ee... | 9a25c4f083f6e21b... |
| aa4ec2a5 | A | 20260703 | 7b52b931bb89bbc3... | 303dfc88384552ee... | 1981d97b6f92a41d... |
| aa4ec2a5 | A | 20260704 | 465185985909b7af... | 303dfc88384552ee... | 07ab14b0ce47affc... |
| 929ab4e9 | A | 20260702 | 87513527eac0a64c... | 303dfc88384552ee... | 6095ddb339bd166c... |
| 929ab4e9 | A | 20260703 | 1f640474299ca2d4... | 303dfc88384552ee... | c242c20af5b4182b... |
| 929ab4e9 | A | 20260704 | 7e8d6fa90652624e... | 303dfc88384552ee... | dd4502c31df3d7d3... |
| 0dfd9992 | B | 20260702 | 2ffd7eebbaf431ec... | 303dfc88384552ee... | 9a25c4f083f6e21b... |
| 0dfd9992 | B | 20260703 | 64c56939d3bbfd72... | 303dfc88384552ee... | 1981d97b6f92a41d... |
| 0dfd9992 | B | 20260704 | 7bb5bebca390f3dd... | 303dfc88384552ee... | 07ab14b0ce47affc... |
| 1c786137 | B | 20260702 | 372c970f45e95ec1... | 303dfc88384552ee... | 9a25c4f083f6e21b... |
| 1c786137 | B | 20260703 | 9debb299e1c809fd... | 303dfc88384552ee... | 1981d97b6f92a41d... |
| 1c786137 | B | 20260704 | 8ca977eb5f29e90e... | 303dfc88384552ee... | 07ab14b0ce47affc... |
| 1f876c06 | B | 20260702 | 14ae222f524b51a7... | 303dfc88384552ee... | 65e54a566008dd1c... |
| 1f876c06 | B | 20260703 | c92171f70ffbf455... | 303dfc88384552ee... | 324360b1958aa83a... |
| 1f876c06 | B | 20260704 | b6fd2d4d48046e51... | 303dfc88384552ee... | 516358d41e7d9dd2... |

## Stratum Summaries

```json
{
  "stratum_a": {
    "cell_count": 18,
    "delta_gt_nu_cells": 15,
    "leave_one_task_out": [
      {
        "mean_delta": 1.7822279923407105,
        "positive": true,
        "removed_task_id": "3ed85e70"
      },
      {
        "mean_delta": 1.8189260945012664,
        "positive": true,
        "removed_task_id": "604001fa"
      },
      {
        "mean_delta": 1.869012501909276,
        "positive": true,
        "removed_task_id": "79fb03f4"
      },
      {
        "mean_delta": 1.2344385366761257,
        "positive": true,
        "removed_task_id": "929ab4e9"
      },
      {
        "mean_delta": 1.3096973172063693,
        "positive": true,
        "removed_task_id": "981571dc"
      },
      {
        "mean_delta": 1.918966068848762,
        "positive": true,
        "removed_task_id": "aa4ec2a5"
      }
    ],
    "mean_delta": 1.6555447519137516,
    "passes": true,
    "positive_delta_cells": 17,
    "positive_task_mean_count": 6,
    "stratum": "A",
    "task_count": 6,
    "thresholds": {
      "cells": 18,
      "delta_gt_nu_threshold": 7,
      "eligible_tasks": 6,
      "positive_delta_threshold": 13,
      "positive_task_mean_threshold": 5
    }
  },
  "stratum_b": {
    "cell_count": 9,
    "delta_gt_nu_cells": 9,
    "leave_one_task_out": [
      {
        "mean_delta": 1.28663462565961,
        "positive": true,
        "removed_task_id": "0dfd9992"
      },
      {
        "mean_delta": 1.8256023699925779,
        "positive": true,
        "removed_task_id": "1c786137"
      },
      {
        "mean_delta": 1.4131180034205755,
        "positive": true,
        "removed_task_id": "1f876c06"
      }
    ],
    "mean_delta": 1.5084516663575878,
    "passes": true,
    "positive_delta_cells": 9,
    "positive_task_mean_count": 3,
    "stratum": "B",
    "task_count": 3,
    "thresholds": {
      "cells": 9,
      "delta_gt_nu_threshold": 4,
      "eligible_tasks": 3,
      "positive_control_valid": true,
      "positive_delta_threshold": 7,
      "positive_task_mean_threshold": 3
    }
  }
}
```

## Per-Task Results

```json
{
  "tasks": [
    {
      "cells": [
        {
          "delta": 2.9584508179761477,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.0046304657524277775,
          "ell_F2": 0.0039320165479423946,
          "ell_S": 2.9627320591263326,
          "nu": 0.0006984492044853829,
          "seed": 20260702,
          "stratum": "B",
          "task_id": "0dfd9992"
        },
        {
          "delta": 2.1080911905972775,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.003422871397891471,
          "ell_F2": 0.0049428485695890275,
          "ell_S": 2.112274050581018,
          "nu": 0.0015199771716975564,
          "seed": 20260703,
          "stratum": "B",
          "task_id": "0dfd9992"
        },
        {
          "delta": 0.7897152346872045,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.0018093111790617432,
          "ell_F2": 0.003985025614222388,
          "ell_S": 0.7926124030838466,
          "nu": 0.002175714435160645,
          "seed": 20260704,
          "stratum": "B",
          "task_id": "0dfd9992"
        }
      ],
      "delta_gt_nu_seed_count": 3,
      "mean_delta": 1.9520857477535432,
      "positive_seed_count": 3,
      "seed_count": 3,
      "stratum": "B",
      "task_id": "0dfd9992"
    },
    {
      "cells": [
        {
          "delta": 1.2102156473313812,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 2.1453373937314297,
          "ell_F2": 2.239542792616224,
          "ell_S": 3.402655740505208,
          "nu": 0.09420539888479418,
          "seed": 20260702,
          "stratum": "B",
          "task_id": "1c786137"
        },
        {
          "delta": 0.26492830572527604,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 2.388475244800836,
          "ell_F2": 2.443788859704387,
          "ell_S": 2.6810603579778873,
          "nu": 0.05531361490355069,
          "seed": 20260703,
          "stratum": "B",
          "task_id": "1c786137"
        },
        {
          "delta": 1.147306824206165,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 3.8707019652861274,
          "ell_F2": 3.337753298052421,
          "ell_S": 4.751534455875439,
          "nu": 0.5329486672337063,
          "seed": 20260704,
          "stratum": "B",
          "task_id": "1c786137"
        }
      ],
      "delta_gt_nu_seed_count": 3,
      "mean_delta": 0.8741502590876075,
      "positive_seed_count": 3,
      "seed_count": 3,
      "stratum": "B",
      "task_id": "1c786137"
    },
    {
      "cells": [
        {
          "delta": 1.4908595081490978,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.0256148142187857,
          "ell_F2": 0.025577118312320336,
          "ell_S": 1.5164554744146508,
          "nu": 3.769590646536536e-05,
          "seed": 20260702,
          "stratum": "B",
          "task_id": "1f876c06"
        },
        {
          "delta": 2.1955573943626745,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.031012341895522516,
          "ell_F2": 0.05035020969652914,
          "ell_S": 2.2362386701587003,
          "nu": 0.019337867801006624,
          "seed": 20260703,
          "stratum": "B",
          "task_id": "1f876c06"
        },
        {
          "delta": 1.4109400741830644,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.028679764546209966,
          "ell_F2": 0.03985146197970247,
          "ell_S": 1.4452056874460206,
          "nu": 0.011171697433492504,
          "seed": 20260704,
          "stratum": "B",
          "task_id": "1f876c06"
        }
      ],
      "delta_gt_nu_seed_count": 3,
      "mean_delta": 1.6991189922316121,
      "positive_seed_count": 3,
      "seed_count": 3,
      "stratum": "B",
      "task_id": "1f876c06"
    },
    {
      "cells": [
        {
          "delta": 0.7600047855492504,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.1712929061904791,
          "ell_F2": 0.12325285834176267,
          "ell_S": 0.9072776678153713,
          "nu": 0.04804004784871643,
          "seed": 20260702,
          "stratum": "A",
          "task_id": "3ed85e70"
        },
        {
          "delta": 1.241838560940043,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.15117536840371001,
          "ell_F2": 0.13521055783169014,
          "ell_S": 1.385031524057743,
          "nu": 0.015964810572019872,
          "seed": 20260703,
          "stratum": "A",
          "task_id": "3ed85e70"
        },
        {
          "delta": 1.064542302847581,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.16580473796160933,
          "ell_F2": 0.15218918127554426,
          "ell_S": 1.2235392624661576,
          "nu": 0.013615556686065067,
          "seed": 20260704,
          "stratum": "A",
          "task_id": "3ed85e70"
        }
      ],
      "delta_gt_nu_seed_count": 3,
      "mean_delta": 1.022128549778958,
      "positive_seed_count": 3,
      "seed_count": 3,
      "stratum": "A",
      "task_id": "3ed85e70"
    },
    {
      "cells": [
        {
          "delta": 1.2113627754126295,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.31766102670336493,
          "ell_F2": 0.2781899218308795,
          "ell_S": 1.5092882496797517,
          "nu": 0.03947110487248545,
          "seed": 20260702,
          "stratum": "A",
          "task_id": "604001fa"
        },
        {
          "delta": 0.7901441312012407,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.2625875465741528,
          "ell_F2": 0.2411517277252947,
          "ell_S": 1.0420137683509645,
          "nu": 0.02143581884885809,
          "seed": 20260703,
          "stratum": "A",
          "task_id": "604001fa"
        },
        {
          "delta": 0.5144072103146609,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.28615328083990255,
          "ell_F2": 0.2421261787499718,
          "ell_S": 0.778546940109598,
          "nu": 0.04402710208993074,
          "seed": 20260704,
          "stratum": "A",
          "task_id": "604001fa"
        }
      ],
      "delta_gt_nu_seed_count": 3,
      "mean_delta": 0.8386380389761771,
      "positive_seed_count": 3,
      "seed_count": 3,
      "stratum": "A",
      "task_id": "604001fa"
    },
    {
      "cells": [
        {
          "delta": 0.4512784952038572,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.06818807616516055,
          "ell_F2": 0.12328487070721866,
          "ell_S": 0.5470149686400468,
          "nu": 0.05509679454205811,
          "seed": 20260702,
          "stratum": "A",
          "task_id": "79fb03f4"
        },
        {
          "delta": 0.35684643711761876,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.08773274740062813,
          "ell_F2": 0.08738425881830886,
          "ell_S": 0.44440494022708726,
          "nu": 0.00034848858231927016,
          "seed": 20260703,
          "stratum": "A",
          "task_id": "79fb03f4"
        },
        {
          "delta": 0.956493073486912,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.09692859341430113,
          "ell_F2": 0.08059004753701438,
          "ell_S": 1.0452523939625697,
          "nu": 0.016338545877286753,
          "seed": 20260704,
          "stratum": "A",
          "task_id": "79fb03f4"
        }
      ],
      "delta_gt_nu_seed_count": 3,
      "mean_delta": 0.5882060019361294,
      "positive_seed_count": 3,
      "seed_count": 3,
      "stratum": "A",
      "task_id": "79fb03f4"
    },
    {
      "cells": [
        {
          "delta": 0.3224827561795297,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.05299628329230822,
          "ell_F2": 0.04639276010865653,
          "ell_S": 0.3721772778800121,
          "nu": 0.006603523183651684,
          "seed": 20260702,
          "stratum": "A",
          "task_id": "929ab4e9"
        },
        {
          "delta": 3.0958112936259483,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.03808860083306448,
          "ell_F2": 0.04982503128365154,
          "ell_S": 3.139768109684306,
          "nu": 0.01173643045058706,
          "seed": 20260703,
          "stratum": "A",
          "task_id": "929ab4e9"
        },
        {
          "delta": 7.864933434500168,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.05441497288688641,
          "ell_F2": 0.05828918185162705,
          "ell_S": 7.921285511869424,
          "nu": 0.0038742089647406386,
          "seed": 20260704,
          "stratum": "A",
          "task_id": "929ab4e9"
        }
      ],
      "delta_gt_nu_seed_count": 3,
      "mean_delta": 3.761075828101882,
      "positive_seed_count": 3,
      "seed_count": 3,
      "stratum": "A",
      "task_id": "929ab4e9"
    },
    {
      "cells": [
        {
          "delta": 3.6460973318870433,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.02358652407416576,
          "ell_F2": 0.015397404990663187,
          "ell_S": 3.6655892964194576,
          "nu": 0.008189119083502575,
          "seed": 20260702,
          "stratum": "A",
          "task_id": "981571dc"
        },
        {
          "delta": 1.744672189677929,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.026562889185259207,
          "ell_F2": 0.021421151179619607,
          "ell_S": 1.7686642098603684,
          "nu": 0.0051417380056396,
          "seed": 20260703,
          "stratum": "A",
          "task_id": "981571dc"
        },
        {
          "delta": 4.763576254787019,
          "delta_gt_0": true,
          "delta_gt_nu": true,
          "ell_F1": 0.026127905547972886,
          "ell_F2": 0.025804698376201644,
          "ell_S": 4.7895425567491055,
          "nu": 0.00032320717177124203,
          "seed": 20260704,
          "stratum": "A",
          "task_id": "981571dc"
        }
      ],
      "delta_gt_nu_seed_count": 3,
      "mean_delta": 3.384781925450664,
      "positive_seed_count": 3,
      "seed_count": 3,
      "stratum": "A",
      "task_id": "981571dc"
    },
    {
      "cells": [
        {
          "delta": -0.18370111170273606,
          "delta_gt_0": false,
          "delta_gt_nu": false,
          "ell_F1": 2.529387581057764,
          "ell_F2": 1.5466669623598477,
          "ell_S": 1.8543261600060696,
          "nu": 0.9827206186979163,
          "seed": 20260702,
          "stratum": "A",
          "task_id": "aa4ec2a5"
        },
        {
          "delta": 0.705840661018013,
          "delta_gt_0": true,
          "delta_gt_nu": false,
          "ell_F1": 1.595408809448589,
          "ell_F2": 0.26368871345820816,
          "ell_S": 1.6353894224714116,
          "nu": 1.3317200959903808,
          "seed": 20260703,
          "stratum": "A",
          "task_id": "aa4ec2a5"
        },
        {
          "delta": 0.49317495240082254,
          "delta_gt_0": true,
          "delta_gt_nu": false,
          "ell_F1": 0.9931439676335576,
          "ell_F2": 0.30741976275064814,
          "ell_S": 1.1434568175929254,
          "nu": 0.6857242048829095,
          "seed": 20260704,
          "stratum": "A",
          "task_id": "aa4ec2a5"
        }
      ],
      "delta_gt_nu_seed_count": 0,
      "mean_delta": 0.3384381672386998,
      "positive_seed_count": 2,
      "seed_count": 3,
      "stratum": "A",
      "task_id": "aa4ec2a5"
    }
  ]
}
```

## All F1/S/F2 NLL Cells

| stratum | task | seed | ell_F1 | ell_S | ell_F2 | nu | delta | delta>0 | delta>nu |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 604001fa | 20260702 | 0.317661 | 1.509288 | 0.278190 | 0.039471 | 1.211363 | True | True |
| A | 604001fa | 20260703 | 0.262588 | 1.042014 | 0.241152 | 0.021436 | 0.790144 | True | True |
| A | 604001fa | 20260704 | 0.286153 | 0.778547 | 0.242126 | 0.044027 | 0.514407 | True | True |
| A | 79fb03f4 | 20260702 | 0.068188 | 0.547015 | 0.123285 | 0.055097 | 0.451278 | True | True |
| A | 79fb03f4 | 20260703 | 0.087733 | 0.444405 | 0.087384 | 0.000348 | 0.356846 | True | True |
| A | 79fb03f4 | 20260704 | 0.096929 | 1.045252 | 0.080590 | 0.016339 | 0.956493 | True | True |
| A | 981571dc | 20260702 | 0.023587 | 3.665589 | 0.015397 | 0.008189 | 3.646097 | True | True |
| A | 981571dc | 20260703 | 0.026563 | 1.768664 | 0.021421 | 0.005142 | 1.744672 | True | True |
| A | 981571dc | 20260704 | 0.026128 | 4.789543 | 0.025805 | 0.000323 | 4.763576 | True | True |
| A | 3ed85e70 | 20260702 | 0.171293 | 0.907278 | 0.123253 | 0.048040 | 0.760005 | True | True |
| A | 3ed85e70 | 20260703 | 0.151175 | 1.385032 | 0.135211 | 0.015965 | 1.241839 | True | True |
| A | 3ed85e70 | 20260704 | 0.165805 | 1.223539 | 0.152189 | 0.013616 | 1.064542 | True | True |
| A | aa4ec2a5 | 20260702 | 2.529388 | 1.854326 | 1.546667 | 0.982721 | -0.183701 | False | False |
| A | aa4ec2a5 | 20260703 | 1.595409 | 1.635389 | 0.263689 | 1.331720 | 0.705841 | True | False |
| A | aa4ec2a5 | 20260704 | 0.993144 | 1.143457 | 0.307420 | 0.685724 | 0.493175 | True | False |
| A | 929ab4e9 | 20260702 | 0.052996 | 0.372177 | 0.046393 | 0.006604 | 0.322483 | True | True |
| A | 929ab4e9 | 20260703 | 0.038089 | 3.139768 | 0.049825 | 0.011736 | 3.095811 | True | True |
| A | 929ab4e9 | 20260704 | 0.054415 | 7.921286 | 0.058289 | 0.003874 | 7.864933 | True | True |
| B | 0dfd9992 | 20260702 | 0.004630 | 2.962732 | 0.003932 | 0.000698 | 2.958451 | True | True |
| B | 0dfd9992 | 20260703 | 0.003423 | 2.112274 | 0.004943 | 0.001520 | 2.108091 | True | True |
| B | 0dfd9992 | 20260704 | 0.001809 | 0.792612 | 0.003985 | 0.002176 | 0.789715 | True | True |
| B | 1c786137 | 20260702 | 2.145337 | 3.402656 | 2.239543 | 0.094205 | 1.210216 | True | True |
| B | 1c786137 | 20260703 | 2.388475 | 2.681060 | 2.443789 | 0.055314 | 0.264928 | True | True |
| B | 1c786137 | 20260704 | 3.870702 | 4.751534 | 3.337753 | 0.532949 | 1.147307 | True | True |
| B | 1f876c06 | 20260702 | 0.025615 | 1.516455 | 0.025577 | 0.000038 | 1.490860 | True | True |
| B | 1f876c06 | 20260703 | 0.031012 | 2.236239 | 0.050350 | 0.019338 | 2.195557 | True | True |
| B | 1f876c06 | 20260704 | 0.028680 | 1.445206 | 0.039851 | 0.011172 | 1.410940 | True | True |

## Tests

- pre-run focused tests: ................                                                         [100%]
16 passed in 3.12s
- post-run focused tests: ................                                                         [100%]
16 passed in 5.62s

## Failed Attempts And Retries

```json
{
  "failed_keys_later_completed": 25,
  "failure_message": "RNG state must be a torch.ByteTensor",
  "failure_type": "TypeError",
  "historical_failed_attempt_rows": 25,
  "note": "Historical failures occurred before optimization because torch.load map_location moved the saved CPU RNG state; the solver loader was patched to load initialization artifacts on CPU first, and all frozen keys later completed. No failed numerical outcome was discarded.",
  "unresolved_failed_runs": 0
}
```

## Notes

- The evaluator-only target-cell NLL used the mean of the final 100 serialized query probability checkpoints and did not affect optimizer updates, stopping, output voting, or answer selection.
- For cells lying inside the evaluator-known target shape but outside a model-active or semantically meaningful padded region, the stored padded probability may not carry an interpretable spatial meaning. This does not alter the paired comparison because F1, S, and F2 use the identical canonical probability representation and evaluator rule.
- Under this local implementation and frozen assay, the full CompressARC pipeline -- per-task optimization, MDL-style objective, multitensor representation, equivariance, and engineered operations together -- showed evidence of using paired support correspondence for hidden-query prediction.
