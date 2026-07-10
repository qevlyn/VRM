# Transport Recolor R5A-1A Matchability Audit

This is a no-training finite-grid audit for the current R5A-1A generic shared-renderer family. It does not change R5A-0 V2, the R5A-0 order audit, the R5A-1A preregistration, the R5A-1A preflight report, any data split, any checkpoint, or any historical verdict.

The audit asks only whether the current generic shared-renderer family contains a candidate inside the frozen matching rectangle under the frozen preflight accounting convention:

```text
parameter_ratio_to_v2 in [0.90, 1.10]
forward_flops_per_valid_output_cell_ratio_to_v2 in [0.85, 1.15]
```

The conclusion is limited to the declared finite grid and the frozen canonical profiling batch. It is not a universal claim about all possible architectures.

## Additive Files

- `/mnt/d/VRM/vrm/twin_tasks/transport_recolor_r5a1_matchability_audit.py`
- `/mnt/d/VRM/tests/test_transport_recolor_r5a1_matchability_audit.py`
- `/mnt/d/VRM/docs/TRANSPORT_RECOLOR_R5A1A_MATCHABILITY_AUDIT.md`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a1a_matchability_audit/search_domain.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a1a_matchability_audit/candidate_ledger.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a1a_matchability_audit/pareto_frontier.json`
- `/mnt/d/VRM/artifacts/transport_recolor_r5a1a_matchability_audit/matchability_report.json`

## Search Domain

The audit uses exhaustive Cartesian enumeration. It does not use coordinate descent, random search, Bayesian optimization, runtime profiling, training loss, accuracy, train/dev/sealed metrics, or any outcome metric.

Default finite grid:

```text
hidden_width:
  every multiple of 8 from 32 through 256 inclusive
  plus widths from existing R5A-1 shared-renderer/preflight configs

encoder_layers:
  0, 1, 2

decoder_layers:
  1, 2, 3, 4

attention_heads:
  1, 2, 4, 8
  only retained when valid for the current implementation

mlp_expansion:
  2, 3, 4, 5, 6
```

The exact `shared_small`, `shared_matched`, and `shared_large` configurations from the frozen preflight report are included. The shared-renderer default configs are also recorded as existing generic-family anchors.

## Accounting

The audit reuses the R5A-1A preflight conventions:

- canonical profiling batch and tensor hash;
- V2 parameter-count convention;
- analytic forward-FLOP convention;
- FLOPs per valid output cell;
- activation proxy convention;
- information-firewall policy.

For every valid candidate, the candidate ledger records parameters, embedding parameters, active parameters on the canonical batch where measurable, inactive lookup rows if any, analytic FLOPs, activation proxy, ratios to V2, tolerance membership, firewall validity, and whether trainable groups are reachable.

## Pareto Analysis

The Pareto frontier is computed with objectives:

```text
abs(parameter_ratio_to_v2 - 1.0)
abs(flop_ratio_to_v2 - 1.0)
```

The report lists all jointly feasible candidates if any, the nearest joint candidate by normalized distance, nearest parameter-matched candidate, nearest FLOP-matched candidate, full frontier artifact, mismatch patterns, and boundary assessment.

## Commands

Run the audit:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m vrm.twin_tasks.transport_recolor_r5a1_matchability_audit \
  --out-dir artifacts/transport_recolor_r5a1a_matchability_audit \
  --pytest-result artifacts/transport_recolor_r5a1a_matchability_audit/pytest_result.txt
```

Run tests:

```bash
cd /mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -m pytest -q
```

## Verdicts

- `R5A1A_MATCHABILITY_PASS`: at least one valid generic candidate is inside both frozen tolerance intervals.
- `R5A1A_MATCHABILITY_GRID_EXPANSION_NEEDED`: no candidate is feasible, and the nearest Pareto candidate is blocked by an obvious unsearched expansion of an existing neutral knob.
- `R5A1A_MATCHABILITY_CURRENT_FAMILY_INSUFFICIENT`: no candidate is feasible in the fully enumerated declared domain, deterministic accounting and firewall checks pass, and the miss is not merely an obvious boundary-extension issue.
- `R5A1A_MATCHABILITY_INCONCLUSIVE`: determinism, accounting, canonical-batch reuse, firewall, exhaustive enumeration, or protected-source checks fail.
