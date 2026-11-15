# P6 Provenance

Classification:

```text
P6_IS_SEPARATELY_TRAINED_ON_DERANGED_SUPPORTS
```

Evidence:

- `vrm/twin_tasks/run_pilot.py` builds `all_runs` by looping over every `probe` in `PROBE_NAMES` and every seed, then calls `train_probe(probe_name=probe, ...)`. This means P6 has its own training runs rather than reusing the trained P5 probe under an altered evaluation view.
- `vrm/twin_tasks/probes.py` routes `P6_RELATION_AWARE_DERANGED` through `feature_scores(..., "P6_RELATION_AWARE_DERANGED", seed)`, which calls `relation_scores(record, deranged=True, seed=seed)`.
- `deranged_supports(record, seed)` deterministically permutes support outputs within each episode while preserving support inputs and output bags. The resulting P6 features are materialized inside `tensors_for_probe(...)` for train, dev, and sealed splits.
- `artifacts/twin_task_pilot_v1/probe_config.json` lists `P6_RELATION_AWARE_DERANGED` as one of the trained `probe_names`.
- `artifacts/twin_task_pilot_v1/probe_train_logs.jsonl` contains rows with `"probe": "P6_RELATION_AWARE_DERANGED"` and per-epoch train/dev accuracies, confirming separate optimization records for P6.
- `artifacts/twin_task_pilot_v1/probe_sealed_metrics.json` reports P6 sealed accuracies near 25% across seeds, generated from those separate P6 runs.

The below-chance P6 result is not a blocker for the VRM Twin Probe. Because the P6 view deterministically deranges the coherent support correspondence while preserving the same query and support bags, a simple relation scorer can become anti-correlated: matching the query to its deranged support output may systematically prefer the alternate candidate instead of producing a random tie. Thus below-chance accuracy can arise from deterministic anti-mapping in the deranged control, not from P6 having access to forbidden labels or candidates during the VRM probe.
