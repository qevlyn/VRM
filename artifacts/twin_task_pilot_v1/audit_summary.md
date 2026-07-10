# VRM V1.2A - Counterfactual Twin-Task Suite Pilot

Synthetic causal testbed only. This is not ARC-2 training and not evidence that VRM solves ARC.

- seed: 20260628
- accepted_twin_pairs: 10000
- rejected_base_instances: 0
- artifact_path: `/mnt/d/VRM/artifacts/twin_task_pilot_v1`

## Probe sealed metrics

|probe|view|sealed_mean_acc|std|ci95|expected|pass|
|---|---|---:|---:|---|---|---|
|P1_QUERY_ONLY|query + candidates|0.500000|0.000000|[0.487352,0.512648]|chance|True|
|P2_SUPPORT_INPUT_BAG_ONLY|query + candidates + canonical support input bag|0.500000|0.000000|[0.487352,0.512648]|chance|True|
|P3_SUPPORT_OUTPUT_BAG_ONLY|query + candidates + canonical support output bag|0.500000|0.000000|[0.487352,0.512648]|chance|True|
|P4_UNPAIRED_SUPPORT_BAG|query + candidates + canonical input/output bags|0.500000|0.000000|[0.487352,0.512648]|chance|True|
|P5_RELATION_AWARE_PAIRED|query + candidates + coherent support pairs|1.000000|0.000000|[0.999360,1.000000]|above_chance|True|
|P6_RELATION_AWARE_DERANGED|query + candidates + deranged support pairs|0.251500|0.010591|[0.240683,0.262635]|deranged_control_le_55|True|

TWIN_PILOT_AUDIT_VERDICT: PASS
