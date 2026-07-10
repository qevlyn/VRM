# VRM Twin Probe v0 - Correspondence Capacity Test

Single-seed synthetic marker twin capacity probe only. This is not ARC-2 training and not an ARC-solving claim.

- seed: 20260628
- device: cpu
- runtime_seconds: 2142.572
- best_dev_checkpoint: `/mnt/d/VRM/checkpoints/twin_vrm_probe_v0/twin_vrm_probe_best_dev_candidate.pt`
- best_dev_step: 2750
- sealed_metrics_accessed_for_selection: False

## Training curve

|step|train loss|dev normal candidate score|dev joint twin accuracy|dev exact|dev mean Hamming|
|---:|---:|---:|---:|---:|---:|
|0|NA|0.500000|0.000000|0.000000|5.000000|
|250|0.227894|0.500000|0.000000|0.105000|4.475000|
|500|0.216888|0.495000|0.021000|0.288500|3.506500|
|750|0.118386|0.498250|0.044000|0.297500|3.366000|
|1000|0.116140|0.503500|0.038000|0.299000|3.349500|
|1250|0.142764|0.499500|0.047000|0.310000|3.342000|
|1500|0.122591|0.495750|0.017000|0.306500|3.343500|
|1750|0.094896|0.495500|0.018000|0.308000|3.339000|
|2000|0.119197|0.491500|0.032000|0.285500|3.405500|
|2250|0.117059|0.498250|0.034000|0.306500|3.361000|
|2500|0.122038|0.496250|0.011000|0.322000|3.296500|
|2750|0.138243|0.511000|0.025000|0.298000|3.369000|
|3000|0.121906|0.497250|0.026000|0.307500|3.343000|
|3250|0.124846|0.502500|0.022000|0.313000|3.338000|
|3500|0.122763|0.500750|0.031000|0.291000|3.389000|
|3750|0.110196|0.506250|0.039000|0.302000|3.315000|
|4000|0.128780|0.495500|0.033000|0.282000|3.386500|
|4250|0.127974|0.500750|0.027000|0.313500|3.325500|
|4500|0.130918|0.499500|0.034000|0.291500|3.383000|
|4750|0.120420|0.499250|0.048000|0.296000|3.314000|
|5000|0.123530|0.504250|0.040000|0.291500|3.358500|

## Final condition metrics

### dev

|condition|candidate score|tie rate|candidate exact|direct exact|shape match|color acc|mean Hamming|
|---|---:|---:|---:|---:|---:|---:|---:|
|NORMAL|0.511000|0.341000|0.587500|0.298000|1.000000|0.931245|3.369000|
|PAIR_ORDER_SHUFFLED|0.498000|0.337000|0.582000|0.291000|1.000000|0.930316|3.414500|
|SUPPORT_EVIDENCE_PATH_ZEROED|0.500000|0.757000|0.080000|0.040000|1.000000|0.898000|4.998000|
|TWIN_SWAP_SUPPORTS|0.489000|0.341000|0.587500|0.289500|1.000000|0.929663|3.446500|

- twin_pair_joint_normal_accuracy: 0.025000
- counterfactual_switch_rate: 0.044000
- correct_mapping_switch_rate: 0.025000
- twin_pair_mapping_flip_consistency: 0.025000

### sealed

|condition|candidate score|tie rate|candidate exact|direct exact|shape match|color acc|mean Hamming|
|---|---:|---:|---:|---:|---:|---:|---:|
|NORMAL|0.502500|0.321000|0.607000|0.305000|1.000000|0.931265|3.368000|
|PAIR_ORDER_SHUFFLED|0.496500|0.339000|0.594500|0.291500|1.000000|0.930173|3.421500|
|SUPPORT_EVIDENCE_PATH_ZEROED|0.500000|0.732000|0.081000|0.040500|1.000000|0.899041|4.947000|
|TWIN_SWAP_SUPPORTS|0.497500|0.321000|0.607000|0.302000|1.000000|0.930918|3.385000|

- twin_pair_joint_normal_accuracy: 0.035000
- counterfactual_switch_rate: 0.061000
- correct_mapping_switch_rate: 0.035000
- twin_pair_mapping_flip_consistency: 0.035000

Support-path-zeroed is supplementary/OOD and uses the existing `disable_support_evidence` flag.
