# Pair Key-Value Reader R1

Standalone learned associative-memory unit test on the frozen Twin dataset.

|split|condition|retrieval top1|mean relevant mass|direct exact|foreground acc|all-cell acc|candidate score|tie rate|twin same-grid|correct swap|pair-order same logical|mean Hamming|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|train|NORMAL|1.000000|0.999969|1.000000|1.000000|1.000000|1.000000|0.000000|0.000000|1.000000|1.000000|0.000000|
|train|TWIN_SWAP_SUPPORTS|1.000000|0.999969|0.000000|0.000000|0.897959|0.000000|0.000000|0.000000|1.000000|1.000000|5.000000|
|train|PAIR_ORDER_SHUFFLED|1.000000|0.999969|1.000000|1.000000|1.000000|1.000000|0.000000|0.000000|1.000000|1.000000|0.000000|
|dev|NORMAL|1.000000|0.999969|1.000000|1.000000|1.000000|1.000000|0.000000|0.000000|1.000000|1.000000|0.000000|
|dev|TWIN_SWAP_SUPPORTS|1.000000|0.999969|0.000000|0.000000|0.897959|0.000000|0.000000|0.000000|1.000000|1.000000|5.000000|
|dev|PAIR_ORDER_SHUFFLED|1.000000|0.999969|1.000000|1.000000|1.000000|1.000000|0.000000|0.000000|1.000000|1.000000|0.000000|
|sealed|NORMAL|1.000000|0.999968|1.000000|1.000000|1.000000|1.000000|0.000000|0.000000|1.000000|1.000000|0.000000|
|sealed|TWIN_SWAP_SUPPORTS|1.000000|0.999968|0.000000|0.000000|0.897959|0.000000|0.000000|0.000000|1.000000|1.000000|5.000000|
|sealed|PAIR_ORDER_SHUFFLED|1.000000|0.999968|1.000000|1.000000|1.000000|1.000000|0.000000|0.000000|1.000000|1.000000|0.000000|

## Sealed Gates

- oracle_direct_exact: 1.000000 == 1.000000 -> PASS
- retrieval_top1_accuracy: 1.000000 >= 0.950000 -> PASS
- mean_relevant_pair_mass: 0.999968 >= 0.950000 -> PASS
- direct_grid_exact_accuracy: 1.000000 >= 0.950000 -> PASS
- foreground_color_accuracy: 1.000000 >= 0.950000 -> PASS
- twin_same_predicted_grid_rate: 0.000000 <= 0.050000 -> PASS
- correct_counterfactual_switch_rate: 1.000000 >= 0.950000 -> PASS
- twin_pair_mapping_flip_consistency: 1.000000 >= 0.950000 -> PASS
- normal_vs_pair_order_same_grid_rate: 1.000000 >= 0.990000 -> PASS
- normal_vs_pair_order_same_logical_pair_rate: 1.000000 >= 0.990000 -> PASS

Verdict: `PASS`

If R1 passes, it proves only that this standalone learned reader solves the marker Twin correspondence unit test.
It does not prove VRM is repaired, ARC is solved, or that raw support-output copying is sufficient for ARC.
