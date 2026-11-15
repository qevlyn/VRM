# VRM V1 - Spatial Evidence Rewrite summary

This is a capacity and unseen-task probe, not a claim of ARC mastery.

- command: `python -m vrm.train --mode v1_capacity50 --steps 1500 --batch-size 1 --seed 20260627`
- data_root: `/mnt/d/VRM/arc-prize-2026-arc-agi-2`
- device: `cpu`
- wall_clock_seconds: 2135.814
- average_seconds_per_optimizer_step: 1.410106
- train_episodes: 156
- unseen_episodes: 160

|step|split|mean_loss|token_accuracy_nonpad|exact_decoded_grid_accuracy|output_shape_match_rate|invalid_padding_pattern_rate|mean_nonpad_hamming_error|median_nonpad_hamming_error|
|---|---|---|---|---|---|---|---|---|
|0|train_50|1.870495|0.106624|0.000000|0.006410|0.987179|141.525641|91.000000|
|0|unseen_50|1.943470|0.033721|0.000000|0.000000|0.987500|133.068750|96.500000|
|50|train_50|2.103380|0.091814|0.000000|0.019231|0.948718|143.871795|89.500000|
|50|unseen_50|2.363175|0.040891|0.000000|0.000000|0.975000|132.081250|93.000000|
|100|train_50|2.429075|0.225387|0.000000|0.019231|0.910256|122.711538|81.000000|
|100|unseen_50|2.872529|0.101752|0.000000|0.000000|0.937500|123.700000|86.000000|
|250|train_50|3.072836|0.437057|0.000000|0.019231|0.596154|89.179487|62.000000|
|250|unseen_50|3.198740|0.375465|0.000000|0.000000|0.618750|86.006250|50.000000|
|500|train_50|18.023413|0.054951|0.000000|0.019231|0.000000|149.711538|81.000000|
|500|unseen_50|17.920460|0.083235|0.000000|0.000000|0.000000|126.250000|94.000000|
|1000|train_50|18.023413|0.054951|0.000000|0.019231|0.000000|149.711538|81.000000|
|1000|unseen_50|17.920460|0.083235|0.000000|0.000000|0.000000|126.250000|94.000000|
|1500|train_50|18.023413|0.054951|0.000000|0.019231|0.000000|149.711538|81.000000|
|1500|unseen_50|17.920460|0.083235|0.000000|0.000000|0.000000|126.250000|94.000000|

## Best train tasks

|task_id|min_hamming|exact_count|shape_rate|
|---|---:|---:|---:|
|d9fac9be|0|0|0.000|
|9720b24f|2|0|0.000|
|77fdfe62|4|0|0.000|
|d0f5fe59|4|0|0.000|
|3b4c2228|6|0|0.000|

## Best unseen tasks

|task_id|min_hamming|exact_count|shape_rate|
|---|---:|---:|---:|
|81c0276b|3|0|0.000|
|e1baa8a4|4|0|0.000|
|137eaa0f|6|0|0.000|
|7b7f7511|6|0|0.000|
|ca8de6ea|7|0|0.000|

V1_CAPACITY_VERDICT: RED_FLAG
V1_UNSEEN_VERDICT: SIGNAL_PRESENT
