# Capacity50 summary

This is a capacity and unseen-task probe only. It is not ARC generalization.

## Run

- command: `python -m vrm.train --mode capacity50 --steps 1500 --batch-size 1 --seed 20260627`
- data_root: `/mnt/d/VRM/arc-prize-2026-arc-agi-2`
- device: `cpu`
- wall_clock_seconds: 796.765
- average_seconds_per_optimizer_step: 0.501410
- train_episodes: 156
- unseen_episodes: 160
- capacity_verdict: CAPACITY_50_PARTIAL
- unseen_verdict: UNSEEN_SIGNAL_PRESENT

## Aggregate metrics

|step|split|mean_loss|token_accuracy_nonpad|exact_decoded_grid_accuracy|output_shape_match_rate|invalid_padding_pattern_rate|mean_nonpad_hamming_error|median_nonpad_hamming_error|
|---|---|---|---|---|---|---|---|---|
|0|train_50|2.781934|0.072108|0.000000|0.000000|1.000000|146.993590|82.500000|
|0|unseen_50|2.811138|0.062222|0.000000|0.000000|1.000000|129.143750|92.500000|
|50|train_50|1.114905|0.162708|0.000000|0.000000|1.000000|132.641026|86.500000|
|50|unseen_50|1.161730|0.127893|0.000000|0.000000|1.000000|120.100000|92.000000|
|100|train_50|0.944695|0.439202|0.000000|0.012821|0.987179|88.839744|51.000000|
|100|unseen_50|1.032279|0.427430|0.000000|0.000000|1.000000|78.850000|51.500000|
|250|train_50|0.717994|0.579007|0.000000|0.012821|0.923077|66.692308|28.500000|
|250|unseen_50|0.866825|0.489834|0.000000|0.006250|0.968750|70.256250|38.500000|
|500|train_50|0.773202|0.473840|0.000000|0.006410|0.993590|83.352564|54.000000|
|500|unseen_50|1.002582|0.350005|0.000000|0.000000|1.000000|89.512500|66.500000|
|1000|train_50|0.590490|0.646461|0.000000|0.019231|0.980769|56.006410|25.000000|
|1000|unseen_50|0.867622|0.559408|0.000000|0.012500|0.975000|60.675000|34.000000|
|1500|train_50|0.442830|0.709020|0.000000|0.108974|0.884615|46.096154|23.000000|
|1500|unseen_50|0.809331|0.567305|0.000000|0.043750|0.956250|59.587500|33.000000|

## Best train tasks

|task_id|min_hamming|exact_count|shape_rate|
|---|---:|---:|---:|
|bbc9ae5d|0|0|0.000|
|d0f5fe59|1|0|0.000|
|d9fac9be|1|0|0.000|
|e57337a4|1|0|0.000|
|3b4c2228|2|0|0.000|

## Best unseen tasks

|task_id|min_hamming|exact_count|shape_rate|
|---|---:|---:|---:|
|253bf280|0|0|0.000|
|4a1cacc2|2|0|0.000|
|7b7f7511|2|0|0.000|
|e1baa8a4|3|0|0.000|
|5ad4f10b|5|0|0.000|
