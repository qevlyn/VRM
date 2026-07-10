# VRM V1.1 - Masked Query Spatial Evidence Rewrite summary

This is an experimental ARC-2 capacity probe, not a claim that VRM solves ARC.

- command: `python -m vrm.train --mode v1_1_capacity50 --steps 1500 --batch-size 1 --seed 20260627 --learning-rate 2e-4 --warmup-steps 100 --min-learning-rate 2e-5 --grad-clip-norm 1.0 --shape-loss-weight 0.5`
- data_root: `/mnt/d/VRM/arc-prize-2026-arc-agi-2`
- device: `cpu`
- task_selection_status: `reused_prior_v1_manifest`
- wall_clock_seconds: 450.683
- average_seconds_per_optimizer_step: 0.300455
- train_episodes: 156
- unseen_episodes: 160
- best_train_hamming_checkpoint_step: 1500

|step|split|total_loss|color_loss|shape_height_accuracy|shape_width_accuracy|output_shape_match_rate|target_rectangle_color_accuracy|exact_decoded_grid_accuracy|mean_target_rectangle_hamming_error|median_target_rectangle_hamming_error|
|---|---|---|---|---|---|---|---|---|---|---|
|0|train_50|3.212437|1.395360|0.076923|0.006410|0.000000|0.675636|0.000000|51.384615|24.500000|
|0|unseen_50|3.408338|1.613698|0.025000|0.000000|0.000000|0.683444|0.000000|43.593750|24.000000|
|50|train_50|2.769562|1.173397|0.173077|0.076923|0.000000|0.675353|0.000000|51.429487|24.500000|
|50|unseen_50|3.140431|1.488655|0.150000|0.050000|0.006250|0.679813|0.000000|44.093750|24.000000|
|100|train_50|2.662668|1.143851|0.185897|0.160256|0.134615|0.674989|0.000000|51.487179|24.500000|
|100|unseen_50|3.095523|1.411187|0.143750|0.131250|0.112500|0.659163|0.000000|46.937500|25.000000|
|250|train_50|2.232690|1.077984|0.371795|0.423077|0.326923|0.675879|0.000000|51.346154|24.500000|
|250|unseen_50|2.898154|1.458842|0.206250|0.181250|0.106250|0.668376|0.000000|45.668750|25.000000|
|500|train_50|1.667580|0.980946|0.628205|0.602564|0.519231|0.678388|0.000000|50.948718|23.000000|
|500|unseen_50|2.803295|1.458140|0.275000|0.231250|0.175000|0.668694|0.000000|45.625000|24.000000|
|1000|train_50|1.111443|0.825158|0.916667|0.903846|0.865385|0.693319|0.038462|48.583333|21.000000|
|1000|unseen_50|2.759369|1.459937|0.331250|0.237500|0.168750|0.674639|0.000000|44.806250|24.500000|
|1500|train_50|0.938860|0.758184|0.955128|0.948718|0.935897|0.702464|0.044872|47.134615|21.000000|
|1500|unseen_50|2.824177|1.457084|0.318750|0.275000|0.187500|0.678179|0.000000|44.318750|24.000000|

## Best train tasks

|task_id|min_hamming|exact_count|shape_rate|
|---|---:|---:|---:|
|3b4c2228|0|2|1.000|
|87ab05b8|0|1|1.000|
|d9fac9be|0|4|1.000|
|77fdfe62|1|0|0.667|
|e57337a4|1|0|1.000|

## Best unseen tasks

|task_id|min_hamming|exact_count|shape_rate|
|---|---:|---:|---:|
|253bf280|1|0|0.250|
|4a1cacc2|2|0|0.250|
|7b7f7511|3|0|0.667|
|e1baa8a4|3|0|0.000|
|1caeab9d|4|0|0.333|

V1_1_STABILITY_VERDICT: STABLE
V1_1_CAPACITY_VERDICT: IMPROVED
V1_1_UNSEEN_VERDICT: SIGNAL_PRESENT
