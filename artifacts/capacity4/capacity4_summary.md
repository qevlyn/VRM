# Capacity4 summary

This is a four-task capacity and wiring overfit run only. It is not ARC generalization.

## Run

- command: `python -m vrm.train --mode capacity4`
- device: `cpu`
- wall_clock_seconds: 768.169
- average_seconds_per_optimizer_step: 0.510099
- optimizer_steps: 1500
- batch_size: 1
- learning_rate: 0.001
- optimizer: AdamW {'name': 'AdamW', 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.01, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}
- seed: 12345
- model_parameter_count: 653067
- leave_one_out_episode_count: 17

## Aggregate metrics

|step|mean_loss|token_accuracy_all|token_accuracy_nonpad|exact_padded_grid_accuracy|exact_decoded_grid_accuracy|output_shape_match_rate|invalid_padding_pattern_rate|mean_nonpad_hamming_error|median_nonpad_hamming_error|minimum_nonpad_hamming_error|
|---|---|---|---|---|---|---|---|---|---|---|
|0|3.177540|0.017582|0.028728|0.000000|0.000000|0.000000|1.000000|125.294118|96|28|
|50|0.774109|0.807582|0.738714|0.000000|0.000000|0.000000|1.000000|33.705882|25|8|
|100|0.535334|0.922157|0.507524|0.000000|0.000000|0.058824|0.882353|63.529412|35|8|
|250|0.313849|0.946144|0.813953|0.000000|0.000000|0.058824|0.823529|24.000000|25|8|
|500|0.179650|0.978758|0.857729|0.000000|0.000000|0.882353|0.117647|18.352941|21|3|
|1000|0.064985|0.990654|0.934793|0.294118|0.294118|1.000000|0.000000|8.411765|8|0|
|1500|0.024050|0.996797|0.977656|0.588235|0.588235|1.000000|0.000000|2.882353|0|0|

## Best per-episode results at step 1500

|task_id|query_pair_index|target_shape|predicted_shape|nonpad_hamming_error|exact_decoded|valid_rectangle|
|---|---:|---|---|---:|---|---|
|00576224|0|6x6|6x6|0|True|True|
|00576224|1|6x6|6x6|0|True|True|
|007bbfb7|0|9x9|9x9|0|True|True|
|007bbfb7|1|9x9|9x9|0|True|True|
|007bbfb7|2|9x9|9x9|0|True|True|
|007bbfb7|3|9x9|9x9|0|True|True|
|007bbfb7|4|9x9|9x9|0|True|True|
|009d5c81|0|14x14|14x14|0|True|True|
|00d62c1b|3|20x20|20x20|0|True|True|
|00d62c1b|4|6x6|6x6|0|True|True|
|00d62c1b|0|10x10|10x10|1|False|True|
|009d5c81|3|14x14|14x14|3|False|True|
|00d62c1b|1|10x10|10x10|3|False|True|
|00d62c1b|2|10x10|10x10|3|False|True|
|009d5c81|4|14x14|14x14|7|False|True|
|009d5c81|1|14x14|14x14|8|False|True|
|009d5c81|2|14x14|14x14|24|False|True|

## Verdict

- exact_decoded_grids_step_1500: 10
- minimum_nonpad_hamming_error_step_1500: 0
- median_nonpad_hamming_error_step_1500: 0
- verdict: CAPACITY_SIGNAL_STRONG
