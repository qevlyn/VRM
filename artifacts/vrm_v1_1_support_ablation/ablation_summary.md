# VRM V1.1 - Support Mapping Counterfactual Ablation

Evaluation-only diagnostic. No optimizer, gradient update, or checkpoint modification was used.

- command: `python -m vrm.train --mode v1_1_support_ablation --checkpoint /mnt/d/VRM/checkpoints/vrm_v1_1_best_train_hamming.pt --seed 20260627 --pytest-result pytest -q: 31 passed in 5.05s`
- checkpoint: `/mnt/d/VRM/checkpoints/vrm_v1_1_best_train_hamming.pt`
- checkpoint_sha256_before: `b86ba9fd645870e185ef1c54fbec98e5437aeac094d0e290e19bc388a5722370`
- checkpoint_sha256_after: `b86ba9fd645870e185ef1c54fbec98e5437aeac094d0e290e19bc388a5722370`
- task_manifest: `/mnt/d/VRM/artifacts/vrm_v1_1_capacity50/vrm_v1_1_task_manifest.json`
- train_episodes: 156
- unseen_episodes: 160
- train_output_deranged_eligible: 140
- unseen_output_deranged_eligible: 148

## Counterfactual metrics

|split|condition|shape_match|target_color_acc|transformation_acc|exact|mean_hamming|
|---|---|---:|---:|---:|---:|---:|
|train_50|NORMAL|0.935897|0.702464|0.247191|0.044872|47.134615|
|unseen_50|NORMAL|0.187500|0.678179|0.279529|0.000000|44.318750|
|train_50|OUTPUT_DERANGED_WITHIN_TASK|0.907143|0.698531|0.240575|0.050000|48.228571|
|unseen_50|OUTPUT_DERANGED_WITHIN_TASK|0.168919|0.673266|0.239695|0.000000|43.378378|
|train_50|PAIR_ORDER_SHUFFLED|0.814103|0.698256|0.241277|0.044872|47.801282|
|unseen_50|PAIR_ORDER_SHUFFLED|0.181250|0.679041|0.279693|0.000000|44.200000|
|train_50|OUTPUT_COLOR_PERMUTED|0.730769|0.695666|0.219791|0.051282|48.211538|
|unseen_50|OUTPUT_COLOR_PERMUTED|0.181250|0.659027|0.254370|0.000000|46.956250|
|train_50|SUPPORT_EVIDENCE_PATH_ZEROED|0.435897|0.682111|0.178987|0.000000|50.358974|
|unseen_50|SUPPORT_EVIDENCE_PATH_ZEROED|0.156250|0.662340|0.256004|0.000000|46.500000|

## Verdicts

- train_50: SUPPORT_PATH_USE_ONLY
- unseen_50: SUPPORT_MAPPING_USE_NOT_DETECTED

Support-path zeroing is an out-of-distribution stress test and is supplementary to output derangement.
