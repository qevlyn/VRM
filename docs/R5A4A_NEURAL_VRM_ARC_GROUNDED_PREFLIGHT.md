# R5A-4A ARC-Grounded Neural VRM Preflight

## 1. Frozen setup

```json
{
  "architecture": "VRM V1.1 - Masked Query Spatial Evidence Rewrite",
  "data_root": "/mnt/d/VRM/arc-prize-2026-arc-agi-2",
  "experiment": "R5A-4A ARC-Grounded Neural VRM Preflight",
  "initial_state_path": "/mnt/d/VRM/artifacts/r5a4a_neural_vrm_arc_grounded_preflight/initial_state_before_any_optimizer_step.pt",
  "initialization_seed": 20260701,
  "model_interface_audit_passed": true,
  "pre_training_commitment": "/mnt/d/VRM/artifacts/r5a4a_neural_vrm_arc_grounded_preflight/pre_training_commitment.json",
  "reused_completed_training": true,
  "selected_checkpoint_step": 12000,
  "selected_trained_state_path": "/mnt/d/VRM/artifacts/r5a4a_neural_vrm_arc_grounded_preflight/selected_trained_state.pt",
  "split_counts": {
    "sentinel_loo_fold_count": 156,
    "sentinel_task_count": 48,
    "training_loo_fold_count": 2876,
    "training_task_count": 888,
    "validation_loo_fold_count": 200,
    "validation_task_count": 64
  }
}
```

## 2. Measured trained-vs-fresh results

```json
{
  "fresh_init": {
    "cell_accuracy_inside_true_target_rectangle": 0.5896616381147192,
    "condition": "FRESH_INIT",
    "exact_full_grid_accuracy": 0.0,
    "exact_output_shape_accuracy": 0.0,
    "fold_count": 156,
    "mean_color_negative_log_likelihood_inside_true_target_rectangle": 1.6120966248787367,
    "mean_fold_cell_accuracy_inside_true_target_rectangle": 0.5141414519043886,
    "mean_height_negative_log_likelihood": 3.5251411765049667,
    "mean_inference_runtime_per_fold_seconds": 0.01879559657692635,
    "mean_total_existing_model_loss": 3.3872355971580896,
    "mean_width_negative_log_likelihood": 3.575414697329203,
    "supported": true,
    "task_count": 48,
    "total_target_rectangle_cells": 23466
  },
  "paired_bootstrap_intervals": {
    "cell_accuracy_improvement_trained_minus_fresh": {
      "bootstrap_replicates": 10000,
      "ci95_high": 0.19753759409677593,
      "ci95_low": 0.11911124529900557,
      "confidence_level": 0.95,
      "count": 156,
      "mean": 0.15751995313476577
    },
    "color_nll_improvement_fresh_minus_trained": {
      "bootstrap_replicates": 10000,
      "ci95_high": 0.7250722191081598,
      "ci95_low": 0.5473421207414224,
      "confidence_level": 0.95,
      "count": 156,
      "mean": 0.6326693275418037
    },
    "exact_grid_improvement_trained_minus_fresh": {
      "bootstrap_replicates": 10000,
      "ci95_high": 0.0,
      "ci95_low": 0.0,
      "confidence_level": 0.95,
      "count": 156,
      "mean": 0.0
    },
    "paired_fold_count": 156,
    "positive_values_favor": "TRAINED_FULL_SUPPORT over FRESH_INIT",
    "shape_accuracy_improvement_trained_minus_fresh": {
      "bootstrap_replicates": 10000,
      "ci95_high": 0.7371794871794872,
      "ci95_low": 0.5897435897435898,
      "confidence_level": 0.95,
      "count": 156,
      "mean": 0.6666666666666666
    },
    "total_loss_improvement_fresh_minus_trained": {
      "bootstrap_replicates": 10000,
      "ci95_high": 2.089262945147661,
      "ci95_low": 1.79964857452955,
      "confidence_level": 0.95,
      "count": 156,
      "mean": 1.9473817840409584
    }
  },
  "trained_full_support": {
    "cell_accuracy_inside_true_target_rectangle": 0.7241114804397852,
    "condition": "TRAINED_FULL_SUPPORT",
    "exact_full_grid_accuracy": 0.0,
    "exact_output_shape_accuracy": 0.6666666666666666,
    "fold_count": 156,
    "mean_color_negative_log_likelihood_inside_true_target_rectangle": 0.9794272973369329,
    "mean_fold_cell_accuracy_inside_true_target_rectangle": 0.6716614050391545,
    "mean_height_negative_log_likelihood": 0.8675906503892923,
    "mean_inference_runtime_per_fold_seconds": 0.015191615608994898,
    "mean_total_existing_model_loss": 1.4398538131171312,
    "mean_width_negative_log_likelihood": 0.9741154135030322,
    "supported": true,
    "task_count": 48,
    "total_target_rectangle_cells": 23466
  }
}
```

## 3. Measured support-ablation results

```json
{
  "condition_aggregates": {
    "TRAINED_FULL_SUPPORT": {
      "cell_accuracy_inside_true_target_rectangle": 0.7241114804397852,
      "condition": "TRAINED_FULL_SUPPORT",
      "exact_full_grid_accuracy": 0.0,
      "exact_output_shape_accuracy": 0.6666666666666666,
      "fold_count": 156,
      "mean_color_negative_log_likelihood_inside_true_target_rectangle": 0.9794272973369329,
      "mean_fold_cell_accuracy_inside_true_target_rectangle": 0.6716614050391545,
      "mean_height_negative_log_likelihood": 0.8675906503892923,
      "mean_inference_runtime_per_fold_seconds": 0.015191615608994898,
      "mean_total_existing_model_loss": 1.4398538131171312,
      "mean_width_negative_log_likelihood": 0.9741154135030322,
      "supported": true,
      "task_count": 48,
      "total_target_rectangle_cells": 23466
    },
    "TRAINED_PERMUTED_SUPPORT_PAIR_ORDER": {
      "cell_accuracy_inside_true_target_rectangle": 0.7288843433052076,
      "condition": "TRAINED_PERMUTED_SUPPORT_PAIR_ORDER",
      "exact_full_grid_accuracy": 0.0,
      "exact_output_shape_accuracy": 0.6858974358974359,
      "fold_count": 156,
      "mean_color_negative_log_likelihood_inside_true_target_rectangle": 0.9805207543839247,
      "mean_fold_cell_accuracy_inside_true_target_rectangle": 0.6723327216150521,
      "mean_height_negative_log_likelihood": 0.9105509509990952,
      "mean_inference_runtime_per_fold_seconds": 0.015181457820507406,
      "mean_total_existing_model_loss": 1.4556786123758707,
      "mean_width_negative_log_likelihood": 0.9900805180894121,
      "supported": true,
      "task_count": 48,
      "total_target_rectangle_cells": 23466
    },
    "TRAINED_QUERY_ONLY": {
      "condition": "TRAINED_QUERY_ONLY",
      "failure_reason": "QUERY_ONLY_ABLATION_UNSUPPORTED_BY_CURRENT_ARCHITECTURE",
      "fold_count": 0,
      "supported": false,
      "task_count": 48
    },
    "TRAINED_ZEROED_SUPPORT_OUTPUTS": {
      "cell_accuracy_inside_true_target_rectangle": 0.5031961135259524,
      "condition": "TRAINED_ZEROED_SUPPORT_OUTPUTS",
      "exact_full_grid_accuracy": 0.0,
      "exact_output_shape_accuracy": 0.5769230769230769,
      "fold_count": 156,
      "mean_color_negative_log_likelihood_inside_true_target_rectangle": 3.0470044131462393,
      "mean_fold_cell_accuracy_inside_true_target_rectangle": 0.40551992096726547,
      "mean_height_negative_log_likelihood": 1.0726053194533354,
      "mean_inference_runtime_per_fold_seconds": 0.015183578916666928,
      "mean_total_existing_model_loss": 3.6124926641201363,
      "mean_width_negative_log_likelihood": 1.1893476752297711,
      "supported": true,
      "task_count": 48,
      "total_target_rectangle_cells": 23466
    }
  },
  "support_ablation_summary": {
    "ablations": {
      "TRAINED_PERMUTED_SUPPORT_PAIR_ORDER": {
        "condition": "TRAINED_PERMUTED_SUPPORT_PAIR_ORDER",
        "decoded_grid_change_rate": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.9102564102564102,
          "ci95_low": 0.8012820512820513,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.8589743589743589
        },
        "delta_cell_accuracy_full_minus_condition": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.006539361534933823,
          "ci95_low": -0.007525604342170321,
          "confidence_level": 0.95,
          "count": 156,
          "mean": -0.0006713165758976064
        },
        "delta_color_nll_condition_minus_full": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.009795697931295786,
          "ci95_low": -0.007732897423780882,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.0010934570469917394
        },
        "delta_exact_grid_accuracy_full_minus_condition": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.0,
          "ci95_low": 0.0,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.0
        },
        "delta_shape_accuracy_full_minus_condition": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.00641025641025641,
          "ci95_low": -0.05128205128205128,
          "confidence_level": 0.95,
          "count": 156,
          "mean": -0.019230769230769232
        },
        "delta_total_loss_condition_minus_full": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.04633477253791613,
          "ci95_low": -0.012199736845034819,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.01582479925873952
        },
        "mean_js_divergence_color_distributions": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.004869504471995415,
          "ci95_low": 0.002882808659271669,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.0037709015461327364
        },
        "paired_fold_count": 156,
        "positive_values_favor": "TRAINED_FULL_SUPPORT",
        "shape_prediction_change_rate": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.14743589743589744,
          "ci95_low": 0.05128205128205128,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.09615384615384616
        }
      },
      "TRAINED_QUERY_ONLY": {
        "condition": "TRAINED_QUERY_ONLY",
        "failure_reason": "QUERY_ONLY_ABLATION_UNSUPPORTED_BY_CURRENT_ARCHITECTURE",
        "supported": false
      },
      "TRAINED_ZEROED_SUPPORT_OUTPUTS": {
        "condition": "TRAINED_ZEROED_SUPPORT_OUTPUTS",
        "decoded_grid_change_rate": {
          "bootstrap_replicates": 10000,
          "ci95_high": 1.0,
          "ci95_low": 0.967948717948718,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.9871794871794872
        },
        "delta_cell_accuracy_full_minus_condition": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.31553748527267456,
          "ci95_low": 0.21931069172607282,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.26614148407188903
        },
        "delta_color_nll_condition_minus_full": {
          "bootstrap_replicates": 10000,
          "ci95_high": 2.3420728766001186,
          "ci95_low": 1.7904083352440443,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 2.067577115809306
        },
        "delta_exact_grid_accuracy_full_minus_condition": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.0,
          "ci95_low": 0.0,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.0
        },
        "delta_shape_accuracy_full_minus_condition": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.14743589743589744,
          "ci95_low": 0.038461538461538464,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.08974358974358974
        },
        "delta_total_loss_condition_minus_full": {
          "bootstrap_replicates": 10000,
          "ci95_high": 2.45925915069305,
          "ci95_low": 1.898378375153511,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 2.172638851003005
        },
        "mean_js_divergence_color_distributions": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.30869505989268065,
          "ci95_low": 0.24569513986853025,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.27760229596395164
        },
        "paired_fold_count": 156,
        "positive_values_favor": "TRAINED_FULL_SUPPORT",
        "shape_prediction_change_rate": {
          "bootstrap_replicates": 10000,
          "ci95_high": 0.3974358974358974,
          "ci95_low": 0.25,
          "confidence_level": 0.95,
          "count": 156,
          "mean": 0.32051282051282054
        }
      }
    },
    "baseline_condition": "TRAINED_FULL_SUPPORT",
    "bootstrap_replicates": 10000,
    "positive_values_favor": "TRAINED_FULL_SUPPORT"
  }
}
```

## 4. No-ID causal audit

```json
{
  "audit_name": "R5A-4A no-ID causal audit",
  "dynamic": {
    "dynamic_no_id_pass": true,
    "failures": [],
    "fold_count": 20,
    "rows": [
      {
        "audit_index": 0,
        "changed_task_id": "d94c3b52_noid_0",
        "decoded_digest": {
          "decoded_color_digest": "7e5a7d3cd969b8ad990d8b93a0eddb54a254da4a7caf71aa2fcf0327a66473b7",
          "decoded_height_digest": "8438c83e47934e8dd87ead74167d3ec2844f8e7addcfb6003c1a5d76b17529f0",
          "decoded_width_digest": "45c8f16b6bd4e48a4232be36869b2a7054762e8099f0b3a52da2e315ba9adfa6"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "b7c885cdded52c3c4ce4b36e0bf5c6aae9e642262c23fb104ab3df0c07db6ce3",
          "query_inputs": "fa5796c31caba9bd556eda6d48ffd264a991105aa19a25e997e9b03050f02b83",
          "support_evidence_colors": "347e7dd4873adaec0e35c5b0a3b0af947d29345c55f8991eac60424a20ed34b4",
          "support_evidence_cols": "016a76705a6beaf2d0e50f016c1bf183da9af9b136ba1597023b304afc865f55",
          "support_evidence_mask": "cd4d8ea1452b42dc26916b99d4a1a52944a84867dfecb64e4a2184cfe2e69fd8",
          "support_evidence_pair_indices": "8c29819a92ec531101b589236043263da5fe05036860ef7e9700e049ee457b25",
          "support_evidence_roles": "25b2f8ff83c048244706bdf34c1fff36a839e543f3d3f074c5cbfc7a64b3d0fb",
          "support_evidence_rows": "d8dce28963ece17923772bd9e27ea99bcdfad062dab43b0a2be3df817604f8cb"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "d94c3b52",
        "output_digest": {
          "color_log_probs": "710d0420eeb56a0c25ce45976f1c0604c4397acc6bfb8be11b9a5ee251fd21d7",
          "color_logits": "710d0420eeb56a0c25ce45976f1c0604c4397acc6bfb8be11b9a5ee251fd21d7",
          "color_probabilities": "7f84ab024f0594f0c19c55e8b339d56c565bfce900344e3668e5baad83904d0b",
          "copy_attention": "d37e80c5c442995e831748fc51cdf0c9ea180fcf9286d305b996a3c8182c1e3c",
          "copy_distribution": "ecde1b58d6f6f30658197ef6ec4de9fe609793ffa14cb9abe34becc631f291e5",
          "copy_entropy": "93dc246e41ab4329c0cb8773089c28ea2cba42dc9504399d28776b602c1a1ba0",
          "copy_gate": "6ea4d7e940090c198a4089e14baae7eb3b76945658fe6d936d1582bb5d54c290",
          "final_canvas": "a92c921db339e7a5e619bb373f8c303e1eedd1cedd19c55206a604b68bd1e97c",
          "final_rewrite_state": "fbaedf6c41d43ccb743500a71f3be71295e593ef18f70d465a841cdec5b55c8a",
          "height_logits": "8760056be42a88700a1c65b6d4a6c8198ff63a12a5f60b9108d8be340ea6df4c",
          "logits": "710d0420eeb56a0c25ce45976f1c0604c4397acc6bfb8be11b9a5ee251fd21d7",
          "paint_distribution": "a6756247361ddf08144c094b4ccfe305648b0d6e5e238b5fbb71fd4903c1e491",
          "paint_entropy": "8ada777ce1ad7a1d8a7c30771eb9628e2d2045d3818b4b64776ae831ffffe226",
          "rule_tokens": "76c726a905096927e646a7bfbd373cff74c9474b0f18a1780149704ce4b32b37",
          "support_evidence_bank": "d331d4326942dce13f59512e642503a6bd63ef0aad9964516380b95f0a863f43",
          "support_evidence_mask": "cd4d8ea1452b42dc26916b99d4a1a52944a84867dfecb64e4a2184cfe2e69fd8",
          "width_logits": "d613ccee286940215be3dc9bc06a9c5d9c68b3d22fdd26a1d7f0fea41e2462ba"
        },
        "query_pair_index": 1
      },
      {
        "audit_index": 1,
        "changed_task_id": "c074846d_noid_1",
        "decoded_digest": {
          "decoded_color_digest": "4adb02d45654b3e7b3d5aeba3a50eff0b9f4d39a2eb6b1521cac5de66b964065",
          "decoded_height_digest": "5182b2de4aed1e28de2e5eee9de93423c7c3e5ff7a70a52e1b794b992a437a5a",
          "decoded_width_digest": "5182b2de4aed1e28de2e5eee9de93423c7c3e5ff7a70a52e1b794b992a437a5a"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "09f274eb649f3d19c3291e14bc83575cd21057b5a2ccb2b305a8c71cfc82cb6a",
          "query_inputs": "76fce419aebfb1d0439cea18383af56f8a53a84d981b607348e50e0814c1fd0c",
          "support_evidence_colors": "614da4e6b691d7ee86f687350d0424ca0b46ed15865efcdfbb9ee6d63f54f6ba",
          "support_evidence_cols": "bcff034e38175744518d6b3a41b68bc8ab5ba2b9adac2056ecccca5769de5e0a",
          "support_evidence_mask": "df9a11f70a0623e105519091538608a67797050ddb656383ab9c9aa6b1654154",
          "support_evidence_pair_indices": "ee1d3241c728e7041f9ed0061952097259644e152321db8b491bbeec8ad0c1fe",
          "support_evidence_roles": "069d0dec9489b62a5c0c855b1ece51758d7defde87d5245948ecc69f51f55189",
          "support_evidence_rows": "b96e6f40d0b649d990ba93825b13aec838d0c9a8ebe1a37ca4652d99e00111a9"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "c074846d",
        "output_digest": {
          "color_log_probs": "04170c1ed6c42f387503d3feaab4e3a42ad7ef7b4828b40b1d5c95aee9732e06",
          "color_logits": "04170c1ed6c42f387503d3feaab4e3a42ad7ef7b4828b40b1d5c95aee9732e06",
          "color_probabilities": "95d0bcd0f41850e9cdded1a04de1fc85243c2ff6865f54cee748f18842fec542",
          "copy_attention": "65ec0df2731915438492fd3c971e19e641fa2028d35e80a9722615eefdf32430",
          "copy_distribution": "e4bc6921f510abcf0c212cd4c7b9e90d586970038340f9fbcb899f4088c5a2ef",
          "copy_entropy": "65891b6c9b130b1cde84546d6c07f04eac75ce6065156647069be037544f1829",
          "copy_gate": "6ea6eb393606b4172e249c6bb6ddaf0ef4ce608004180039032d4a39f51f9972",
          "final_canvas": "89cf010d12a60e035e3cbf3831d808cc9da7fbbb75ac42d126adfad25e6bd18d",
          "final_rewrite_state": "704784f91000e494cc4488e9b7197f2325bdb7b5e1dc64fde1855bc94dc0773a",
          "height_logits": "ce997609211c6217fb1182e81c5a2142747e675a0734d4d1675844428d662e9e",
          "logits": "04170c1ed6c42f387503d3feaab4e3a42ad7ef7b4828b40b1d5c95aee9732e06",
          "paint_distribution": "359a224f4119f5db7037f009ac21d512fd2071bccab125b59f02b749386b9d9c",
          "paint_entropy": "e46f24f4f08dfa6e7e59c1236055d2c017b822ce90170f79436596186f59aa7e",
          "rule_tokens": "f16b254b4db092983e152f3b4def9051333cf1d19893f5a52fea5446d4f408d9",
          "support_evidence_bank": "12edc748422748d963f8f5cd02dda3f5e849a01942cfef5d054658555e497ecd",
          "support_evidence_mask": "df9a11f70a0623e105519091538608a67797050ddb656383ab9c9aa6b1654154",
          "width_logits": "485184e15f54fb9ec792c09b47ed3665a191a30248d77e154f94a08a1d86a739"
        },
        "query_pair_index": 3
      },
      {
        "audit_index": 2,
        "changed_task_id": "b1948b0a_noid_2",
        "decoded_digest": {
          "decoded_color_digest": "93fb028514d2f3658cc4cf7f2b57236fd5650f12a2c682a52593ce19ae8f2998",
          "decoded_height_digest": "f8c55961e5a9816150353765e9d9d613d7d9f9af438b8ada1e9309e995035684",
          "decoded_width_digest": "f8c55961e5a9816150353765e9d9d613d7d9f9af438b8ada1e9309e995035684"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "569b15e4923b4fda363e9f0f623484f7554b6fc6e21abe51ab11984ebdff981a",
          "query_inputs": "ef476d4421c58628c97f90a5ffd37d27e670bb3f36e0175aa48a2d05815dfeb3",
          "support_evidence_colors": "0f22ae9832c9274f8d399fc9b08412e73a956016c679e459b6a9c9e2cd304422",
          "support_evidence_cols": "1334810d33241aeb35b34df0eb470b37cdbe08d25a6e7f8533dcc94dee4a0b09",
          "support_evidence_mask": "da6b8ce495c4cbee529c3008293c0f1746c21e2cb3b9a19f494ffe3ed6145f7a",
          "support_evidence_pair_indices": "ed94779e8463a5bc7717e6c80d196db95101cd6f15b466967a9855290beff6f5",
          "support_evidence_roles": "64039bf6eddec30f34f45c0f5dee24efe6d6454f32de924a51ac06ac32c53bc6",
          "support_evidence_rows": "d53d24bb1e941a8768ca7089f22cc6e73855745903bfc407f5215c0c8805dcaa"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "b1948b0a",
        "output_digest": {
          "color_log_probs": "3bfacb7556a3a0230ec605b6febdc18d893167e0652aef0eb909bf29943a9485",
          "color_logits": "3bfacb7556a3a0230ec605b6febdc18d893167e0652aef0eb909bf29943a9485",
          "color_probabilities": "faea0ed1e103aa6307375362749df161f8f9999e9a03c41acddf5675cd5a9e59",
          "copy_attention": "5b1e48233fe464bd98b9f65cf58bec993c371c1b76aa29875dcb6f4d464b02e2",
          "copy_distribution": "0aa52e3fa809866cfd3096e55edbd3fa6b00678f80585750fa99e2664b1facc0",
          "copy_entropy": "2663b906ff1a3b899067d28bf888931cfbcd96ed23744e7f95b0975b386f1e2c",
          "copy_gate": "4b26343d0a0e6019eabb9ecde993417378ea91fcba2c85645513c2025f9c978f",
          "final_canvas": "e0bf28ae1ed3afb19bc49d261991e8482aea522203fb1fe8592845cf12499b7d",
          "final_rewrite_state": "8fb135267d8ace811fad2518125b615db059da06424be48963582283ea95e990",
          "height_logits": "89b3646bede1b720407ca67e6fe55ba95ab62e7752bf664715de2dc3d9ec76b7",
          "logits": "3bfacb7556a3a0230ec605b6febdc18d893167e0652aef0eb909bf29943a9485",
          "paint_distribution": "539907f65204d01caa0657e60f025eeae8b022a99123ae191a9dfce63a0bf24a",
          "paint_entropy": "e8a054a61af24018b270cfcbde8959c852e62b8d63cc8e8e92fdb3c3bc19e7f8",
          "rule_tokens": "cdb2cec68c4cf781c7a51c68357ee21d0d57d1002af805c85f1d190761f8104b",
          "support_evidence_bank": "acad2812ad7c3601870056c171a689d202810e82e7f72f12f02c054631032383",
          "support_evidence_mask": "da6b8ce495c4cbee529c3008293c0f1746c21e2cb3b9a19f494ffe3ed6145f7a",
          "width_logits": "af67df4e8f3ecbb2568a01980100681e0e21a736c3f0cc50bef7bbdbd1801cdd"
        },
        "query_pair_index": 2
      },
      {
        "audit_index": 3,
        "changed_task_id": "ac605cbb_noid_3",
        "decoded_digest": {
          "decoded_color_digest": "03929a8c8d5cb3d25ea0d85e6708c1f29c3de20068a5da6a4d88c1b102e9d2ff",
          "decoded_height_digest": "02c62f8b9c3e921393d1bb7dc34a5a3d04a3aa3a2044b44ed50fc2d42c36576d",
          "decoded_width_digest": "02c62f8b9c3e921393d1bb7dc34a5a3d04a3aa3a2044b44ed50fc2d42c36576d"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "9eff6bd11d686fa80740959fc70c3f4c34dc90f4278edd82f9059d4c071f8b1f",
          "query_inputs": "f43d4beff32c364d998fadedc4450d2094e9206a9a8eb5d0d0a01a213caa4f71",
          "support_evidence_colors": "11853f2a393fc16109a31200e9ad0bbd7cd8181963a4cc4f12aeb54fd39aae98",
          "support_evidence_cols": "187f8b24bdbc201c9bd38b7fe82919bdd9db31a507509042299cfb8afce05054",
          "support_evidence_mask": "d424930f1f4a71e85860d9ef74b269893f00f83ec3e023192d8db112df7b35bb",
          "support_evidence_pair_indices": "4a7138b2bb219608789b841a7ed09302e93bff6e787a4773c6b10c1e47de6b72",
          "support_evidence_roles": "ebbc0774cde70806a6f0a6c0f845afed3bf9766b6cd7c8c3c3d72f6a2024b7d5",
          "support_evidence_rows": "6ac55e970e8197d876ef75fc71df1fb1094da43db0a7565395eeb6dfe5201be4"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "ac605cbb",
        "output_digest": {
          "color_log_probs": "cd591329896fa427a912069a00ccdd8036e8e0e2e3e12268b82599046c83d09c",
          "color_logits": "cd591329896fa427a912069a00ccdd8036e8e0e2e3e12268b82599046c83d09c",
          "color_probabilities": "899a46f2337257c2b4389eb0c5a9be544fd09b5f5e2c01e24b961c395f8cc828",
          "copy_attention": "52437cd510c14468aeb6cc14efd4d9474ba2872f682f04b91efc76a8d97c408e",
          "copy_distribution": "2aec92ced39efc082ba15caf07ef32d03c28ea4e95b84e0b3073d1bbea738c86",
          "copy_entropy": "029e8e9e501da82a60b8fd6d9483701a94983831e2aef66d5b887feb2794edc4",
          "copy_gate": "1c37c25f8bcfc95bc496f43db675bff79d1b182b1531d99841ba533803519f94",
          "final_canvas": "de83c4e3e11fa3c4a7f6e38bdec4c51590925e0c9beab18811c64302fdba7331",
          "final_rewrite_state": "ac8816a9a2496d84236987c4107d5e25a901a030fb435081c4e611709f7cfea0",
          "height_logits": "720f29e9bebf6d03c9be1c886c33a01bdee042cb3daf8ab78e49831793303d00",
          "logits": "cd591329896fa427a912069a00ccdd8036e8e0e2e3e12268b82599046c83d09c",
          "paint_distribution": "6b46ef2fafa40d746bef1f7832db88c302f52a61b192eb0284b3770f0b05268d",
          "paint_entropy": "d3723799b43b20c9eee3f09a13817a604f75fcd0ef34b24691ca83a532f8944e",
          "rule_tokens": "1f9e36da591c30456fd06eed2dc5f7f9a2154aa5b22a17514871dfd2f1702878",
          "support_evidence_bank": "7d5c6b313da75461946a120f3a5e5a0bdf4ff9461a5c3c1fc65b1d2176c441e5",
          "support_evidence_mask": "d424930f1f4a71e85860d9ef74b269893f00f83ec3e023192d8db112df7b35bb",
          "width_logits": "06f6fdbff7b4305dcb50c522c2e2aaea33a651c5d648d17117ddf391c4a8a530"
        },
        "query_pair_index": 5
      },
      {
        "audit_index": 4,
        "changed_task_id": "84db8fc4_noid_4",
        "decoded_digest": {
          "decoded_color_digest": "08448d8ee745934a5c545dad909274f63ade7e6a06ac7dcd1108de0cb6c0902f",
          "decoded_height_digest": "a7a991a2c6f6c88328c36dea39a728e1eafb554d42cff293d0f4cde43b914582",
          "decoded_width_digest": "a7a991a2c6f6c88328c36dea39a728e1eafb554d42cff293d0f4cde43b914582"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "154ddba6b569945d9e9ad00eb00d2ec24a5f31f063787feb4f2eed30a6831c95",
          "query_inputs": "0161b56cfc6eef04f2616648d275edad45746f89844e8ab9ad805e6853b70b53",
          "support_evidence_colors": "8158597b516c888f28010d3424e0bb05be2583b0443d778498d01e6ede8e7805",
          "support_evidence_cols": "7e97de7bf19b10bd58606e022a91d6d3733e9f8be88ecb31c627e5128ed3d56d",
          "support_evidence_mask": "f588eb75d1cc3070395568485a87b93902885e131d2b7928be668309522a8293",
          "support_evidence_pair_indices": "b751f92fdb48846b6104ef878c615281fb664499bc7a32c874bfb8f20c2733a3",
          "support_evidence_roles": "a001d145bdd49646a438a121a5ea48360799ab8ffa531de391b97c4fa9d70e6a",
          "support_evidence_rows": "4492847a289d0dd8ab080a2264318128e4c49ae289884e9a0a8d3fa1ce9ef6ab"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "84db8fc4",
        "output_digest": {
          "color_log_probs": "273b6a18333a8f35c6419e77c92a5e33a1f1519a15c033ee22f6b11c384611c4",
          "color_logits": "273b6a18333a8f35c6419e77c92a5e33a1f1519a15c033ee22f6b11c384611c4",
          "color_probabilities": "2d5d8c44d4b3ee6fa3f475cbabbbb64a5d896d5ad1b46d0b68177a2ec1faa5a9",
          "copy_attention": "00903d99c7f78d3af6d8dcb87ee427ce02e4ae11bdcdd466a6fdac21463f432a",
          "copy_distribution": "68418449770833637fc464394998728736402dffc9539e210180730b9600c52a",
          "copy_entropy": "c9451ea1658744872bf8e740b93e2d0339ca3fea9339af3469ca886974f9e360",
          "copy_gate": "16b2f2c1d13b409db57b212bfd36c9c4f16774537a2ab8671b9ddd15f12dfaad",
          "final_canvas": "5373a8d536ebca7c0cf38d4c17ed919d2cba8d780e19a1014a833cedcd4ca73b",
          "final_rewrite_state": "5e3424954f58a4be5fa33372f954213d926339ef704eb0711cd60c150f61f344",
          "height_logits": "160819e38580c1810161d1953473a6d5435cb834a8ca6e4d42d811f8bce085d3",
          "logits": "273b6a18333a8f35c6419e77c92a5e33a1f1519a15c033ee22f6b11c384611c4",
          "paint_distribution": "bc37136adbdc4ea0dbb742efcae117dfd616c1746479a6244491e5b83b413ff2",
          "paint_entropy": "86cb7a016eaa05e53ada1c7edf7137f1fff1b40749b904ae8c5421bf64660e2b",
          "rule_tokens": "5cf6b697693eb5f46e71fd6be01afe720a2937805f8677c80eb00283b60a8d10",
          "support_evidence_bank": "3e86e46ba421a25686b77247f8800606660e21a096847bb2c18e343635f53c0c",
          "support_evidence_mask": "f588eb75d1cc3070395568485a87b93902885e131d2b7928be668309522a8293",
          "width_logits": "1bea9d4165ceced139e404f65b67e70023853a8035616bca535f0e46f2a762a9"
        },
        "query_pair_index": 3
      },
      {
        "audit_index": 5,
        "changed_task_id": "e9b4f6fc_noid_5",
        "decoded_digest": {
          "decoded_color_digest": "1ae1bbb6f0ceb480a417f42465ddd29d77121222d302b181fdca5dc386070bc6",
          "decoded_height_digest": "706fdb66473e5281bd103f163f4c76f274c96c405a3b4563d4a330245b5e0105",
          "decoded_width_digest": "f8c55961e5a9816150353765e9d9d613d7d9f9af438b8ada1e9309e995035684"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "39037d837c18e02885443f592be9e38fa86d886423c995f6e7178106b720e844",
          "query_inputs": "1b1d5fb7eb28627d05fd7ce6dadd098dd03a013a4076b0fe33ca369ce4a01260",
          "support_evidence_colors": "0dd9604103b18679f6210a6ee400c2e7c17a5f0e559c6bf32d35c2e9e6d6176c",
          "support_evidence_cols": "acb353ed6f6aae2ad5cbc9cd1db92fd23258c467f987f5d922171d2d8e5faa45",
          "support_evidence_mask": "8ace52b674e08318a50abf570ee31f21975c4d3ed0114e6061e29bccd9f201cd",
          "support_evidence_pair_indices": "38ece61e5972e1d30eb64db2ef2167e154a90a5f264b4cd88adebd4178da99d7",
          "support_evidence_roles": "76014c8bcaea1c06b7fac973f40bc6275313cd4ec846757189bff5d15f6ce7fd",
          "support_evidence_rows": "2ba8ee3ab20c2bcf89f2af997edb4f3772dd956bf6cfb171ac2e402d4c5cb3fb"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "e9b4f6fc",
        "output_digest": {
          "color_log_probs": "21ad5d5f9d72c64ee39c6a721adb96028779f0bbcc684cdd1b4ba703ed74d0ce",
          "color_logits": "21ad5d5f9d72c64ee39c6a721adb96028779f0bbcc684cdd1b4ba703ed74d0ce",
          "color_probabilities": "033214abf6169021318c5a5ed6d132b4d99334f5b2a75d6a67039db9461cbafe",
          "copy_attention": "9f76978b33d1aa8cb24f7fda5b9eabbff7833b57af30163b14230f4f67bfb05a",
          "copy_distribution": "c53523975cbe2e107b29559531bac3b859492e67cfed9893eabf7b19d2c21e8c",
          "copy_entropy": "773719fb1c8b6736e808c21ad7df93fa92c7ee3eb9f1746d5e59b8f59f306f23",
          "copy_gate": "8b78899c13c5a0db5f5e3f6976baeaa314758c9f5ef60177c72b3e11dd39810a",
          "final_canvas": "cf285aaf3914025a3713f895bd50335b45c954cc6d85cf9e3620ac716d59298e",
          "final_rewrite_state": "23d3e553ddc6e20f0cc1cc758bcb71448f6b255e9c5288ddb4f1006634ca8089",
          "height_logits": "0f475a604330e4b88332f815095bfd6a5c2c74ad542be3ecaeef1454d5f6d2c6",
          "logits": "21ad5d5f9d72c64ee39c6a721adb96028779f0bbcc684cdd1b4ba703ed74d0ce",
          "paint_distribution": "b7854c7c2100bc6018033a8dac18fabb2e2c819ef517781e1cf7419f8e7e79fe",
          "paint_entropy": "1c16d1a00d2b88497c943fa981dd85e38ecf8a057c050e4cbdd0fd4146768efe",
          "rule_tokens": "7556cf9c39645a34e0d516b078eb190ebc6d7f2f964f7911c78793f5942f42ec",
          "support_evidence_bank": "d156669177f1d0495e2acd9af756c8505ae73bc9e2d911e384c3eba34ddc66d2",
          "support_evidence_mask": "8ace52b674e08318a50abf570ee31f21975c4d3ed0114e6061e29bccd9f201cd",
          "width_logits": "9f308b6d9cd98b950ae03b29f07bf670e127309b87f06fb7990f75b5106d3c63"
        },
        "query_pair_index": 0
      },
      {
        "audit_index": 6,
        "changed_task_id": "2f767503_noid_6",
        "decoded_digest": {
          "decoded_color_digest": "3c92556ad124bbd3a19526a4644f0c8a3128dfb27e15edd91ed773ce93fb7e5b",
          "decoded_height_digest": "dbe1fcbad67227f1c774afec9edfddae94ea4f7af2891f0366f4a408521e5b4f",
          "decoded_width_digest": "47ffd52e47bef6317339f5ff5b06f67df9ffa1bcf4d331e36cb3d6009443f958"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "11f74078f3122a339b330d12cfa88103a9ab16a747d1f1a95274bfceb292277e",
          "query_inputs": "4b653dd1572aba36841f7b0ffc6f7176a7239e13344725fdb3107b8b021471ff",
          "support_evidence_colors": "5917f672d6fcb7899886b410c60befd187a8af8ccc41f0100c2b2a25252e2c01",
          "support_evidence_cols": "a9eebec7a1f3a25c23fba1c843fd0c79ec8677ed07898e63bea859bb86056887",
          "support_evidence_mask": "0e2d93817d39a3bc783056e70df68ef4609febedfbaafbb4713a09ee2d4f858b",
          "support_evidence_pair_indices": "3738975ddf3373252b3692ec67eaf60ce5a26c6cb22d539ce0ef7848de2be869",
          "support_evidence_roles": "11a80921e7dfc57d3c6dd805c5b5dacb8d87ebc758a898fc9dec76273780460c",
          "support_evidence_rows": "eb630a8133e98961e74c3933da772659636fe361113322b70c6fa7816f90bf6f"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "2f767503",
        "output_digest": {
          "color_log_probs": "d8e54fa89109cac862b249df639fcfa6d068732ef7010462701d8e5156fd9242",
          "color_logits": "d8e54fa89109cac862b249df639fcfa6d068732ef7010462701d8e5156fd9242",
          "color_probabilities": "60d3733956cd152ef63546dc2d7f1b5b613788848ac7346365e47ff3d14b4a63",
          "copy_attention": "3b0e175e5731d8e3a32d02a52b77caaba9c2a13f5efdfe48c813679304f56c11",
          "copy_distribution": "52390a41ecb1f3fb2d8758ea0fe65bdc714366bd72c31875797e5c392dc77e24",
          "copy_entropy": "a66881b529e27972c2e0343ae91537bb9520d56985641cfd7c933d9c0a8beca9",
          "copy_gate": "a3cc26d3aee5291420934fec93ad9890087d29f91ff77a64b3e94de3bdd282ab",
          "final_canvas": "79062cad0f24cae8ce0e8cb60ce220f18fca047f152914894c743901c8b1d20f",
          "final_rewrite_state": "cd0b9c41bb4535565785b901ab5e41aaec3baeb623a05e2059f52ad7cac975c2",
          "height_logits": "5dcc531b8bab0787375f13641f7c14d61ef32929a13a8b58a49e6d80766fff5a",
          "logits": "d8e54fa89109cac862b249df639fcfa6d068732ef7010462701d8e5156fd9242",
          "paint_distribution": "d242b9907def447a809ab7b0cc37e960a69a707c49215c66e2c98f5acbcd7f62",
          "paint_entropy": "17f5dc245a98d3453cbf8b2623f3ebf13afe6ef6b0d6b268a73628618a6e768d",
          "rule_tokens": "23b9d720e95793ac4f685ef936cc2c4b5a316eda536448de63bad583f59251fc",
          "support_evidence_bank": "8a7b5e13f098dd74ff33027473b9ed44676e447fcbee288adc2068429db7a964",
          "support_evidence_mask": "0e2d93817d39a3bc783056e70df68ef4609febedfbaafbb4713a09ee2d4f858b",
          "width_logits": "3c2e5bf4456bd3712f3f873ea9e515238ec1e1939d75cf0cdb714de15479da21"
        },
        "query_pair_index": 1
      },
      {
        "audit_index": 7,
        "changed_task_id": "7b6016b9_noid_7",
        "decoded_digest": {
          "decoded_color_digest": "bea356ebd309ba82d9c6b8132af283e65d3a4dddbad83332aebd2faf37a450f7",
          "decoded_height_digest": "b512577c66cc1dd42fa59b05ddffd5c836c9172a310fbe1a4c8654347162b6e6",
          "decoded_width_digest": "edb42ac8365616ae788242305ba5dcff22efb27ced93e0f150026024f1f656fe"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "e6a4b2b2c8e532d7636222de06fbb9d8e2d0778037567f44dcee78bfca682cd7",
          "query_inputs": "bd16ebe08940f9b2544095e9d1b76bd48f67b8e1a189d207cf05abdae9b62692",
          "support_evidence_colors": "e4628e99350434095ee4457f75c5cfaf852dd865759389479419a459ecba4ea0",
          "support_evidence_cols": "6e3bb15bf99b0046d3a2cad4815385e76757e91ab7ad776b27fc5c2cd54a1f60",
          "support_evidence_mask": "19b8788857a410c60707c48bdc15cec23681d936654c092a08faa4ce4624ea95",
          "support_evidence_pair_indices": "6f8a667064c3f85d08b467c81f6e49ab77f2d13f83bd085895ec4266145c01ec",
          "support_evidence_roles": "6a07a854bfba65092fb7e4a75e94037917531297eb9ba72a7c988c3a2ac8ed1f",
          "support_evidence_rows": "52830df82eb84085e9a5cd3a77672fda9bc3ace6a572785c77fa673890fb7e58"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "7b6016b9",
        "output_digest": {
          "color_log_probs": "1f43ccd153b2fee0b737eaba033166be64427d09f032670b03e87cadfd45a13b",
          "color_logits": "1f43ccd153b2fee0b737eaba033166be64427d09f032670b03e87cadfd45a13b",
          "color_probabilities": "318c2f6ab482d840450ca07838a346ac1e0bb539af5924cfd835e4be4751013a",
          "copy_attention": "4df6aa1e2f6ad92b216babdf6bca8467ea5ca4f69928a5ade3568b350c577e49",
          "copy_distribution": "aaec45f0e3e8ba9ddc401e9125258bed46e3bde132c85457f7bf0249ce1b6754",
          "copy_entropy": "ddf6df39c0be91dc3d4937c0573655d84c8e6eb732e08f1ef3d417ba954130ec",
          "copy_gate": "49ca7802e23e497893a775389cd9fb1452c7edab8d56accaa620704987c5e831",
          "final_canvas": "f1025ee8558740c7730342815e3fcab6bf4186772cb02488e1bafa8c86a087c2",
          "final_rewrite_state": "9e644873b93e33a136fc5570316bd6ce3890e01b8257d1ec637297fc5dc06d4b",
          "height_logits": "641589884909204265273af28649cf8c9d5d35e631a6b276ae36a43777f1da6a",
          "logits": "1f43ccd153b2fee0b737eaba033166be64427d09f032670b03e87cadfd45a13b",
          "paint_distribution": "9fe0c4c72d6af4ce5e0f92fa529f36b0001bb4ddd427f9141c08b0498c811596",
          "paint_entropy": "02b4c6567b561a48be512307449f3061a170133505781a3e02f6b6cdaff4ef58",
          "rule_tokens": "8c2f0737a5d446f9e24c8667a4bb8f737bb4497b2d87ba208af03d449b4ba2fd",
          "support_evidence_bank": "32ffeff2629e6d6cf07e58e50d52d227d9831d3be2a1632bd9a5d7c03ef8e2cc",
          "support_evidence_mask": "19b8788857a410c60707c48bdc15cec23681d936654c092a08faa4ce4624ea95",
          "width_logits": "e1d5e05663a97947eb70afcc6e4c73fd3b3ac7d3e918183c686b844e3d4d1cd9"
        },
        "query_pair_index": 1
      },
      {
        "audit_index": 8,
        "changed_task_id": "22425bda_noid_8",
        "decoded_digest": {
          "decoded_color_digest": "51c8386c3146188208b11e39158f6b9e74da56a8e6276faebec07c6597375de8",
          "decoded_height_digest": "b363af63b372f1e48ec48fcf885b668ead57668489f7169ff790b8f9ffe92e90",
          "decoded_width_digest": "b363af63b372f1e48ec48fcf885b668ead57668489f7169ff790b8f9ffe92e90"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "f39904fda54217529ea7a4f2ac3c8c709c389d8359a37d7cdfad4bac118c7635",
          "query_inputs": "f92d1d7420ef27fbde3fda6043cc7a9e2b385b74b942e0ddb862cadee32115c0",
          "support_evidence_colors": "b2ccea0e8c3c443577315698073e89dd8589e63bbfd1dc3a36e4e215a157bc61",
          "support_evidence_cols": "61f1b108730bd3666b413554a31bb1c3ea323164ae5fea558dd3e7422847b446",
          "support_evidence_mask": "668004a0f719c651ceab1ae7d1d47e488859e5e98172fa9e72588cb137d4c8e4",
          "support_evidence_pair_indices": "c131f868bf57dda38869803a599a071812725ae776f6a9198f09889ffe980a16",
          "support_evidence_roles": "56e0207dde7cf08d1200cf51aaf71cf4c57ec1bee025e2b03e78969b3495cc9f",
          "support_evidence_rows": "1e1c9e143d4cdd411c4bd30ed87e8176666f4bee4c3f6fc7902c55d716a199b1"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "22425bda",
        "output_digest": {
          "color_log_probs": "2c3a036c0bb0216cb6dd19450ef5fcf71c78e25b3103ddd3641a22133d75282a",
          "color_logits": "2c3a036c0bb0216cb6dd19450ef5fcf71c78e25b3103ddd3641a22133d75282a",
          "color_probabilities": "ee5472e11bf1ed2b499784a54a9ef716606a1cdc7c710261989deb994a4c423d",
          "copy_attention": "7116325bb6b69066164630016fd8a4b410a74f5380827bc814951dbddafcc08c",
          "copy_distribution": "8c6587dfb2e18609704e889de05101b7d8613d779d4a93d2ba5786e519e053d8",
          "copy_entropy": "93936e3e57014d00a8de82cb1bd654cda46a86245b0731e9d2148e8eb7a5103c",
          "copy_gate": "9c51df0569f11f37ce94925e860ad37041c2f5007635334e4e48175ceb9d24b1",
          "final_canvas": "cd966d1a80ac42d8ebaf2ec845b49a2e6d9e5fbde3d1ba9a06de5222a6148caf",
          "final_rewrite_state": "4e98d584c66b2c2137d6864206f8e26228cd654aec4cffd21d70e88fe2ed98e2",
          "height_logits": "d81d5380de4be712f9393304a7a234709a4154a2351c4df88d87003f04d9a49d",
          "logits": "2c3a036c0bb0216cb6dd19450ef5fcf71c78e25b3103ddd3641a22133d75282a",
          "paint_distribution": "c47ec904956227e485bb83236472fc5ec6b3accb80789382d0ce522137f9b574",
          "paint_entropy": "b3a9d821048b2958c4cab2044a00140d44273a4d27c7ffff0664958ed2baff51",
          "rule_tokens": "ac4972f8e85d5a5e8c9f0a8d3d88968c723fa3e51c81191e9371544e9f2802ee",
          "support_evidence_bank": "80f964350ffcb98359910d58005ca5d18694de34a636ac644d5fb3c8fe8a727a",
          "support_evidence_mask": "668004a0f719c651ceab1ae7d1d47e488859e5e98172fa9e72588cb137d4c8e4",
          "width_logits": "4a01e63ea08ee6231af7b04d03d25093e2084dc039c7ffd56c0c1e2f915ad931"
        },
        "query_pair_index": 3
      },
      {
        "audit_index": 9,
        "changed_task_id": "d492a647_noid_9",
        "decoded_digest": {
          "decoded_color_digest": "dd1fd4073ef8b0dc670b60dd815f27a3e06ead32c2763f1e2afea1f7e71e983c",
          "decoded_height_digest": "794dfb0680bfbcaeb43c20ca1b2b463c5ca92e96e992dc501ee4ff9d9a82c258",
          "decoded_width_digest": "794dfb0680bfbcaeb43c20ca1b2b463c5ca92e96e992dc501ee4ff9d9a82c258"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "39037d837c18e02885443f592be9e38fa86d886423c995f6e7178106b720e844",
          "query_inputs": "8f5211560851d54a143cbbf26e57d862e07f9d5bf18b742f9b7b1bb28e77417d",
          "support_evidence_colors": "76eebae16bf70ac7db8ebc4a7fcd3cdee43278071d6249176964b99ef913285a",
          "support_evidence_cols": "54260f84590a2453b3816ce0d9730bc14fc1653704a9748fca289e9586a09bb7",
          "support_evidence_mask": "97f391375c9704a9aafba9635da6927c4ebe65324e9bf21f78873c33848d22f0",
          "support_evidence_pair_indices": "deef942f31bc9185a83c600a3e00b5014526193356ee9f2754c643d952effd86",
          "support_evidence_roles": "232db5aabd7c588d538239781d3801616bb86eac543173a47f4989f5125c526b",
          "support_evidence_rows": "e4a001a6993e10890b02daf90b719519f9d6dc82033bd6164f65ddfe6e5bc052"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "d492a647",
        "output_digest": {
          "color_log_probs": "8d25b4927dd047921146787d0665accbe0177db3288ef5e7c58959307291a272",
          "color_logits": "8d25b4927dd047921146787d0665accbe0177db3288ef5e7c58959307291a272",
          "color_probabilities": "b45894b7e599da9eb3825c5f51f62f9b28e64a6a14e493056b5c06c2c73a2d0a",
          "copy_attention": "3dd25af46ea478486f1e239915e27b8e5dde4411b1b5f708ce4d8a063f7da3d8",
          "copy_distribution": "ea1c1ea87e7ee60869f00d9175bae6d0ffefb7ab77283c019e10bac3f166806d",
          "copy_entropy": "877073d0c211bd9aca2eb6db849541b866eb87823b725e2eba39c7ddcee21b63",
          "copy_gate": "d298a54c13a7003c5da4b6221960f4642a5e6039287d7c9068d72b5c8d336f65",
          "final_canvas": "39dce2dc99e5d3fa60018abef583dc375ca11eea8dc137bfefd3ca4eb29b5f00",
          "final_rewrite_state": "ab7b643571ef91787699cd3def3541819e368ac6193bb6f5f4d3d6868795053a",
          "height_logits": "06440411f24548242113f0e14188644b362f432fea85eeac718923b37de340fe",
          "logits": "8d25b4927dd047921146787d0665accbe0177db3288ef5e7c58959307291a272",
          "paint_distribution": "0cca36ff4311fece46de48489d2bb4f7a1a76de5a3515b102909ba083fac7d77",
          "paint_entropy": "7e2ebcde1a11d514918352a2f918e3df0cf56dcaf8960570f53333a31a02c337",
          "rule_tokens": "78510fe889f6f3b6ac2f311b2058ea75414b91911fc3ef526b163e0d2021378f",
          "support_evidence_bank": "7cbfff0260121bc08ac9cfa9d35d3fb9021f720a5a10f7d3bbb5de8ba5fb1bca",
          "support_evidence_mask": "97f391375c9704a9aafba9635da6927c4ebe65324e9bf21f78873c33848d22f0",
          "width_logits": "a2bbc8200f6e24eddb3aa67ba6ed0f1bef07033659594ec2d9171a0b2679b936"
        },
        "query_pair_index": 0
      },
      {
        "audit_index": 10,
        "changed_task_id": "2072aba6_noid_10",
        "decoded_digest": {
          "decoded_color_digest": "5650d03cdc72f34ca93d69a5a49971fdb995325f32a0fd7b8eff09a731e40808",
          "decoded_height_digest": "f8c55961e5a9816150353765e9d9d613d7d9f9af438b8ada1e9309e995035684",
          "decoded_width_digest": "f8c55961e5a9816150353765e9d9d613d7d9f9af438b8ada1e9309e995035684"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "bf2b6c0e3710d87c89f40274294ab8149f10ddcf69f028f095927134e1d0d00a",
          "query_inputs": "87045d528581db8ace95db4602f6db4c6a1891fda398dbbfbf9ab9d3091d75dd",
          "support_evidence_colors": "18256be486100c43c92bb50fdaf74a6705c2dd63e7df6f23759e99832a99223f",
          "support_evidence_cols": "1a707b83a7ea391ec8f928d20dfef93eb20d98bd9fac1e91b10093843390cf01",
          "support_evidence_mask": "dbb1f6ee6766876850cef4869a7695acbb22804949e9428d8720dee4fb41c4e3",
          "support_evidence_pair_indices": "b972abf77b57ef55128922b8f63a1ffc6fdb879d3bbbe975b9a7dc5a45e5e745",
          "support_evidence_roles": "5249c5877a0f020826eb85d7cc371b369bfeefdda0a37525a49f962619fe755d",
          "support_evidence_rows": "2bdc5fad7c9fada10b71868958f967af491fc93fb9f9b50dc8cc4b1b164b2bfa"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "2072aba6",
        "output_digest": {
          "color_log_probs": "25831b904173fe64ecabca8dd8d37c471b28b433e5e0b0119158abaec4dbba95",
          "color_logits": "25831b904173fe64ecabca8dd8d37c471b28b433e5e0b0119158abaec4dbba95",
          "color_probabilities": "656b67b0ebf86f27b8e15ddec910669ee7730b449ba880836e0099c324cc6fbb",
          "copy_attention": "7e3fd46352d1a39f67423aa412f6fffb3e03f1c629949f8e7f3a078c334f6016",
          "copy_distribution": "e868e6b616c6b04b43c1d9b331db647f2f5cc574942e6db76508cd4ffa6933af",
          "copy_entropy": "e6cb94e49b0a759dd2b56e0de5164047541a51a9f408bd74d37956333a3cc93a",
          "copy_gate": "c9aa7e9b7e884f16822182d71661e8e4ac71188736225e227fb9a9eab0dfb2c5",
          "final_canvas": "502e75c090a662597b56f3638178db0cfeb58c64410b76ac2f2562b3d075bd01",
          "final_rewrite_state": "0feb330b37a1a766b66c0be28402fcf6d990ad9786e4ba7101111bdd90c79ffc",
          "height_logits": "bd00897a5c73ba40d3b5bd1a4d9855ae0d44bce42a507ba279062c82bcd42b48",
          "logits": "25831b904173fe64ecabca8dd8d37c471b28b433e5e0b0119158abaec4dbba95",
          "paint_distribution": "6e1a1289e4c3480b13094602602575e72cf56f08eca12126d6de5d56b7ab1140",
          "paint_entropy": "6db2767261be09e86e07721eee18bfc4b770ca4b89b213c75f1f2891d7163512",
          "rule_tokens": "cf1602b3b3da965dc041707d66857201123ef82a821bcbe9b5fc4cd9cee21971",
          "support_evidence_bank": "d435d2cae325317be5c7660332f317e07b6b4c991584065ce29d5282cd1cf914",
          "support_evidence_mask": "dbb1f6ee6766876850cef4869a7695acbb22804949e9428d8720dee4fb41c4e3",
          "width_logits": "adcf661da3d791413d213c76748ee78dca0b35cb9c2f3f799ade37be46d08b60"
        },
        "query_pair_index": 0
      },
      {
        "audit_index": 11,
        "changed_task_id": "963e52fc_noid_11",
        "decoded_digest": {
          "decoded_color_digest": "7d07213cd2246d7a542e4726e10ffefa216f1e0b76aa4b6aea71ddcfec593437",
          "decoded_height_digest": "bdab27466b0f872c7536642ec036002aa6e247555933d2006b9363d4cba7a0f0",
          "decoded_width_digest": "f8c55961e5a9816150353765e9d9d613d7d9f9af438b8ada1e9309e995035684"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "afda47fd35100667db45dded86ffdf7b782b44aea0dd0de464512baed8a5859f",
          "query_inputs": "6fec2cbd790918c862f1df9cac4e08499b430692afcce5a95cae0c2ed349dd18",
          "support_evidence_colors": "045aac30959b33b9ee36cf81571303413171dd62613ae320922db51e393eb130",
          "support_evidence_cols": "c3c5c4235ceede083f4fb8486f646aaefb0ae762639c097513b2c6015c706bcc",
          "support_evidence_mask": "c436df0fa374afa6bf90f1b6e811cd54e8ed41e416ae3028684e8f6d6ffd2557",
          "support_evidence_pair_indices": "9d6c6bb655366d7e329ab48541a02d2969b1ee9daedb66545e377635f5f79193",
          "support_evidence_roles": "23ce870e794824c9a98404625d2463d14626fd5b0b03396e6796854e831dcd5e",
          "support_evidence_rows": "4c2bc58387d681b433926d9b3ef24add48e317ea0cf87c3a490baefe67e99adc"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "963e52fc",
        "output_digest": {
          "color_log_probs": "6067453e75e01085dc072bd999537e9b9aa429a940de7cd6ac618e9663cd366a",
          "color_logits": "6067453e75e01085dc072bd999537e9b9aa429a940de7cd6ac618e9663cd366a",
          "color_probabilities": "f65f73f99c9c5087f06c938a7ee98ce0da0e04c263efc74460351ab29267b545",
          "copy_attention": "38a40de1d34d358e12704c996b7b02aad4175259e1a5d888dc7000db8c072bdf",
          "copy_distribution": "a1b850f7d4f582418194c4ccdea16c3a39bc23e2a93cd865d75a3e4dc7731ceb",
          "copy_entropy": "9454fc6c01f0249e8fb32b3f4345adc8db7f7560ad136e550fa951d01b37bf14",
          "copy_gate": "d14cec4241ad0ec808c1fe20cc0dde96001aec0e871a9c01b3d76a424c6b9a8e",
          "final_canvas": "b47a2362d61501879384614edba2adbfe8c11a32ed049266f36deea9eda890da",
          "final_rewrite_state": "60fb495df597dcb5bfc0a9c7be9f295c66b61b58254d0caa825349b9835c6647",
          "height_logits": "7597ea8f49d7f63b88008e688c7582153ce2377298df6bb40496dd49b2fece8b",
          "logits": "6067453e75e01085dc072bd999537e9b9aa429a940de7cd6ac618e9663cd366a",
          "paint_distribution": "543786be1b9f1fe06afa6645cdd1e2e65f9eebf7386fedfb6cf0ed6dd00bd93a",
          "paint_entropy": "03b3f1afa4043984092add653c5e7c1b7f32e93e338c6acb6859d547094ea583",
          "rule_tokens": "40968ee7f769a22bd0edeedd15a22e76d69cbc99751693b9bdbedc5340fe9d11",
          "support_evidence_bank": "8de8f3576104cf97b3bddae13a8ec3cd58bba5115ae1bc743b961c7e4d3027a4",
          "support_evidence_mask": "c436df0fa374afa6bf90f1b6e811cd54e8ed41e416ae3028684e8f6d6ffd2557",
          "width_logits": "21e712ef8ed08311cfde460c2a0eb1083ff22d9eab152330985fe6109d728ff6"
        },
        "query_pair_index": 1
      },
      {
        "audit_index": 12,
        "changed_task_id": "7df24a62_noid_12",
        "decoded_digest": {
          "decoded_color_digest": "5ad202a91f1b00bad1a2c7df66a0363c4efe67a240ebf73f6c4c852f015d5466",
          "decoded_height_digest": "b512577c66cc1dd42fa59b05ddffd5c836c9172a310fbe1a4c8654347162b6e6",
          "decoded_width_digest": "b512577c66cc1dd42fa59b05ddffd5c836c9172a310fbe1a4c8654347162b6e6"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "e6a4b2b2c8e532d7636222de06fbb9d8e2d0778037567f44dcee78bfca682cd7",
          "query_inputs": "4e20e7866f25bf3131b72d6e1573918ae8236def3232cd26e2f17abb9aaf2970",
          "support_evidence_colors": "be6756a19a4e8ea9783ccce36210cba8423e86a3839795ddff9eda959de1248f",
          "support_evidence_cols": "4667d8c31d0485e95dae21a46e2487bffe04a29e4ea025d7c091b0e253db384f",
          "support_evidence_mask": "67bc574ed00cf056d85bcd8466eb7740a626f39e7e47613d0c381de81e6ff58d",
          "support_evidence_pair_indices": "83db3882573488e2e6c07aa93ca673e097a1b61fc70e4134d265d38e026c832a",
          "support_evidence_roles": "6a32ab032cf107e46304a20602109f4e4fb59c8dcf7f8b81ca5445cb951ab2d9",
          "support_evidence_rows": "42cd78eafadd693fabda449ba32bb93d8d8abb9273bfee015e1c9027f71eee11"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "7df24a62",
        "output_digest": {
          "color_log_probs": "ae3e3b1b8dfab532aaa27f47d2669c93bc82f8aaed959f6cb78ef8d441ad5cab",
          "color_logits": "ae3e3b1b8dfab532aaa27f47d2669c93bc82f8aaed959f6cb78ef8d441ad5cab",
          "color_probabilities": "e7da484404e8da348be7cfe0feb9828c50d0666b0912dc58d527b5d61e431a18",
          "copy_attention": "e07f4504c691cb8ece6f690d8982f69e46be507cf8f6a24fa81de430b9a2ce59",
          "copy_distribution": "77cfcf7f1acb9699c3816d334e9771f7d72eb00c152cb6bf2a39b9990a9ffe88",
          "copy_entropy": "26ac50fed776ca59f0b3df127d66cf27bf70f5bccb918662273d87e881aa5578",
          "copy_gate": "b186085059b21e519c1cfd22c53f71245ad0ecd2b0c34e98cfcf04a1ebc92fe1",
          "final_canvas": "46f4a2d696273bde91ab0df709c6c371256b6560c5377a32a932b426d8caa644",
          "final_rewrite_state": "8bcc3774319c8c8cb070946f9894dc38dfb29cfd182a7d6286832bbf8ee775a7",
          "height_logits": "59c613cbf9112129b3577fd8dbb8e74d0fcf0e340a70cfda1f03022cd31e2ed4",
          "logits": "ae3e3b1b8dfab532aaa27f47d2669c93bc82f8aaed959f6cb78ef8d441ad5cab",
          "paint_distribution": "315408d27cab5dc9a11f6a87b1e0179082185b4e517fef5feb270e11ff569fdc",
          "paint_entropy": "b3d91490ee7e954f06605474ddcafdc1e940695b865094af17df5bf54c6f3421",
          "rule_tokens": "179d63d2e16b861d66fb9b7d71be7b7fe4b7d43bd91de3056b15cc12b2d8995f",
          "support_evidence_bank": "092345304ac77f42a1267373ea856d33c1ec8ddc6658737537425dcf17984a4a",
          "support_evidence_mask": "67bc574ed00cf056d85bcd8466eb7740a626f39e7e47613d0c381de81e6ff58d",
          "width_logits": "eeafbaef23cf606f35d3c0baadb91097379cee0f55308fdd68e7f5ddd9544cc1"
        },
        "query_pair_index": 3
      },
      {
        "audit_index": 13,
        "changed_task_id": "b94a9452_noid_13",
        "decoded_digest": {
          "decoded_color_digest": "fdc381fd24f99b9939c0992886d7433c5587da3605ccbe463c54988720398870",
          "decoded_height_digest": "bdab27466b0f872c7536642ec036002aa6e247555933d2006b9363d4cba7a0f0",
          "decoded_width_digest": "bdab27466b0f872c7536642ec036002aa6e247555933d2006b9363d4cba7a0f0"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "efaf76f9c4cb481e9a4f2626d00d02958cdbe93565d246feb5fc3d9352ff5c53",
          "query_inputs": "09c12188119dd89d4f9e8b6e6d226675bbc6687662cdb83c9a42748862c1c393",
          "support_evidence_colors": "6f3a1788baa03c39782cf3c3698dbdd0b0d8c482cb116c6c0a2e6758052816b3",
          "support_evidence_cols": "b3c6e00055864aff003275db58073b4ab3b8c87b84694a0eabcbe41fb79d0dbd",
          "support_evidence_mask": "9c2af342ddd7488e21cbd27acb4dc34c57b2c65c7e7589557d3a1f501cb5c253",
          "support_evidence_pair_indices": "8e0070cdcb8b69183d18af1d1e62750dda1f87aebd8cd5a30dbc37411233b90d",
          "support_evidence_roles": "b848d0ce45d51b22c13d7f8af443133c259516742af9d4fb4a75359d60e80933",
          "support_evidence_rows": "1931f4e0b34bca49439ed16ed0eb9a27bca9ac9c465371a1f0483c30f5593c06"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "b94a9452",
        "output_digest": {
          "color_log_probs": "85c38910795226028b73704e580da74bfed78e382a1e8484f8e4772403c10fb7",
          "color_logits": "85c38910795226028b73704e580da74bfed78e382a1e8484f8e4772403c10fb7",
          "color_probabilities": "8d1d56ef5a54c8cac8f93a324ea559270fa5fb13e6334744a508b95dfb65cf23",
          "copy_attention": "3d094d0ff564f84c8d78a937a27ff384cc605ab6d94b459237b6973a6795276d",
          "copy_distribution": "9253e6df97d328899c98b13e88ff0291e6f199aabca7d6fc7b2fd73e2fd3773f",
          "copy_entropy": "d42495cd372adfb95d770250367cad7cbe975938d324acc16f12084e28ad06a5",
          "copy_gate": "3cc89e0d64218c7f8d2942bb440a7effb5c9b20465125e1d55d3dd70c4c6749f",
          "final_canvas": "6f8c444b10d70c381decce1037b859f0634867359a936c168faa1bb00e32edf9",
          "final_rewrite_state": "28f5ac416b3b22c57ac5d7d8c4267d5280022430bc2fc91adc2cb5d329eedab9",
          "height_logits": "0c32ae3133c1e1c50e96f1af45229924dfb070e1a8d2e3dea931f710b69af200",
          "logits": "85c38910795226028b73704e580da74bfed78e382a1e8484f8e4772403c10fb7",
          "paint_distribution": "9038479b85f0129255e7b9622515dad8c31386918f0c6cc2e77f63a202c63e34",
          "paint_entropy": "c606bc0488ee66c9cb2544a8c749f09d1d3275fbd9be62edc3257c3a16ed320f",
          "rule_tokens": "0e4d25abd8efaae2a1df592374ac0207221d1ca193a2f2f1f2a4a7216990cd0c",
          "support_evidence_bank": "aa9d4895eb405bf8b649fb5a5e0dbf969829d640d2aa4c8d653542406f9481dc",
          "support_evidence_mask": "9c2af342ddd7488e21cbd27acb4dc34c57b2c65c7e7589557d3a1f501cb5c253",
          "width_logits": "c3358c98131c5426262e7b07e1e0765ecca46edddc7f6aa78a564c988480ee04"
        },
        "query_pair_index": 0
      },
      {
        "audit_index": 14,
        "changed_task_id": "aa18de87_noid_14",
        "decoded_digest": {
          "decoded_color_digest": "0f93a523691dd8e40607bbd244f74504f67ef5e5738e24a064de9f9ea9c40984",
          "decoded_height_digest": "706fdb66473e5281bd103f163f4c76f274c96c405a3b4563d4a330245b5e0105",
          "decoded_width_digest": "8be83a41c01e6af078aad73b63461fa5327fae57f8762450d5ba5803742ac0d8"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "bf9b7474d9d65889e591b13b7455f8ec7316d17887b4b087e223614a6bb08667",
          "query_inputs": "b45241bd9f0bf07b0370c2b02cc44bcf919ec5789959519726530264033de2e7",
          "support_evidence_colors": "1b867c2634695eae20703af6e4bc5a47c95acb29229fffed8ca2b0393d2e7101",
          "support_evidence_cols": "ade84d1a967946fafa5ed0c97e8ec71e8257b4222197cb40f25e04383f8f3c72",
          "support_evidence_mask": "33ad526d07f8f37a8f0ab48489af853f9a2d4105808f35c95d73654820b46da8",
          "support_evidence_pair_indices": "40f9401c35d25a3acb80aacd419a8180d07d2fa3db772f80389dfff0ba2f347c",
          "support_evidence_roles": "a304ae9858293df0b621d86f70c5da65c0c4b87ea2875106925a828159128b09",
          "support_evidence_rows": "61de478ad869b4aea9db87ab2da4961d27eeea00b92a8a52c6ca5e47812c55db"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "aa18de87",
        "output_digest": {
          "color_log_probs": "3debf945c5507af2b79ceaa00e886e13eb4757bbb6723afaaa78181786ac8945",
          "color_logits": "3debf945c5507af2b79ceaa00e886e13eb4757bbb6723afaaa78181786ac8945",
          "color_probabilities": "e5dc5379811fa8603b26ec4b97bac815a446de701108a45b5767eab5263e82d1",
          "copy_attention": "43b6c8cc394f7a1a8309873cfbe8e653d575bd7bd0e9861ca3752cd9f63cb25e",
          "copy_distribution": "d7bcff3a2ea3da8c42e442178914cf5d995fc3266eedc1639031519e8ec55d95",
          "copy_entropy": "46885d547ec2e53cdf1934e2c6b723ef20ac2fd47cec4f10bbfcad919abb50bf",
          "copy_gate": "ff5c1fffe6092e1aa8f421bd947129e1685f3f6ca11559d3a720b7119935d5f6",
          "final_canvas": "4324833f66cdcdf48726ada1b28304a0a9384a534954404f935d0f00348949b5",
          "final_rewrite_state": "d2ae3d76ca8c13b5b072650e8b62a5ad132bcaee2a228070f932380d2a71f81f",
          "height_logits": "efa7cc79fa7ac73de5071de888244cb4a63a8636cd748bd530b474ad1b1f6d05",
          "logits": "3debf945c5507af2b79ceaa00e886e13eb4757bbb6723afaaa78181786ac8945",
          "paint_distribution": "da185574d5567f768b526d20a273071393d79aac61572a5eb9121f889982878d",
          "paint_entropy": "433fa3f1098a737cf8583940a8ec0b35de7db9843058e3025b1801b422266397",
          "rule_tokens": "673b6f07483920d4a2042ccda0448863bcdcf3832890fc4a1e83541923daf14f",
          "support_evidence_bank": "22ab5b55c870a0648b439667a503c6233d413e89693210c89a6ad1664c2b7b47",
          "support_evidence_mask": "33ad526d07f8f37a8f0ab48489af853f9a2d4105808f35c95d73654820b46da8",
          "width_logits": "5226a2c36987e0e2055c5d76ed204bc0739b255fb74eb6ef7d636dc250b424ca"
        },
        "query_pair_index": 3
      },
      {
        "audit_index": 15,
        "changed_task_id": "d968ffd4_noid_15",
        "decoded_digest": {
          "decoded_color_digest": "fc4d38746acd96612e2e22a9b1d57592ff2e70c7d7fbc09afc33b197b9c21202",
          "decoded_height_digest": "f8c55961e5a9816150353765e9d9d613d7d9f9af438b8ada1e9309e995035684",
          "decoded_width_digest": "bdab27466b0f872c7536642ec036002aa6e247555933d2006b9363d4cba7a0f0"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "6f803a40f419f05c62a4daf164b5e15088c32881feac675a7c64994c05fc39ab",
          "query_inputs": "a33435204f0568e4b4586c5e5a42fe95ebcdc1bf925f9a5a41a5c2d35ce38e2a",
          "support_evidence_colors": "da042f98d2a1ba8a9848500946e57af3ec60c0824f9e72d5352238748f7cb217",
          "support_evidence_cols": "d21396284f7c5983d9fe96c80c5ff8b9ee734c79d0590cdbb1ff22044d2848c2",
          "support_evidence_mask": "1bbbdc91a1bab388ff86953d5e2a1e8c86f97dbb14dfaa8e5e80e75f21ba261b",
          "support_evidence_pair_indices": "159bc97a654f5f05e1687088c184906212692fde2ced4a4dd249900dabe9eccf",
          "support_evidence_roles": "b7b89e5a2085caec6274432bca64c8cff7fa885c9eac8e000f7cbc6d90a56c03",
          "support_evidence_rows": "93442b11aa0a3376837810021bb44022739fbd553e7e3f8657a4bfc52921730f"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "d968ffd4",
        "output_digest": {
          "color_log_probs": "a59d24a5cdb53636d48152ef048e1e5b201414361485f0bfc0f73f683c8d9776",
          "color_logits": "a59d24a5cdb53636d48152ef048e1e5b201414361485f0bfc0f73f683c8d9776",
          "color_probabilities": "f465f742b05fb4ccd2485f4246d43d668b88a44971503252d2d15a913d88ce84",
          "copy_attention": "cf5c95a74227859ed81fcd5fcd96d86d65fc085821780fd3259431510cee6114",
          "copy_distribution": "f6567af48fb186b1b3244716504763b243b1be4d2d5693227211e6a406c815d5",
          "copy_entropy": "5e7920d712a42acae4e907d0cf30e7fe3a7cd25ac3e725b687393c789d400914",
          "copy_gate": "7964fd1069ee89b5294ca422d5278fa1e598516f62b410b4852e3eff3fc0b919",
          "final_canvas": "3b0af5fba8aeda4fbaa500b37e6a1f1a127fe8f0c583e87ec8a2737de803dea7",
          "final_rewrite_state": "c8c622fa8354004cb919434051c4c33fcdb65d98c7861c796085e5988686f989",
          "height_logits": "83098ef3bdc4ab507b2cd682e41c6573b7db1fbef253cb48ec9edd043411dae1",
          "logits": "a59d24a5cdb53636d48152ef048e1e5b201414361485f0bfc0f73f683c8d9776",
          "paint_distribution": "7cbe6cf8d5f5eab11bdf99c18120a96c423da67e3e08be0f0ed883ada7d01f5c",
          "paint_entropy": "a1e33e4543c6a22259313712e8468d5ae023d8e9ed415603d0b8c7626a10d17f",
          "rule_tokens": "eec32b4b99ccd821ccee2a149bdc4fa27fe7c1c7406de153c3bb34b2e6bb4cfc",
          "support_evidence_bank": "b3d9f16589a00b52b309b0560307963b60be6afdbd75e82a8ba947fdf882925f",
          "support_evidence_mask": "1bbbdc91a1bab388ff86953d5e2a1e8c86f97dbb14dfaa8e5e80e75f21ba261b",
          "width_logits": "bf52b34326f82c7a824be5417e105fd1d91ac9ec35eb7f53fb87a9c7afbbad42"
        },
        "query_pair_index": 2
      },
      {
        "audit_index": 16,
        "changed_task_id": "496994bd_noid_16",
        "decoded_digest": {
          "decoded_color_digest": "848c55a74d262c2e780ac90ee0f8863ea75f71e3a289b16a3f5dfff560c2f801",
          "decoded_height_digest": "a7a991a2c6f6c88328c36dea39a728e1eafb554d42cff293d0f4cde43b914582",
          "decoded_width_digest": "bdab27466b0f872c7536642ec036002aa6e247555933d2006b9363d4cba7a0f0"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "ee5e4cb6fef8e966b086dc333705242e569235c152672d4ec09093cb5f3ab60b",
          "query_inputs": "3a7788316a7d51ef6ad82f5000fa175031566cadf4d2e020ce90cf23cadd9d7f",
          "support_evidence_colors": "eb1966c9a5b0c2624a2cd92b726dc444a8f3f4d5b11919a3adae5eb4a9351bfc",
          "support_evidence_cols": "609b78abd4361146b50589f4733a623a6f9af53fc2729eda09b1beceefbcc7d2",
          "support_evidence_mask": "a41d79b52e926aafbbf30ed59537a574e90f57aa44f07cbc7a4d2dcdb822e09e",
          "support_evidence_pair_indices": "a3ae7431318f075f91d036488f02a5611c4cedb34e994b99517be381596a03e7",
          "support_evidence_roles": "db24c4a32968e67765dabcd5eeaec41d84c3975f586f3acf561a5dc86175ea24",
          "support_evidence_rows": "cd1a76ee5aed09aa77e09ef132efe70880a4e392c88971494128a3c7170045de"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "496994bd",
        "output_digest": {
          "color_log_probs": "23406ce2fc2d5e2cf09e81f01d9ccf9c01c3cd43b8c754cfa32857f2a417230f",
          "color_logits": "23406ce2fc2d5e2cf09e81f01d9ccf9c01c3cd43b8c754cfa32857f2a417230f",
          "color_probabilities": "1d32e52529c78492028df57af9204da3209a470e5406943bf765baeecd44bf6a",
          "copy_attention": "4aee41d372ff2b48df7df58857febad2efec2f86b1b567799ce0f4eeb0c9fb15",
          "copy_distribution": "33aa1e5ec9d253e9e2f57eeb1b6a771cd8557cc18a5e050ecbb33a5fb1547476",
          "copy_entropy": "8cf32d41e9bf5e8478460f0339f72f86503c6e4e909c27e05997b2289ced745b",
          "copy_gate": "9e1496d66ebf8caac53679866e2c1830eb0d0d0909307627aab67497107cd9b0",
          "final_canvas": "38f3f481d28187cbd50e8877152d5ee9d254351e1d58036d350f07200b5f9061",
          "final_rewrite_state": "8253b7c9b662f811b9a8449bba035c991dff4feeb4dcad354c6c0224cb914362",
          "height_logits": "23a624e6cf8aafa3a1e38d9427fc3abdc5f7348bacf5edefd16f047bb192aa55",
          "logits": "23406ce2fc2d5e2cf09e81f01d9ccf9c01c3cd43b8c754cfa32857f2a417230f",
          "paint_distribution": "c31f89d5e1b2293d125821fca52d0a3ba7fd537081553279dfb7e7b92bb7a38f",
          "paint_entropy": "2c0271e8b6e5db1407c4c324b4545a7b361a6ee487b165dd7c615d6354323c01",
          "rule_tokens": "8cda8c1791711a30d4613b882a0d403283c605206922f74c052d4a9ffcf217ae",
          "support_evidence_bank": "e3eba0c3dd5a5f147616ca83310f6ea05642e5aad3181f0ea2ffc643fd685e30",
          "support_evidence_mask": "a41d79b52e926aafbbf30ed59537a574e90f57aa44f07cbc7a4d2dcdb822e09e",
          "width_logits": "850a2a30f86dfcf55e2bb5e4b060715b018af751cceb4fd31b80c04af579f686"
        },
        "query_pair_index": 0
      },
      {
        "audit_index": 17,
        "changed_task_id": "137eaa0f_noid_17",
        "decoded_digest": {
          "decoded_color_digest": "9b245d84deba449f34f828f6fde32a99f5a04f1f340d3afabbf75e68703bc23c",
          "decoded_height_digest": "de2978a648a9f6175654a233388cd1f224e5906662c185ee8e24b2d6060c506b",
          "decoded_width_digest": "de2978a648a9f6175654a233388cd1f224e5906662c185ee8e24b2d6060c506b"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "308a51b32da610bebd9c348d9123e831d0cbc4dd790192c9a3e60c75f97bea61",
          "query_inputs": "a2a3e8c81ab871b6954cdac7bf42f82f14f96bdbcb96cc1df8e09b20810170fd",
          "support_evidence_colors": "c4b0148c9564ec5310f3e511d178d9c2d8251af6e650658aa38a69af235269db",
          "support_evidence_cols": "d5983dff07e18d1305461bbd0c0170231da5af1c042147b5459e81748d08baa5",
          "support_evidence_mask": "112915f271221fe5d347b68e9115e4adabb835e45b4a145f6dd743faf7ee5324",
          "support_evidence_pair_indices": "e0370cafcaac026d4e537731efe38bb16c6291fe61a0a0a0743759b44a4e99b6",
          "support_evidence_roles": "3424ffb6effe7f3dbfabb7bcb0f04c56d3c5ffad6677b2d25343f61fd52f277f",
          "support_evidence_rows": "7c9a64dfe9c5f6fced44ae7c2f2fa1044401aee5fedd4483c025da67a9214e0b"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "137eaa0f",
        "output_digest": {
          "color_log_probs": "f7cda67b82e753afef16a8aeb27eb7e29d160ae83c9ef7efb657ec42b330b359",
          "color_logits": "f7cda67b82e753afef16a8aeb27eb7e29d160ae83c9ef7efb657ec42b330b359",
          "color_probabilities": "35f924bda0ca9326374448343361d0301c7eb96d9770f891074905967cbb7be9",
          "copy_attention": "edfe687364bbf7f09d2d178237383a2874ff45502e80b156da2fcd2a298c2f4d",
          "copy_distribution": "5a73c8a2948e96ac997d85bf4d6edaa7239067c3522809f24e9a3885c46365cc",
          "copy_entropy": "21399ce57b56ee45775da5f82a37f92c1113aa093db6edddaab682746f481dac",
          "copy_gate": "811b023cfc58203eb63c2254fe98d07f105c28929c51c66d1c4c3099412e311f",
          "final_canvas": "99c80e497fd360c29324a5d7a608f680dbf46a70237bfc6cbffb77ca4efa55ef",
          "final_rewrite_state": "332d8a84ac09494b63c309823defddd548ce42ff7561cd627cbf2662b36c19ca",
          "height_logits": "1ac870423878b5064da0697b6b48741b01cd6f5d1bbc0effa0161ce013aa2bf4",
          "logits": "f7cda67b82e753afef16a8aeb27eb7e29d160ae83c9ef7efb657ec42b330b359",
          "paint_distribution": "29fdb69f9992529f158ec3615d26ef0e00715eb3693b2d539e60e5b56a88c823",
          "paint_entropy": "0a4a6e93558141faa4a22a17e4d07b289cd95543ff1d4b7ce4dabf3fcab80598",
          "rule_tokens": "ff0c32bc5ad46baadb3479a59a5bc56a9b6f19e974e3304b21b0ac8b0164a46a",
          "support_evidence_bank": "3b21976871a84e127370e99ed4251407cf4a87acaba6f60f7d394b1cacb05b95",
          "support_evidence_mask": "112915f271221fe5d347b68e9115e4adabb835e45b4a145f6dd743faf7ee5324",
          "width_logits": "c16689402564fc963f21896e1edcdd5442d987694ab645b870a6f359659584b4"
        },
        "query_pair_index": 1
      },
      {
        "audit_index": 18,
        "changed_task_id": "a64e4611_noid_18",
        "decoded_digest": {
          "decoded_color_digest": "95cf21bac70a4c532d9d4d0fb96ed03742bcc69125f8032b3bfd6c272795306b",
          "decoded_height_digest": "83edbc1cb4f3febe31199c74742bfa9643b20b6705254466f0d6b97ff58d04c2",
          "decoded_width_digest": "83edbc1cb4f3febe31199c74742bfa9643b20b6705254466f0d6b97ff58d04c2"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "cdf9869e97d2584330d73360cfa1b12304644bf3c293c966a061916985d536b9",
          "query_inputs": "e65c5d8f45a8c1e861b3aaf7608c40a7e094c0938ba8e6d19e1ac15800a065e5",
          "support_evidence_colors": "2e1c543906310ec845161cb55832e3a8e5118b45f9378892ef0f4601e0fb7aef",
          "support_evidence_cols": "241706c9a38bcd9278f88fcf5f774863ab52be3fe783397036ae8282562f2a00",
          "support_evidence_mask": "31538e8b48cfabcfe6d6abaf4318bcc4ef4189fc274c16f18efccb66e3be4b02",
          "support_evidence_pair_indices": "d40424975802f973ca99b1c943614d119353de69c468354ae8b9a152ae49f9d4",
          "support_evidence_roles": "9d43b210e216309e93ae35bf7882df3e426ffc067fde22e7cc1a333c181da8d9",
          "support_evidence_rows": "64ca428ee2218e3bb9f8f088ec29ea1793f61cf1220e8d3da90bc6c4dff8304f"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "a64e4611",
        "output_digest": {
          "color_log_probs": "4f3ae16563819cba8a709e869b0536c58de9f502483249bd7260ddb0da5125f5",
          "color_logits": "4f3ae16563819cba8a709e869b0536c58de9f502483249bd7260ddb0da5125f5",
          "color_probabilities": "7f031ad6b318db790cb43269f548635deb03006ad9ec9c6437ec69c95d270116",
          "copy_attention": "be12ff0403ebcd0c33988329da51a7d4bdcd654d19a12e995a0f71499447d5d4",
          "copy_distribution": "6a3884c2720540fc2269808c51b89a5e1f9cec9508296f8c4c85f3bc67ffbee7",
          "copy_entropy": "6ca4fe5d23ac42f970451b79656bf33b3923ef711ee85ecb9f6028a32c5b1459",
          "copy_gate": "c37174a87e042c217fefe5ba8a84713d611952f6e9c949d4df8b4f107878b008",
          "final_canvas": "78c5aa9411da3d381dd906cbaa6085a4c2d42ed3a0523ec6176fbc145dca1dda",
          "final_rewrite_state": "584ed6cbb44a648a75a917c9e9d9f68714dcff36f4927452cf1d6b93fa852c42",
          "height_logits": "d2b414c446fd46aa53c9bddf0d02156e8e7d7e33e067f16aeb85da388d4061ba",
          "logits": "4f3ae16563819cba8a709e869b0536c58de9f502483249bd7260ddb0da5125f5",
          "paint_distribution": "dbf01798291b553d1fbbe89eb03eefc3c8f077f06d15fa11263d945f3106d3ea",
          "paint_entropy": "9408bc756f8724e198afbf6c3ffc5e4325f6178e6621cf17a7487988138c1ac2",
          "rule_tokens": "f49b69f6be42ae6e8087ee318f8ba92cfb70ebf3e0ff6e10668a4a3aadff7ef2",
          "support_evidence_bank": "6a8278c7affa53155528bd536b27f690236d2a8e3a780cdd3ec3260511d48e92",
          "support_evidence_mask": "31538e8b48cfabcfe6d6abaf4318bcc4ef4189fc274c16f18efccb66e3be4b02",
          "width_logits": "814243b0d35d0f475676c203681eddf2362a3b4cb2ed7440ac72b5d88f73f506"
        },
        "query_pair_index": 0
      },
      {
        "audit_index": 19,
        "changed_task_id": "7b6016b9_noid_19",
        "decoded_digest": {
          "decoded_color_digest": "d153bcb28f1243d138e82d371544ecf2f8d8724424fdd40f46580f2eec069d43",
          "decoded_height_digest": "b512577c66cc1dd42fa59b05ddffd5c836c9172a310fbe1a4c8654347162b6e6",
          "decoded_width_digest": "edb42ac8365616ae788242305ba5dcff22efb27ced93e0f150026024f1f656fe"
        },
        "decoded_outputs_byte_identical": true,
        "input_digest": {
          "query_input_mask": "6d38b048977a9ca09de481352ecf8ec92c5bc863bbcabf197699b6a8b5849123",
          "query_inputs": "789715eda0177143513cfaf706628a6359710377e0a36237c0df69f50c2eecff",
          "support_evidence_colors": "bbb2b574433e6357dfe0e66cd38441dfa78e5b80f73a061de300b1c1b951f7c8",
          "support_evidence_cols": "f1f7a3f37397d419c538200d2ddcf53e5750576f8e465efeff662e203a1a45fa",
          "support_evidence_mask": "83b9b4a468588f09cbd5d21619874b8652a5e7ebc3d6f4f09ca3a00607337427",
          "support_evidence_pair_indices": "bf3298068dd762860380973ea3e392c97b33c270d116ad0d5d1345465b3ddefb",
          "support_evidence_roles": "36a79ff074b391d0f2702693b2ada9a5380bfd66f684ae4125241673e168dede",
          "support_evidence_rows": "299859cc0675864db33fde1a4bd8509dc7ad5787189949a4f1e05233c2c1ab3a"
        },
        "metadata_changed": true,
        "model_input_tensors_byte_identical": true,
        "model_output_tensors_byte_identical": true,
        "original_task_id": "7b6016b9",
        "output_digest": {
          "color_log_probs": "cfc295f2d4d27bd0701aed8a9c7c8fe1e0cee421a263883c8c9a0f3c82e11a88",
          "color_logits": "cfc295f2d4d27bd0701aed8a9c7c8fe1e0cee421a263883c8c9a0f3c82e11a88",
          "color_probabilities": "8596ac33eb87082efbc65d288ba18e767e0b95a2c3b91cc9041af044e2fc8eb2",
          "copy_attention": "d4f56407320018d537a92aedd663a07b9dd84e2425bcd726ef17a73bbde88abd",
          "copy_distribution": "e104aa19f825c965c58b704c27b1c326a5efb160eea604b72ed01905524af1fc",
          "copy_entropy": "b9458932dc59c60966cdf125b64fc7c776aa8a5434e806e2d5f7cdb7037d5655",
          "copy_gate": "ed1b2937e57792a6506bf1124e0ff38a762f3af1095009792f5308993393e6b8",
          "final_canvas": "40e0ff0aec6b6f1cfd2a406a03ca6ee4fbeb99f3f05657a6cfc36556d098151d",
          "final_rewrite_state": "7cd5a8e43eb0256be2c0fd03f75b80576ba109fec981b6d9bbae519144c527be",
          "height_logits": "6a69198edc81ce8f8d85a22151c9210eb3e7552154d82546ca0a606ed555d63f",
          "logits": "cfc295f2d4d27bd0701aed8a9c7c8fe1e0cee421a263883c8c9a0f3c82e11a88",
          "paint_distribution": "28d9ce8b2fd69f39b8c5267fb599dee2afc46901ba37edaf47050d2e3f11faa3",
          "paint_entropy": "5826cf810ba3de1c9247ecc10f9b9eca821dd1c2628153a38d25dfe0e5b28004",
          "rule_tokens": "ebe9fbbeed5bc0afe6503a988783f25c37b1f4090fe1eed16cc4dd1ae5727f70",
          "support_evidence_bank": "37bcf0fd8127ea8bd5fff09233e2fec15a2f0bf6ca3c1e285afca95e2683f279",
          "support_evidence_mask": "83b9b4a468588f09cbd5d21619874b8652a5e7ebc3d6f4f09ca3a00607337427",
          "width_logits": "db697c7112eb82dd25ef2f83283cfc1163af4ddb5384b73759453d9fa139d090"
        },
        "query_pair_index": 2
      }
    ]
  },
  "forward_input_provenance_path": "/mnt/d/VRM/artifacts/r5a4a_neural_vrm_arc_grounded_preflight/forward_input_provenance.json",
  "no_id_causal_audit_pass": true,
  "static_source_scan": {
    "failure_count": 0,
    "failures": [],
    "occurrence_count": 298,
    "scan_terms": [
      "task_id",
      "puzzle_id",
      "identifier",
      "embedding",
      "cache",
      "checkpoint",
      "filename",
      "path",
      "dataset_index"
    ],
    "static_source_scan_pass": true
  }
}
```

## 5. Exact verdict category

```json
{
  "query_only_supported": false,
  "verdict": "INCONCLUSIVE"
}
```

## 6. What this establishes

```json
{
  "determinism_verification": {
    "original_fold_count": 156,
    "rerun_condition": "TRAINED_FULL_SUPPORT",
    "rerun_fold_count": 156,
    "stable_artifacts_match_excluding_runtime": true
  },
  "source_artifact_integrity": {
    "added": [],
    "changed": [],
    "missing": [],
    "protected_file_count": 108,
    "protected_r5a2_r5a3_unchanged": true
  },
  "statement": "This preflight measures trained-vs-fresh behavior and support-output sensitivity on a fixed real-ARC sentinel leave-one-out set under a no-ID input audit."
}
```

## 7. What this does not establish

```json
{
  "not_claimed": [
    "ARC solving",
    "reasoning",
    "abstraction",
    "hidden tools",
    "generalization beyond this frozen sentinel preflight",
    "synthetic task performance"
  ]
}
```

## 8. Narrowest justified next action

```json
{
  "do_not_automatically_start": true,
  "next_action": "D. Investigate why query-only or zeroed-support behavior dominates despite no-ID compliance."
}
```
