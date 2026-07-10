# VRM V1.1 — Masked Query Spatial Evidence Rewrite

VRM V1.1 is an experimental ARC-2 direct-grid capacity probe. It did not solve ARC and is not a solver claim.

V1.1 keeps the core V1 spatial evidence idea:

- packed support evidence bank;
- direct support cross-attention at every rewrite cycle;
- query input encoding;
- recurrent spatial output canvas;
- copy-and-paint color prediction.

It does not add latent rule refinement, test-time training, program search, a verifier, RoPE, or augmentation.

## Architecture

Support demonstrations are packed into valid-cell evidence tokens. Each support evidence token carries color, row, column, role, and support-pair index. The support evidence bank remains directly accessible to the recurrent rewrite state at every rewrite cycle.

The query input is still represented as a 900-position 30x30 grid, but V1.1 also passes a boolean `query_input_mask`. Query cross-attention and copy attention may attend only to valid query cells inside the original query rectangle. Padded query positions are masked explicitly; color 0 remains a real ARC color and is never used to infer padding.

The rewrite state is:

```text
[height shape token]
[width shape token]
[900 output canvas slots]
```

After the final rewrite cycle, a final RMSNorm feeds separate heads:

```text
height_logits: [batch, 30]
width_logits:  [batch, 30]
color_logits:  [batch, 900, 10]
```

The color output has only ARC colors `0..9`; there is no PAD color class in the V1.1 output head. Output shape is predicted explicitly by the height and width heads.

The color objective trains only cells inside the true target rectangle. Cells outside target `H x W` contribute zero color loss. Height and width targets are encoded as `target_height - 1` and `target_width - 1`.

Default model settings:

```text
hidden_size = 128
num_heads = 4
pair_encoder_layers = 1
rewrite_layers = 2
rewrite_steps = 4
rule_slots = 32
dropout = 0.0
```

Default optimizer settings:

```text
optimizer = AdamW
learning_rate = 2e-4
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
warmup_steps = 100
min_learning_rate = 2e-5
grad_clip_norm = 1.0
shape_loss_weight = 0.5
```

## Data

The local ARC data root convention is:

```bash
/mnt/d/VRM/arc-prize-2026-arc-agi-2
```

The data layer uses ARC training challenges for leave-one-out episodes:

```text
supports = all other train pairs from the same task
query input = held-out train-pair input
query target = held-out train-pair output
```

Task IDs are metadata only and never enter `VRM.forward`. Query targets, target shape, and target colors are used only for loss and metrics.

V1.1 reuses the prior V1 capacity50 task manifest when available:

```text
/mnt/d/VRM/artifacts/vrm_v1_capacity50/vrm_v1_task_manifest.json
```

## Commands

Run from WSL Ubuntu:

```bash
source ~/.venvs/vrm/bin/activate
cd /mnt/d/VRM
```

Run tests:

```bash
pytest -q
```

Run the V1.1 25-step dry check:

```bash
python -m vrm.train --mode v1_1_capacity50 --steps 25 --batch-size 1 --seed 20260627
```

Run the V1.1 50-task capacity and unseen-task probe:

```bash
python -m vrm.train \
  --mode v1_1_capacity50 \
  --steps 1500 \
  --batch-size 1 \
  --seed 20260627 \
  --learning-rate 2e-4 \
  --warmup-steps 100 \
  --min-learning-rate 2e-5 \
  --grad-clip-norm 1.0 \
  --shape-loss-weight 0.5
```

Artifacts are written to:

```text
/mnt/d/VRM/artifacts/vrm_v1_1_capacity50/
```

Checkpoints are written to:

```text
/mnt/d/VRM/checkpoints/
```

The source snapshot is:

```text
/mnt/d/VRM/VRM_V1_1_SOURCE_SNAPSHOT.zip
```

The source ZIP excludes ARC data, checkpoints, full artifacts, prediction dumps, per-episode dumps, training logs, caches, virtual environments, and model-weight files.
