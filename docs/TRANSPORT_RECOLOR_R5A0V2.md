# R5A-0 V2 - Direct Factorized Transport-Recolor Baseline

R5A-0 V2 is the versioned repair of the R5A-0 transport/recolor experiment. V1 stopped correctly because its grammar contained exact functional collisions. V2 replaces that grammar with a non-colliding heterogeneous set:

- singles: `C`, `H`, `V`, `L`, `K`
- seen mixed train/dev classes: `C->L`, `L->H`, `H->K`, `K->V`
- held-out sealed mixed classes: `L->C`, `H->L`, `C->K`, `K->C`

The architecture remains direct, one-shot, and factorized:

```text
output_probs[u,c] = sum_v A[u,v] * B[u,v,c]
```

There are no operation labels, sequence labels, intermediate labels, recurrence, test-time training, connection-cost loss, sparsity loss, modularity loss, or auxiliary losses. Training uses final output cross entropy only.

## Commands

Dry run:

```bash
python -m vrm.twin_tasks.run_transport_recolor_r5a0v2 \
  --mode dry_run \
  --seeds 20260628,20260629,20260630 \
  --batch-size 128 \
  --max-batches 4 \
  --data-out /mnt/d/VRM/artifacts/transport_recolor_r5a0_v2 \
  --checkpoint-out /mnt/d/VRM/checkpoints/transport_recolor_r5a0_v2
```

Full run:

```bash
python -m vrm.twin_tasks.run_transport_recolor_r5a0v2 \
  --mode full \
  --seeds 20260628,20260629,20260630 \
  --epochs 12 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --data-out /mnt/d/VRM/artifacts/transport_recolor_r5a0_v2 \
  --checkpoint-out /mnt/d/VRM/checkpoints/transport_recolor_r5a0_v2
```

Tests:

```bash
pytest -q
```

CUDA is mandatory. The runner stops with `CUDA_REQUIRED_NOT_AVAILABLE` if `torch.cuda.is_available()` is not true.

## Artifacts

V2 writes artifacts only under:

```text
/mnt/d/VRM/artifacts/transport_recolor_r5a0_v2/
```

V2 writes checkpoints only under:

```text
/mnt/d/VRM/checkpoints/transport_recolor_r5a0_v2/
```

R5A-0 V1 source and artifacts are intentionally not patched or overwritten.

## 2026-06-29 Order Audit Addendum

An additive order-distinguishability and semantic-function-overlap audit was added after the original V2 run:

```text
docs/TRANSPORT_RECOLOR_R5A0V2_ORDER_AUDIT.md
vrm/twin_tasks/transport_recolor_r5a0v2_order_audit.py
artifacts/transport_recolor_r5a0_v2/order_audit_report.json
artifacts/transport_recolor_r5a0_v2/order_audit_samples.json
```

The addendum does not rewrite the original V2 result and does not change generator semantics, operation grammar, model architecture, training, checkpoints, or split definitions.
