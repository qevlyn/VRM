# R4A-1Q - Raw Pointer Metric Reconciliation

R4A-1Q is an evaluation-only forensic reconciliation of pointer metrics from R4A-0R V2 and R4A-1P.

It does not train, create an optimizer, call backward, mutate model parameters, save checkpoints, choose checkpoints, or edit earlier experiments. It loads only the three frozen R4A-0R V2 best-dev checkpoints.

The reconciliation compares the same raw pointer-attention tensor:

```text
A[u, v]
```

where `u` is a target cell and `v` is one of the 49 query source coordinates. It computes foreground-only oracle-coordinate mass, foreground raw max probability, foreground raw entropy, and the historical R4A-0R all-cell max/entropy reductions.

Run:

```bash
python -m vrm.twin_tasks.run_composition_r4a1q \
  --mode full \
  --seeds 20260628,20260629,20260630 \
  --batch-size 128 \
  --data-out /mnt/d/VRM/artifacts/composition_r4a1q_v1
```

The intended outcome is a metric-semantics reconciliation only. It is not a new training result, not an ARC-solving claim, and not evidence of explicit primitive execution.
