#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/VRM
export PYTHONPATH=/mnt/d/VRM
/home/quyen/.venvs/vrm/bin/python -u -m vrm.twin_tasks.run_mixed_router_r3b2 \
  --mode full \
  --seeds 20260628,20260629,20260630 \
  --epochs 8 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --data-out /mnt/d/VRM/artifacts/mixed_router_r3b2_v1 \
  --checkpoint-out /mnt/d/VRM/checkpoints/mixed_router_r3b2_v1 \
  > /mnt/d/VRM/artifacts/mixed_router_r3b2_v1/full_run_console.log 2>&1
