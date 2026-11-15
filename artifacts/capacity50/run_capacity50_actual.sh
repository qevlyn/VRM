#!/usr/bin/env bash
set -euo pipefail

source ~/.venvs/vrm/bin/activate
cd /mnt/d/VRM
exec python -m vrm.train --mode capacity50 --steps 1500 --batch-size 1 --seed 20260627
