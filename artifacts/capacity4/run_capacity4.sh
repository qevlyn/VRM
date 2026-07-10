#!/usr/bin/env bash
set -euo pipefail

source ~/.venvs/vrm/bin/activate
cd /mnt/d/VRM
exec python -m vrm.train --mode capacity4
