#!/usr/bin/env bash
# Fresh 20-epoch budget comparison; reuse prepared data and fixed semantic IDs.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs
exec 9>runs/pipeline.lock
flock -n 9 || { printf '%s\n' 'Another Books pipeline is already active'; exit 1; }
stage() { printf '%s %s\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$1" | tee runs/pipeline_stage.txt; }
trap 'stage "FAILED_20epochs line=$LINENO"' ERR
stage formal_training_20epochs
bash start_train.sh --epochs 20 --batch-size 512 --eval-batch-size 64 \
  --val-users 20000 --beam-size 20 --lr 0.001 --patience 3 --seed 42
stage complete_20epochs
printf '%s\n' PIPELINE_COMPLETE
