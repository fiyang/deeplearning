#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec 9>runs/pipeline.lock
flock -n 9 || { printf '%s\n' 'Another Books task is already active'; exit 1; }
stage() { printf '%s %s\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$1" | tee runs/pipeline_stage.txt; }
trap 'stage "FAILED_tiger_w256 line=$LINENO"' ERR
run_name="books-tiger-w256-20epochs-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
run_dir="$(pwd)/runs/$run_name"
printf '%s\n' "$run_dir" > runs/latest_run.txt
stage tiger_w256_training
../generative_beauty/.venv/bin/python -u tiger_scaling/train.py --run-dir "$run_dir" \
  --epochs 20 --lr 0.001 --batch-size 512 --eval-batch-size 64 --val-users 20000 \
  --beam-size 20 --patience 3 --seed 42 2>&1 | tee "$run_dir.log"
stage complete_tiger_w256
printf '%s\n' PIPELINE_COMPLETE
