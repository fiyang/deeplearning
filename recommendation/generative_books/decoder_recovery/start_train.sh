#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec 9>runs/pipeline.lock
flock -n 9 || { printf '%s\n' 'Another Books task is already active'; exit 1; }
stage() { printf '%s %s\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$1" | tee runs/pipeline_stage.txt; }
trap 'stage "FAILED_decoder_recovery line=$LINENO"' ERR
python_bin="$(pwd)/../generative_beauty/.venv/bin/python"
parent="$(pwd)/runs/books-generative-20261009-153342-14533"
run_name="books-decoder-recovery-lr1e-4-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
run_dir="$(pwd)/runs/$run_name"
printf '%s\n' "$run_dir" > runs/latest_run.txt
stage decoder_recovery_training
"$python_bin" -u decoder_recovery/train_recovery.py --parent-run "$parent" --run-dir "$run_dir" \
  --additional-epochs 15 --lr 0.0001 --warmup-steps 1000 --batch-size 512 --eval-batch-size 64 \
  --patience 3 --seed 42 2>&1 | tee "$run_dir.log"
stage complete_decoder_recovery
printf '%s\n' PIPELINE_COMPLETE
