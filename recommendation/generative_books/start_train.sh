#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python_bin="$(pwd)/../generative_beauty/.venv/bin/python"
run_name="books-generative-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
run_dir="$(pwd)/runs/$run_name"
mkdir -p runs
printf '%s\n' "$run_dir" > runs/latest_run.txt
"$python_bin" -u train.py --run-dir "$run_dir" "$@" 2>&1 | tee "$run_dir.log"
