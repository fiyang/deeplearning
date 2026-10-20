#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p runs
RUN_NAME="beauty-tiger-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
LOG_PATH="$PWD/runs/$RUN_NAME.log"
echo "Run: $PWD/runs/$RUN_NAME"
echo "Log: $LOG_PATH"
"$PYTHON_BIN" -u train.py --run-dir "$PWD/runs/$RUN_NAME" "$@" 2>&1 | tee "$LOG_PATH"
