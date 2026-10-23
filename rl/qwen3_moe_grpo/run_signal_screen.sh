#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ "$(uname -s)" == Linux && "$PWD" == /root/autodl-tmp/* ]] || exit 2
: "${QWEN3_SCREEN_APPROVED_BUDGET_CNY:?Confirm the new 1024-question screening budget}"
: "${QWEN3_PRICE_CNY_PER_HOUR:?Verify the current price}"
: "${QWEN3_INSTANCE_STARTED_AT:?Actual instance start time in Unix seconds}"
: "${QWEN3_PLATFORM_DEADLINE:?Confirmed platform shutdown time in Unix seconds}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/root/autodl-tmp/qwen3-hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNSLOTH_MOE_BACKEND=unsloth_triton UNSLOTH_COMPILE_DISABLE=1
PYTHON="${QWEN3_VENV:-/root/qwen3-moe-venv}/bin/python"
RUN_ID="qwen3-signal-screen-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p runs
exec > >(tee -a "runs/$RUN_ID.log") 2>&1
finish() { code=$?; printf 'SIGNAL_SCREEN_EXIT code=%s time=%s\n' "$code" "$(date -Iseconds)"; sync; /usr/bin/shutdown; }
trap finish EXIT
DEADLINE="$("$PYTHON" - <<'PY'
import json,os,time
from screen_training_signal import budget_deadline
with open('config_screen.json') as f:
    config=json.load(f)
print(budget_deadline(config,
    float(os.environ['QWEN3_SCREEN_APPROVED_BUDGET_CNY']),
    float(os.environ['QWEN3_PRICE_CNY_PER_HOUR']),time.time(),
    float(os.environ['QWEN3_INSTANCE_STARTED_AT']),
    float(os.environ['QWEN3_PLATFORM_DEADLINE'])))
PY
)"
printf '%s\n' "$RUN_ID" > runs/latest_signal_screen_run.txt
ln -sfn "$RUN_ID.log" runs/latest_signal_screen.log
SECONDS_LEFT=$((DEADLINE - $(date +%s)))
(( SECONDS_LEFT > 600 )) || exit 2
"$PYTHON" -m unittest -v test_signal_audit.py test_signal_screen.py
timeout --signal=TERM --kill-after=60s "${SECONDS_LEFT}s" "$PYTHON" screen_training_signal.py \
  --model-path /root/autodl-tmp/qwen3-30b-a3b-bf16 --run-dir "runs/$RUN_ID" --deadline "$DEADLINE"
"$PYTHON" analyze_signal_screen.py "runs/$RUN_ID"
