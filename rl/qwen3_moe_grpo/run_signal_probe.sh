#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ "$(uname -s)" == Linux && "$PWD" == /root/autodl-tmp/* ]] || exit 2
: "${QWEN3_PROBE_APPROVED_BUDGET_CNY:?Confirm this new inference-only diagnostic budget}"
: "${QWEN3_PRICE_CNY_PER_HOUR:?Verify the current price}"
: "${QWEN3_HARD_DEADLINE:?Confirm a new platform shutdown timer}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/root/autodl-tmp/qwen3-hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNSLOTH_MOE_BACKEND=unsloth_triton UNSLOTH_COMPILE_DISABLE=1
PYTHON="${QWEN3_VENV:-/root/qwen3-moe-venv}/bin/python"
RUN_ID="qwen3-signal-probe-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p runs
exec > >(tee -a "runs/$RUN_ID.log") 2>&1
finish() { code=$?; printf 'SIGNAL_PROBE_EXIT code=%s time=%s\n' "$code" "$(date -Iseconds)"; sync; /usr/bin/shutdown; }
trap finish EXIT
DEADLINE="$("$PYTHON" - <<'PY'
import math,os,time
b=float(os.environ['QWEN3_PROBE_APPROVED_BUDGET_CNY']);p=float(os.environ['QWEN3_PRICE_CNY_PER_HOUR'])
if not all(math.isfinite(v) and v>0 for v in [b,p]) or b>15 or p>5.59:
    raise ValueError('Budget/price outside this diagnostic plan')
print(min(int(time.time())+8400,int(time.time())+int(b/p*3600)-600,int(os.environ['QWEN3_HARD_DEADLINE'])))
PY
)"
SECONDS_LEFT=$((DEADLINE - $(date +%s)))
(( SECONDS_LEFT > 600 )) || exit 2
printf '%s\n' "$RUN_ID" > runs/latest_signal_probe_run.txt
ln -sfn "$RUN_ID.log" runs/latest_signal_probe.log
timeout --signal=TERM --kill-after=60s "${SECONDS_LEFT}s" "$PYTHON" probe_training_signal.py \
  --model-path /root/autodl-tmp/qwen3-30b-a3b-bf16 \
  --source-run /root/autodl-tmp/deeplearning-lab/rl/qwen3_moe_grpo/runs/qwen3-round2-20261022-194758-1550 \
  --run-dir "runs/$RUN_ID" --deadline "$DEADLINE" --per-cohort 64
