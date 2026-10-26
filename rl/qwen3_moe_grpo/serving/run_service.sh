#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ "$(uname -s)" == Linux && "$PWD" == /root/autodl-tmp/* ]] || exit 2
: "${QWEN3_SERVE_APPROVED_BUDGET_CNY:?This service experiment needs its own approved budget}"
: "${QWEN3_PRICE_CNY_PER_HOUR:?Verify the current GPU price}"
: "${QWEN3_INSTANCE_STARTED_AT:?Actual boot Unix timestamp}"
: "${QWEN3_PLATFORM_DEADLINE:?Set and verify the platform shutdown time first}"
: "${QWEN3_API_KEY:?Set a random secret with at least 24 characters}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export UNSLOTH_MOE_BACKEND=unsloth_triton UNSLOTH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${QWEN3_VENV:-/root/qwen3-moe-venv}/bin/python"
LIMIT="$("$PYTHON" - <<'PY'
import os,time
from protocol import service_seconds
budget=float(os.environ['QWEN3_SERVE_APPROVED_BUDGET_CNY'])
price=float(os.environ['QWEN3_PRICE_CNY_PER_HOUR'])
started=float(os.environ['QWEN3_INSTANCE_STARTED_AT'])
platform=float(os.environ['QWEN3_PLATFORM_DEADLINE'])
print(service_seconds(budget,price,started,platform,time.time()))
PY
)"
mkdir -p ../runs
RUN_ID="qwen3-serving-$(date +%Y%m%d-%H%M%S)-$$"
exec > >(tee -a "../runs/$RUN_ID.log") 2>&1
finish() { code=$?; printf 'SERVING_EXIT code=%s time=%s\n' "$code" "$(date -Iseconds)"; sync; /usr/bin/shutdown; }
trap finish EXIT
ln -sfn "$RUN_ID.log" ../runs/latest_serving.log
timeout --signal=TERM --kill-after=30s "${LIMIT}s" "$PYTHON" serve.py "$@"
