#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ "$(uname -s)" == Linux && "$PWD" == /root/autodl-tmp/* ]] || exit 2
: "${QWEN3_SERVE_APPROVED_BUDGET_CNY:?approved budget}"
: "${QWEN3_PRICE_CNY_PER_HOUR:?verified hourly price}"
: "${QWEN3_INSTANCE_STARTED_AT:?actual boot time}"
: "${QWEN3_PLATFORM_DEADLINE:?verified platform shutdown deadline}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export UNSLOTH_MOE_BACKEND=unsloth_triton UNSLOTH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${QWEN3_VENV:-/root/qwen3-moe-venv}/bin/python"
LIMIT="$("$PYTHON" -c 'import os,time; from protocol import service_seconds; print(service_seconds(float(os.environ["QWEN3_SERVE_APPROVED_BUDGET_CNY"]),float(os.environ["QWEN3_PRICE_CNY_PER_HOUR"]),float(os.environ["QWEN3_INSTANCE_STARTED_AT"]),float(os.environ["QWEN3_PLATFORM_DEADLINE"]),time.time()))')"
mkdir -p ../runs
RUN_ID="qwen3-agent-$(date +%Y%m%d-%H%M%S)-$$"
mkdir "../runs/$RUN_ID"
exec > >(tee -a "../runs/$RUN_ID/experiment.log") 2>&1
finish() { code=$?; printf 'AGENT_EXPERIMENT_EXIT code=%s time=%s\n' "$code" "$(date -Iseconds)"; sync; /usr/bin/shutdown; }
trap finish EXIT
ln -sfn "$RUN_ID/experiment.log" ../runs/latest_agent.log
timeout --signal=TERM --kill-after=15s "${LIMIT}s" "$PYTHON" agent_experiment.py --run-dir "../runs/$RUN_ID" "$@"
