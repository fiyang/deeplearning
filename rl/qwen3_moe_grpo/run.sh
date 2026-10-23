#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ "$(uname -s)" == Linux && "$PWD" == /root/autodl-tmp/* ]] || {
  echo 'Run only on the remote AutoDL training host.' >&2; exit 2;
}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/root/autodl-tmp/qwen3-hf-cache
export MODELSCOPE_CACHE=/root/autodl-tmp/qwen3-modelscope-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNSLOTH_MOE_BACKEND=unsloth_triton
export UNSLOTH_COMPILE_DISABLE=1
VENV="${QWEN3_VENV:-/root/qwen3-moe-venv}"
PYTHON="$VENV/bin/python"
MODEL=/root/autodl-tmp/qwen3-30b-a3b-bf16
RUN_ID="qwen3-moe-grpo-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p runs
printf '%s\n' "$RUN_ID" > runs/latest_run.txt
ln -sfn "$RUN_ID.log" runs/latest.log
exec > >(tee -a "runs/$RUN_ID.log") 2>&1
printf 'RUN_START id=%s time=%s\n' "$RUN_ID" "$(date -Iseconds)"
finish() {
  code=$?
  printf 'RUN_EXIT code=%s time=%s\n' "$code" "$(date -Iseconds)"
  sync
  if [[ "${AUTO_SHUTDOWN:-0}" == 1 ]] ||
     [[ "${AUTO_SHUTDOWN:-0}" == after-smoke && -f "runs/$RUN_ID.shutdown-enabled" ]]; then
    /usr/bin/shutdown
  fi
}
trap finish EXIT
[[ -x "$PYTHON" ]] || { echo 'Run bootstrap.sh first.' >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /root/autodl-tmp
"$PYTHON" -m pip freeze > "runs/$RUN_ID.environment.txt"
# Model and data downloads, smoke and full run all share a single ten-hour budget.
DEADLINE="$("$PYTHON" -c 'import os,time; from common import load_config; end=int(time.time())+load_config()["job_time_limit_seconds"]; print(min(end,int(os.environ.get("QWEN3_HARD_DEADLINE",end))))')"
JOB_SECONDS=$((DEADLINE - $(date +%s)))
(( JOB_SECONDS >= 1800 )) || { echo 'Less than 30 minutes remain before the hard deadline.' >&2; exit 2; }
export QWEN3_JOB_DEADLINE="$DEADLINE" QWEN3_ACTIVE_RUN="$RUN_ID"
timeout --signal=TERM --kill-after=120s "${JOB_SECONDS}s" bash -Eeuo pipefail -c '
  PYTHON="${QWEN3_VENV:-/root/qwen3-moe-venv}/bin/python"
  MODEL=/root/autodl-tmp/qwen3-30b-a3b-bf16
  RUN_ID="$QWEN3_ACTIVE_RUN"
  "$PYTHON" prepare_data.py
  "$PYTHON" download_model.py --model-path "$MODEL"
  "$PYTHON" train.py --model-path "$MODEL" --run-dir "runs/$RUN_ID-smoke" \
    --smoke --deadline "$QWEN3_JOB_DEADLINE"
  "$PYTHON" check_smoke.py "runs/$RUN_ID-smoke/summary.json" --deadline "$QWEN3_JOB_DEADLINE"
  touch "runs/$RUN_ID.shutdown-enabled"
  "$PYTHON" train.py --model-path "$MODEL" --run-dir "runs/$RUN_ID" \
    --deadline "$QWEN3_JOB_DEADLINE"
  echo TRAINING_COMPLETE
'
