#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ "$(uname -s)" == Linux && "$PWD" == /root/autodl-tmp/* ]] || {
  echo 'Run only on the remote AutoDL training host.' >&2; exit 2;
}
# Set these only after the new-round budget and platform timer are confirmed.
: "${QWEN3_APPROVED_BUDGET_CNY:?A new-round compute budget is required}"
: "${QWEN3_PRICE_CNY_PER_HOUR:?Verify the current hourly price}"
: "${QWEN3_HARD_DEADLINE:?Set an absolute deadline before the platform shutdown}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/root/autodl-tmp/qwen3-hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNSLOTH_MOE_BACKEND=unsloth_triton UNSLOTH_COMPILE_DISABLE=1
PYTHON="${QWEN3_VENV:-/root/qwen3-moe-venv}/bin/python"
RUN_ID="qwen3-round2-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p runs
printf '%s\n' "$RUN_ID" > runs/latest_round2_run.txt
printf '%s\n' "$RUN_ID" > runs/latest_run.txt
ln -sfn "$RUN_ID.log" runs/latest_round2.log
ln -sfn "$RUN_ID.log" runs/latest.log
exec > >(tee -a "runs/$RUN_ID.log") 2>&1
finish() {
  code=$?
  printf 'ROUND2_EXIT code=%s time=%s\n' "$code" "$(date -Iseconds)"
  sync
  /usr/bin/shutdown
}
trap finish EXIT
[[ -x "$PYTHON" ]] || { echo 'Existing verified venv is missing.' >&2; exit 2; }
DEADLINE="$("$PYTHON" - <<'PY'
import math, os, time
from common import load_config
c=load_config('config_round2.json')
budget=float(os.environ['QWEN3_APPROVED_BUDGET_CNY'])
price=float(os.environ['QWEN3_PRICE_CNY_PER_HOUR'])
if not all(math.isfinite(v) and v > 0 for v in [budget,price]):
    raise ValueError('Invalid budget or price')
if budget > c['max_compute_budget_cny'] or price > c['max_price_cny_per_hour']:
    raise ValueError('Budget/price exceeds the reviewed plan')
# Retain at least 30 minutes of compute allowance for launch and shutdown.
seconds=min(c['job_time_limit_seconds'],int(budget/price*3600)-1800)
print(min(int(time.time())+seconds,int(os.environ['QWEN3_HARD_DEADLINE'])))
PY
)"
JOB_SECONDS=$((DEADLINE - $(date +%s)))
(( JOB_SECONDS >= 1800 )) || { echo 'Less than 30 minutes remain.' >&2; exit 2; }
export QWEN3_JOB_DEADLINE="$DEADLINE" QWEN3_ACTIVE_RUN="$RUN_ID"
printf 'ROUND2_START id=%s deadline=%s time=%s\n' "$RUN_ID" "$DEADLINE" "$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /root/autodl-tmp
timeout --signal=TERM --kill-after=120s "${JOB_SECONDS}s" bash -Eeuo pipefail -c '
  PYTHON="${QWEN3_VENV:-/root/qwen3-moe-venv}/bin/python"
  MODEL=/root/autodl-tmp/qwen3-30b-a3b-bf16
  RUN_ID="$QWEN3_ACTIVE_RUN"
  "$PYTHON" prepare_data.py
  "$PYTHON" -m unittest -v test_experiment.py test_eval_full.py test_round2.py
  "$PYTHON" train_round2.py --model-path "$MODEL" --run-dir "runs/$RUN_ID-smoke" \
    --smoke --deadline "$QWEN3_JOB_DEADLINE"
  "$PYTHON" check_round2.py "runs/$RUN_ID-smoke/summary.json" --deadline "$QWEN3_JOB_DEADLINE"
  "$PYTHON" train_round2.py --model-path "$MODEL" --run-dir "runs/$RUN_ID" \
    --deadline "$QWEN3_JOB_DEADLINE"
  echo ROUND2_TRAINING_AND_FULL_EVAL_COMPLETE
'
