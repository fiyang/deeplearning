#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ "$(uname -s)" == Linux && "$PWD" == /root/autodl-tmp/* ]] || exit 2
SOURCE_RUN="${1:?Pass the completed training run ID}"
[[ "$SOURCE_RUN" =~ ^qwen3-moe-grpo-[0-9-]+$ ]] || exit 2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/qwen3-hf-cache
export UNSLOTH_MOE_BACKEND=unsloth_triton UNSLOTH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/root/qwen3-moe-venv/bin/python
RUN_ID="qwen3-full-eval-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "runs/$RUN_ID"
printf '%s\n' "$RUN_ID" > runs/latest_eval_run.txt
ln -sfn "$RUN_ID.log" runs/latest_eval.log
exec > >(tee -a "runs/$RUN_ID.log") 2>&1
finish() {
  code=$?
  printf 'EVAL_EXIT code=%s time=%s\n' "$code" "$(date -Iseconds)"
  sync
  if [[ "${AUTO_SHUTDOWN:-0}" == 1 ]]; then
    /usr/bin/shutdown
  fi
}
trap finish EXIT
printf 'EVAL_START id=%s source=%s time=%s\n' "$RUN_ID" "$SOURCE_RUN" "$(date -Iseconds)"
DEADLINE="$("$PYTHON" -c 'import os,time; print(min(int(time.time())+14400,int(os.environ["QWEN3_HARD_DEADLINE"])))')"
SECONDS_LEFT=$((DEADLINE - $(date +%s)))
(( SECONDS_LEFT >= 12000 )) || { echo 'Insufficient time for full paired evaluation'; exit 2; }
for PHASE in preflight baseline final aggregate; do
  SECONDS_LEFT=$((DEADLINE - $(date +%s)))
  (( SECONDS_LEFT > 0 )) || exit 124
  timeout --signal=TERM --kill-after=60s "${SECONDS_LEFT}s" "$PYTHON" eval_full.py \
    --source-run "runs/$SOURCE_RUN" --run-dir "runs/$RUN_ID" \
    --model-path /root/autodl-tmp/qwen3-30b-a3b-bf16 \
    --data-file ../qwen7b_grpo/data/gsm8k_test.jsonl \
    --phase "$PHASE" --deadline "$DEADLINE"
done
