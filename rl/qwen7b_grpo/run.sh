#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/root/autodl-tmp/hf-cache
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT_DIR="$PWD"
VENV=/root/autodl-tmp/qwen-grpo-venv
RUN_ID="qwen7b-grpo-$(date +%Y%m%d-%H%M%S)"
mkdir -p runs
printf '%s\n' "$RUN_ID" > runs/latest_run.txt
printf '%s\n' "$RUN_ID" > runs/latest_attempt.txt
exec > >(tee -a "runs/$RUN_ID.log") 2>&1
finish() {
  result=$?
  printf 'RUN_EXIT code=%s time=%s\n' "$result" "$(date -Iseconds)"
  sync
  # Opt in only on a disposable remote training instance.
  if [[ "${AUTO_SHUTDOWN:-0}" == 1 ]]; then /usr/bin/shutdown; fi
}
trap finish EXIT
echo "PREPARING $RUN_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /root/autodl-tmp
/root/miniconda3/bin/python -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --no-cache-dir -r requirements.txt
"$VENV/bin/python" -m pip freeze > "runs/$RUN_ID.environment.txt"
"$VENV/bin/python" -m unittest -v test_rewards.py
"$VENV/bin/python" prepare_data.py
"$VENV/bin/python" download_model.py
"$VENV/bin/python" experiment.py --model-path /root/autodl-tmp/qwen7b-grpo-model \
  --run-dir "$ROOT_DIR/runs/$RUN_ID-smoke" --smoke
"$VENV/bin/python" experiment.py --model-path /root/autodl-tmp/qwen7b-grpo-model \
  --run-dir "$ROOT_DIR/runs/$RUN_ID"
echo TRAINING_COMPLETE
