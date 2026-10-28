#!/usr/bin/env bash
# Bounded continuation on the existing instance. Keep the machine running.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p runs
exec 9>runs/pipeline.lock
flock -n 9 || { printf '%s\n' 'Another Books task is already active'; exit 1; }
python_bin="$(pwd)/../generative_beauty/.venv/bin/python"
parent="$(pwd)/runs/books-tiger-growth-w256-lr3e-5-20261028-103055-5734"
run_dir="$(pwd)/runs/books-tiger-wide-continue5-lr3e-5-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
printf '%s\n' "$run_dir" > runs/latest_run.txt
printf '%s\n' "$run_dir" > runs/latest_tiger_continuation.txt
printf '%s %s\n' "$(TZ=Asia/Shanghai date -Iseconds)" tiger_wide_continuation > runs/pipeline_stage.txt
set +e
timeout --signal=TERM --kill-after=30 10800 "$python_bin" -u tiger_continuation/train_continuation.py \
  --parent-run "$parent" --run-dir "$run_dir" --additional-epochs 5 --lr 0.00003 \
  --warmup-steps 500 --batch-size 512 --eval-batch-size 64 --patience 2 --seed 42 \
  2>&1 | tee "$run_dir.log"
code=${PIPESTATUS[0]}
set -e
printf 'RUN_EXIT code=%s time=%s\n' "$code" "$(TZ=Asia/Shanghai date -Iseconds)" | tee -a "$run_dir.log"
printf '%s\n' "$code" > "$run_dir.exit"
if [[ "$code" -eq 0 ]]; then
  printf '%s %s\n' "$(TZ=Asia/Shanghai date -Iseconds)" complete_tiger_wide_continuation > runs/pipeline_stage.txt
else
  printf '%s failed_tiger_wide_continuation_exit_%s\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$code" > runs/pipeline_stage.txt
fi
exit "$code"
