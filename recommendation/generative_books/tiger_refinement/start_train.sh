#!/usr/bin/env bash
# Run on the existing Books machine; bounded training, no machine lifecycle changes.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p runs
exec 9>runs/pipeline.lock
flock -n 9 || { printf '%s\n' 'Another Books task is already active'; exit 1; }
python_bin="$(pwd)/../generative_beauty/.venv/bin/python"
parent="$(pwd)/runs/books-generative-20261009-153342-14533"
run_name="books-tiger-refinement-lr3e-5-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
run_dir="$(pwd)/runs/$run_name"
printf '%s\n' "$run_dir" > runs/latest_run.txt
printf '%s\n' "$run_dir" > runs/latest_tiger_refinement.txt
printf '%s %s\n' "$(TZ=Asia/Shanghai date -Iseconds)" tiger_refinement_training > runs/pipeline_stage.txt
set +e
timeout --signal=TERM --kill-after=30 5400 "$python_bin" -u tiger_refinement/train_refinement.py \
  --parent-run "$parent" --run-dir "$run_dir" --additional-epochs 3 --lr 0.00003 \
  --warmup-steps 500 --batch-size 512 --eval-batch-size 64 --patience 2 --seed 42 \
  2>&1 | tee "$run_dir.log"
code=${PIPESTATUS[0]}
set -e
printf 'RUN_EXIT code=%s time=%s\n' "$code" "$(TZ=Asia/Shanghai date -Iseconds)" | tee -a "$run_dir.log"
printf '%s\n' "$code" > "$run_dir.exit"
if [[ "$code" -eq 0 ]]; then
  printf '%s %s\n' "$(TZ=Asia/Shanghai date -Iseconds)" complete_tiger_refinement > runs/pipeline_stage.txt
else
  printf '%s failed_tiger_refinement_exit_%s\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$code" > runs/pipeline_stage.txt
fi
exit "$code"
