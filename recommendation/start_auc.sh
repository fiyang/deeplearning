#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi
run_dir="${RECOMMENDATION_RUN_DIR:-$PWD/runs/auc-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$}"
mkdir -p -- "$(dirname -- "$run_dir")"
"$python_bin" prepare_data.py
"$python_bin" -m unittest test_features test_auc
echo "Run: $run_dir"
echo "Log: ${run_dir}.log"
"$python_bin" -u train_auc.py --output "$run_dir" "$@" 2>&1 | tee "${run_dir}.log"
