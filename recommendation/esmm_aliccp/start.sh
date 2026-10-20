#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin=python3; fi
run_dir="${ALICCP_RUN_DIR:-$PWD/../runs/esmm-aliccp-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$}"
mkdir -p -- "$(dirname -- "$run_dir")"
"$python_bin" -m unittest -v test_aliccp
"$python_bin" data.py
echo "Run: $run_dir"
echo "Log: ${run_dir}.log"
printf '%s\n' "$run_dir" > current_run.txt
"$python_bin" -u train.py --output "$run_dir" "$@" 2>&1 | tee "${run_dir}.log"
"$python_bin" -u verify.py --run "$run_dir" 2>&1 | tee -a "${run_dir}.log"
"$python_bin" -u report.py --run "$run_dir" 2>&1 | tee -a "${run_dir}.log"
echo "EXPERIMENT_COMPLETE $run_dir" | tee -a "${run_dir}.log"
