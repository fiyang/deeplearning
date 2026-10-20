#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
vit_python="${PYTHON_BIN:-/root/miniconda3/bin/python}"
vit_output="${1:-runs/vit-tiny12-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)}"
if [ "$#" -gt 0 ]; then shift; fi
"$vit_python" prepare_data.py
"$vit_python" -u train_vit.py --output "$vit_output" "$@"
