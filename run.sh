#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN_DIR="${1:-runs/cifar10-cnn-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)}"
mkdir -p runs
"$PYTHON_BIN" -u prepare_data.py --data-dir data
"$PYTHON_BIN" -u train.py --data-dir data --output "$RUN_DIR" --epochs 8
"$PYTHON_BIN" -u predict.py --checkpoint "$RUN_DIR/best.pt" --image "$RUN_DIR/sample_test_image.png" > "$RUN_DIR/inference_check.json"
