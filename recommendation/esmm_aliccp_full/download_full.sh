#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="/root/miniconda3/bin/python"
data_dir="/root/autodl-tmp/deeplearning-lab/recommendation/data/aliccp-full"
mkdir -p "$data_dir/archives"
if [[ -f "$data_dir/archives/sample_train.tar.gz" && -f "$data_dir/archives/sample_test.tar.gz" ]]; then
  "$python_bin" -u download.py --data-dir "$data_dir" --extract-existing
else
  "$python_bin" -u download.py --data-dir "$data_dir" --urls "${1:-/tmp/aliccp-links.json}"
fi
