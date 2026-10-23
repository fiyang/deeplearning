#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ "$(uname -s)" == Linux && "$PWD" == /root/autodl-tmp/* ]] || {
  echo 'Install only on the remote AutoDL training host.' >&2; exit 2;
}
export PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false
VENV="${QWEN3_VENV:-/root/qwen3-moe-venv}"
/root/miniconda3/bin/python -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip --index-url https://pypi.org/simple
"$VENV/bin/python" -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" -m pip install -r requirements.txt --index-url https://pypi.org/simple
"$VENV/bin/python" -m pip check
mkdir -p runs
"$VENV/bin/python" -m pip freeze > runs/environment-resolved.txt
"$VENV/bin/python" -c 'from unsloth import FastLanguageModel; from trl import GRPOTrainer, GRPOConfig; import torch; print("IMPORT_OK", torch.__version__, torch.cuda.get_device_name())'
