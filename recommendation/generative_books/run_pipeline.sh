#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python_bin="$(pwd)/../generative_beauty/.venv/bin/python"
mkdir -p runs
exec 9>runs/pipeline.lock
flock -n 9 || { printf '%s\n' 'Another Books pipeline is already active'; exit 1; }
stage_file="$(pwd)/runs/pipeline_stage.txt"
stage() { printf '%s %s\n' "$(TZ=Asia/Shanghai date -Iseconds)" "$1" | tee "$stage_file"; }
trap 'stage "FAILED line=$LINENO"' ERR
stage "waiting_for_download"
# The range downloader was started separately and writes the manifest only after hashing each file.
while ! "$python_bin" -c 'import json,pathlib; p=pathlib.Path("data/download_manifest.json"); d=json.loads(p.read_text()) if p.exists() else []; assert len(d)==2 and all(pathlib.Path("data/raw", x["file"]).stat().st_size == x["bytes"] for x in d)' 2>/dev/null; do
    if ! pgrep -f '^/root/miniconda3/bin/python -u download.py$' >/dev/null; then
        stage "download_resume"
        "$python_bin" -u download.py
    fi
    sleep 20
done
stage "unit_checks"
"$python_bin" -u test_pipeline.py
stage "prepare_full_books"
"$python_bin" -u prepare.py
stage "books_gpu_smoke"
bash start_train.sh --smoke
stage "formal_training"
bash start_train.sh --epochs 5 --batch-size 512 --eval-batch-size 64 --val-users 20000 --beam-size 20
stage "complete"
printf '%s\n' PIPELINE_COMPLETE
