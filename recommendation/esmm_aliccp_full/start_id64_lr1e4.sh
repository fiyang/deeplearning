#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('../data/aliccp-full')
for split in ['train','test']:
    assert (root/'cache'/split/'manifest.json').is_file(),f'{split} cache not ready'
    assert json.loads((root/f'verification-{split}.json').read_text())['status']=='passed'
PY
run_id="esmm-id64-lr1e-4-$(date +%Y%m%d-%H%M%S)-$$"
run_dir="/root/autodl-tmp/deeplearning-lab/recommendation/runs/$run_id"
log="$run_dir.log"
reference="/root/autodl-tmp/deeplearning-lab/recommendation/runs/esmm-id64-lr3e-4-20261007-174558-11340"
mkdir -p "$(dirname "$run_dir")"
screen -dmS "$run_id" bash -c 'set -euo pipefail; /root/miniconda3/bin/python -u train.py --output "$1" --embedding-dim 18 --id-embedding-dim 64 --lr 0.0001 --epochs 10 --patience 3 --batch-size 4096 --reference-run "$3" --comparison-change lr 2>&1 | tee "$2"' _ "$run_dir" "$log" "$reference"
printf 'RUN=%s\nLOG=%s\n' "$run_dir" "$log"
