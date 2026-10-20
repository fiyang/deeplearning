#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
base=/root/autodl-tmp/deeplearning-lab/recommendation
reference="$base/runs/esmm-aliccp-full-20261007-133218-3374"
prepared="$base/data/aliccp-aitm-author-seed42-v1"
run_id="aitm-author-$(date +%Y%m%d-%H%M%S)-$$"
run_dir="$base/runs/$run_id"
log="$run_dir.log"
mkdir -p "$base/runs"
screen -dmS "$run_id" bash -c '
  set -euo pipefail
  {
    if [ ! -f "$1/manifest.json" ]; then
      /root/miniconda3/bin/python -u prepare.py --reference-run "$2" --output "$1"
    fi
    /root/miniconda3/bin/python -u train.py --prepared "$1" --output "$3"
  } 2>&1 | tee "$4"
' _ "$prepared" "$reference" "$run_dir" "$log"
printf 'RUN=%s\nLOG=%s\n' "$run_dir" "$log"
