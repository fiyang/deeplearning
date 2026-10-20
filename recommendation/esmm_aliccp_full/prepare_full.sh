#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
data_dir="${1:-/root/autodl-tmp/deeplearning-lab/recommendation/data/aliccp-full}"
g++ -O3 -std=c++17 prepare.cpp -o prepare
mkdir -p "$data_dir/cache"
for split in train test; do
  raw="$data_dir/raw/$split"
  /root/miniconda3/bin/python - "$raw/extraction.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text());assert len(d['files'])==2
for f in d['files']:assert (p.parent/f['name']).stat().st_size==f['bytes']
PY
  if [[ ! -f "$data_dir/cache/$split/manifest.json" ]]; then
    ./prepare "$raw/common_features_$split.csv" "$raw/sample_skeleton_$split.csv" "$data_dir/cache/$split"
  fi
done
