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
run_id="esmm-field-attn64-h4-lr1e-4-$(date +%Y%m%d-%H%M%S)-$$"
run_dir="/root/autodl-tmp/deeplearning-lab/recommendation/runs/$run_id"
log="$run_dir.log"
reference="/root/autodl-tmp/deeplearning-lab/recommendation/runs/esmm-id64-lr1e-4-20261007-185351-13286"
data_dir="/root/autodl-tmp/deeplearning-lab/recommendation/data/aliccp-full"
bundle="$run_dir.launch"
# Freeze the source now so later edits do not alter the queued experiment.
/root/miniconda3/bin/python - "$bundle" <<'PY'
from pathlib import Path
import shutil
import sys
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
for file in Path('.').iterdir():
    if file.is_file() and file.suffix in ['.py','.sh','.md']:
        shutil.copy2(file,out/file.name)
PY
screen -dmS "$run_id" bash -c 'set -euo pipefail; /root/miniconda3/bin/python -u "$1/run_when_ready.py" --reference-run "$2" --output "$3" --data-dir "$4" 2>&1 | tee "$5"' _ "$bundle" "$reference" "$run_dir" "$data_dir" "$log"
printf 'RUN=%s\nLOG=%s\nQUEUE=%s.queue.json\n' "$run_dir" "$log" "$run_dir"
