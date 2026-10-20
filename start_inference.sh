#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
inference_python="$PWD/.venv-inference/bin/python"
inference_checkpoint="${1:-$PWD/runs/vit-scaling-20261006-200157-1726/width192-600epochs/best.pt}"
inference_port="${PORT:-8008}"
inference_name="vit-inference-${inference_port}-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
inference_log="$PWD/runs/$inference_name.log"
test -x "$inference_python"
test -f "$inference_checkpoint"
"$inference_python" - "$inference_port" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
screen -dmS "$inference_name" bash -c 'exec "$1" -u inference_service.py --checkpoint "$2" --port "$3" > "$4" 2>&1' _ "$inference_python" "$inference_checkpoint" "$inference_port" "$inference_log"
"$inference_python" - "$inference_port" "$inference_log" <<'PY'
import json, sys, time, urllib.request
for _ in range(60):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/health', timeout=1) as response:
            health=json.load(response)
        print(json.dumps(health, indent=2))
        print('Log:', sys.argv[2])
        break
    except (OSError, ValueError):
        time.sleep(.5)
else:
    raise SystemExit('Service not ready; inspect ' + sys.argv[2])
PY
