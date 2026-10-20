#!/usr/bin/env bash
# 两组实验按顺序运行：384 维 × 300 轮，然后 192 维 × 600 轮。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
vit_project="$PWD"
vit_python="${PYTHON_BIN:-/root/miniconda3/bin/python}"
if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    exec "$vit_python" run_vit_scaling.py --dry-run "$@"
fi
command -v screen >/dev/null 2>&1 || { printf '未找到 screen。\n' >&2; exit 1; }
vit_name="vit-scaling-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
vit_output="$vit_project/runs/$vit_name"
mkdir -p "$vit_project/runs"
screen -dmS "$vit_name" bash -c '
    vit_python=$1
    vit_output=$2
    shift 2
    if "$vit_python" -u run_vit_scaling.py --output "$vit_output" "$@" > "$vit_output.log" 2>&1; then
        vit_status=0
    else
        vit_status=$?
    fi
    printf "%s\n" "$vit_status" > "$vit_output.exit"
    exit "$vit_status"
' _ "$vit_python" "$vit_output" "$@"
sleep 1
if [[ -f "$vit_output.exit" ]]; then
    vit_status="$(cat "$vit_output.exit")"
    tail -n 25 "$vit_output.log"
    exit "$vit_status"
fi
printf '实验队列已提交，两组从头训练按顺序运行。\n'
printf '总日志：tail -F "%s.log"\n' "$vit_output"
printf '第一组：tail -F "%s/width384-300epochs.log"\n' "$vit_output"
printf '第二组：tail -F "%s/width192-600epochs.log"\n' "$vit_output"
printf '结果：%s/comparison.json\n' "$vit_output"
