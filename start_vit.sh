#!/usr/bin/env bash
# 从头启动一轮 ViT 训练。使用 screen 后台运行，退出 SSH 不影响训练。
set -euo pipefail

vit_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$vit_project"
vit_epochs=300
vit_model=vit_tiny12
vit_dry_run=0
vit_python="${PYTHON_BIN:-/root/miniconda3/bin/python}"

vit_usage() {
    cat <<'HELP'
用法：bash start_vit.sh [--model vit_tiny12|vit_small12] [--epochs 300] [--dry-run]

默认：12 层 ViT，从头训练 300 轮，预热 5 轮，其余参数沿用 train_vit.py。
  --epochs N   设置总训练轮数，必须大于 5。
  --model NAME 选择 192 维 vit_tiny12 或 384 维 vit_small12，默认前者。
  --dry-run    仅检查启动配置，不启动训练、不创建结果文件。
  -h, --help   显示帮助。

日志：runs/<本次名称>.log
结束状态：runs/<本次名称>.exit（0 表示成功，非 0 表示失败）
结果：runs/<本次名称>/best.pt、config.json、metrics.csv 等。
现有 checkpoint 仅含权重，本脚本不提供断点续训。
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [[ $# -lt 2 ]]; then
                printf '错误：--model 后需要模型名称。\n' >&2
                exit 2
            fi
            vit_model="$2"
            shift 2
            ;;
        --epochs)
            if [[ $# -lt 2 ]]; then
                printf '错误：--epochs 后需要一个轮数。\n' >&2
                exit 2
            fi
            vit_epochs="$2"
            shift 2
            ;;
        --dry-run) vit_dry_run=1; shift ;;
        -h|--help) vit_usage; exit 0 ;;
        *) printf '未知参数：%s\n' "$1" >&2; vit_usage >&2; exit 2 ;;
    esac
done

if [[ "$vit_model" != "vit_tiny12" && "$vit_model" != "vit_small12" ]]; then
    printf '错误：模型名称必须是 vit_tiny12 或 vit_small12。\n' >&2
    exit 2
fi
if [[ ! "$vit_epochs" =~ ^[1-9][0-9]{0,5}$ ]] || (( vit_epochs <= 5 )); then
    printf '错误：训练轮数必须是大于 5 的整数（最多六位）。\n' >&2
    exit 2
fi
if ! command -v screen >/dev/null 2>&1; then
    printf '错误：未找到 screen，无法后台启动。\n' >&2
    exit 1
fi
if [[ ! -x "$vit_python" || ! -f run_vit.sh || ! -f train_vit.py ]]; then
    printf '错误：请检查 Python 路径和项目内的 run_vit.sh、train_vit.py。\n' >&2
    exit 1
fi

# 时间戳加进程号，避免多次启动覆盖已有日志或模型。
vit_name="${vit_model//_/-}-${vit_epochs}epochs-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
vit_output="$vit_project/runs/$vit_name"
printf '模型：%s；训练轮数：%s；预热：5 轮；从随机权重开始\n结果目录：%s\n' "$vit_model" "$vit_epochs" "$vit_output"
if (( vit_dry_run )); then
    printf '启动配置检查通过，未启动训练。\n'
    exit 0
fi
if [[ -e "$vit_output" || -e "$vit_output.log" || -e "$vit_output.exit" ]]; then
    printf '错误：本次结果路径已存在，请重新启动以生成新名称。\n' >&2
    exit 1
fi
mkdir -p "$vit_project/runs"
printf 'ViT 后台训练：%s\n' "$vit_name" > "$vit_output.log"

screen -dmS "$vit_name" bash -c '
    vit_output=$1
    shift
    if bash run_vit.sh "$vit_output" "$@" >> "$vit_output.log" 2>&1; then
        vit_status=0
    else
        vit_status=$?
    fi
    printf "%s\n" "$vit_status" > "$vit_output.exit"
    exit "$vit_status"
' _ "$vit_output" --model "$vit_model" --epochs "$vit_epochs" --warmup-epochs 5

# 捕捉路径或解释器等导致的立即失败；正式训练状态仍以日志和 .exit 为准。
sleep 1
if [[ -f "$vit_output.exit" ]]; then
    vit_status="$(cat "$vit_output.exit")"
    printf '训练进程已结束，退出码：%s\n' "$vit_status"
    tail -n 20 "$vit_output.log"
    exit "$vit_status"
fi
printf '\n后台任务已启动，断开 SSH 后仍会运行。\n'
printf '查看日志：\n  tail -F "%s.log"\n' "$vit_output"
printf '查看会话：\n  screen -r "%s"\n' "$vit_name"
printf '结束后查看退出码：\n  cat "%s.exit"\n' "$vit_output"
printf '按 Ctrl+C 退出日志查看不会停止训练。\n'
