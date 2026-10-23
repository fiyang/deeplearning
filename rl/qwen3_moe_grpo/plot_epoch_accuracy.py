"""Plot measured full-test results and within-run validation separately."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, fontManager
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
font_path = os.environ.get("QWEN3_PLOT_FONT")
if font_path:
    fontManager.addfont(font_path)
    font_name = FontProperties(fname=font_path).get_name()
else:
    font_name = None
    for name in ["Arial Unicode MS", "Hiragino Sans GB", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei"]:
        try:
            findfont(FontProperties(family=name), fallback_to_default=False)
        except ValueError:
            continue
        font_name = name
        break
    if font_name is None:
        raise RuntimeError("Install a CJK font or set QWEN3_PLOT_FONT to a Chinese-capable font file.")
plt.rcParams.update({
    "font.family": font_name,
    "font.size": 12,
    "axes.unicode_minus": False,
    "svg.fonttype": "path",
})

reference = json.loads((ROOT / "reports/round1_full_eval.json").read_text())
current = json.loads((ROOT / "reports/round2_summary.json").read_text())
assert reference["completed"] and current["completed"]
assert reference["full"]["n"] == current["final"]["n"] == 1319
history = current["validation_history"]
assert [(r["epoch"], r["n"]) for r in history] == [(1, 256), (2, 256)]

navy, muted, blue, teal = "#172B4D", "#526377", "#246BCE", "#00857C"
fig, axes = plt.subplots(1, 2, figsize=(15.4, 8.3), sharey=True)
fig.patch.set_facecolor("#F5F8FC")
fig.subplots_adjust(left=.075, right=.96, bottom=.26, top=.73, wspace=.19)
fig.text(.075, .923, "训练轮数增加，准确率提升有多大？", color=navy, fontsize=25, weight="bold")
fig.text(.075, .865, "Qwen3-30B-A3B · BF16 LoRA + GRPO   |   rank 8 · alpha 16 · 峰值学习率 5e-6", color=muted, fontsize=12)

for ax in axes:
    ax.set_facecolor("white")
    ax.set_ylim(90, 96)
    ax.set_yticks(range(90, 97))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", color="#E2E8F0", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=12, colors=muted)
    for spine in ax.spines.values():
        spine.set_visible(False)

left, right = axes
left.set_title("完整测试：不同实验之间的对比", loc="left", pad=42, color=navy, fontsize=16, weight="bold")
left.text(0, 1.035, "同一套 1,319 题；数据量与轮数同时变化", transform=left.transAxes, color=muted, fontsize=11)
left.set_ylabel("准确率", color=muted, labelpad=12)
values = [reference["full"]["baseline_accuracy"] * 100, reference["full"]["final_accuracy"] * 100, current["final"]["accuracy"] * 100]
correct = [reference["full"]["baseline_correct"], reference["full"]["final_correct"], current["final"]["correct"]]
left.plot([0, 1, 2], values, color=blue, linewidth=2, linestyle=(0, (4, 4)), alpha=.65)
left.scatter([0, 1, 2], values, s=100, color=blue, zorder=3, edgecolors="white", linewidths=2)
left.set_xlim(-.35, 2.35)
left.set_xticks([0, 1, 2], ["原模型\n0 epoch", "实验 1 · 512 题\n1 epoch", "实验 2 · 1,024 题\n2 epochs"])
for x, value, count in zip([0, 1, 2], values, correct):
    left.annotate(f"{value:.2f}%", (x, value), xytext=(0, 18), textcoords="offset points", ha="center", color=blue, fontsize=17, weight="bold")
    left.annotate(f"{count:,} / 1,319", (x, value), xytext=(0, -24), textcoords="offset points", ha="center", color=muted, fontsize=10)
left.text(.52, 91.05, f"+{values[1]-values[0]:.2f} 个百分点", color=blue, ha="center", fontsize=11)
left.text(1.62, 92.08, f"+{values[2]-values[1]:.2f} 个百分点", color=blue, ha="center", fontsize=11)

right.set_title("验证集：同一次实验内增加轮数", loc="left", pad=42, color=navy, fontsize=16, weight="bold")
right.text(0, 1.035, "固定 1,024 道训练题；同一套 256 题验证集", transform=right.transAxes, color=muted, fontsize=11)
val = [row["accuracy"] * 100 for row in history]
right.plot([1, 2], val, color=teal, linewidth=2.5)
right.scatter([1, 2], val, s=100, color=teal, zorder=3, edgecolors="white", linewidths=2)
right.set_xlim(.55, 2.45)
right.set_xticks([1, 2], ["第 1 个 epoch\n累计 256 步", "第 2 个 epoch\n累计 512 步"])
for x, row in zip([1, 2], history):
    right.annotate(f"{row['accuracy']*100:.2f}%", (x, row["accuracy"] * 100), xytext=(0, 18), textcoords="offset points", ha="center", color=teal, fontsize=17, weight="bold")
    right.annotate(f"{row['correct']} / 256", (x, row["accuracy"] * 100), xytext=(0, -25), textcoords="offset points", ha="center", color=muted, fontsize=10)
right.text(1.5, 92.45, f"多训练 1 个 epoch\n多答对 1 题 · +{val[1]-val[0]:.2f} 个百分点", ha="center", va="center", color=teal, fontsize=16, linespacing=1.8,
           bbox=dict(boxstyle="round,pad=.85", facecolor="#EDF8F5", edgecolor="none"))

fig.text(.075, .133, "读图提示：左图不能单独归因于轮数；右图显示本次第 2 个 epoch 的验证收益较小。", color=navy, fontsize=12)
fig.text(.075, .085, "纵轴统一放大至 90%–96%。测试集与验证集不同，分数不可直接横向比较；连线仅连接已测结果，不表示中间值。", color=muted, fontsize=10)
fig.text(.075, .045, "来源：已完成实验的 summary.json；单次随机种子结果。第二次实验第 1 个 epoch 未评估完整测试集。", color=muted, fontsize=10)

out = ROOT / "figures"
out.mkdir(exist_ok=True)
for suffix in ["png", "svg"]:
    path = out / f"epochs_accuracy_20261023.{suffix}"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    if suffix == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(path)
plt.close(fig)
