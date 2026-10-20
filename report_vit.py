"""将新 ViT 的完整实验与之前的 ViT、ResNet 结果对照。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    configs = [("Old ViT (6 layers)", args.baseline / "vit"),
               ("New ViT (12 layers)", args.run), ("ResNet-18", args.baseline / "resnet18")]
    summaries = [read_json(path / "summary.json") for _, path in configs]
    assert len({row["split_sha256"] for row in summaries}) == 1
    config = read_json(args.run / "config.json")
    history = list(csv.DictReader((args.run / "metrics.csv").open()))
    matrix = np.loadtxt(args.run / "confusion_matrix.csv", delimiter=",", dtype=int)
    assert len(history) == config["epochs"]
    assert [int(row["epoch"]) for row in history] == list(range(1, config["epochs"] + 1))
    assert abs(max(float(row["val_accuracy"]) for row in history) - summaries[1]["best_validation_accuracy"]) < 1e-12
    assert matrix.sum() == 10000 and np.trace(matrix) == summaries[1]["test_correct"]
    assert np.isfinite(np.load(args.run / "sample_features.npy")).all()
    assert read_json(args.run / "inference_check.json")["features_finite"]
    result = {"note": "Same split and seed, different model/training recipe/epochs/precision; not a controlled depth ablation.",
        "runs": [{"label": label, "path": str(path), **row}
                 for (label, path), row in zip(configs, summaries)],
        "test_gain_percentage_points": (summaries[1]["test_accuracy"] - summaries[0]["test_accuracy"]) * 100,
        "additional_test_correct": summaries[1]["test_correct"] - summaries[0]["test_correct"]}
    (args.run / "comparison_to_baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for (label, path), color in zip(configs[:2], ["#8c9dac", "#2977c9"]):
        rows = list(csv.DictReader((path / "metrics.csv").open()))
        axes[0].plot([int(row["epoch"]) for row in rows],
                     [100 * float(row["val_accuracy"]) for row in rows], label=label, color=color)
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)", ylim=(0, 100), title="Validation selects the checkpoint")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="lower right")
    positions = np.arange(3)
    bars = axes[1].bar(positions, [100 * row["test_accuracy"] for row in summaries],
                       color=["#8c9dac", "#2977c9", "#52937c"], width=0.6)
    axes[1].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[1].set(xticks=positions,
        xticklabels=[f"{label}\n{row['epochs']} epochs" for (label, _), row in zip(configs, summaries)],
        ylabel="Held-out test accuracy (%)", ylim=(0, 100), title="Different training budgets and recipes")
    fig.suptitle("CIFAR-10 | From scratch | Same train / validation / test split | Seed 42")
    fig.savefig(args.run / "comparison_to_baseline.png", dpi=170)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("RESULT_ARTIFACTS_VERIFIED")


if __name__ == "__main__":
    main()
