"""Plot actual training curves and export a small comparison table."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    summary = json.loads((root / "summary.json").read_text())
    records = [dict(name="Existing DeepFM", **summary["existing_deepfm"])]
    for name in ["mean", "transformer"]:
        records.append(dict(name=name, **summary["results"][name]))
    table = [dict(model=r["name"], validation_auc=r["validation"]["auc"], test_auc=r["test"]["auc"],
                  test_ndcg_at_10=r["test"]["ndcg_at_10"], test_accuracy=r["test"]["accuracy_at_0_5"],
                  test_log_loss=r["test"]["log_loss"], best_epoch=r.get("best_auc_epoch")) for r in records]
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = {"mean": "#637b91", "transformer": "#167f70"}
    for name in ["mean", "transformer"]:
        rows = list(csv.DictReader((root / name / "metrics.csv").open()))
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["val_auc"]) for row in rows], marker="o", ms=4, color=colors[name], label=name.capitalize())
        axes[1].plot(epochs, [float(row["train_loss"]) for row in rows], marker="o", ms=4, color=colors[name], label=name.capitalize())
    axes[0].axhline(summary["existing_deepfm"]["validation"]["auc"], ls="--", color="#ba7830", label="Existing DeepFM")
    axes[0].set(title="Validation AUC: higher is better", xlabel="Fine-tuning epoch", ylabel="AUC")
    axes[1].set(title="Training loss", xlabel="Fine-tuning epoch", ylabel="Binary cross-entropy")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
    fig.suptitle("MovieLens-1M | 50-event histories, 64-d interest features\nSame inputs and pretrained DeepFM; seed 42", fontsize=14)
    fig.savefig(root / "sequence_training.png", dpi=170)
    fig.savefig(root / "sequence_training.pdf")
    plt.close(fig)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
