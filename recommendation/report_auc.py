"""Create a compact figure and a per-trial table from a completed AUC experiment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    first_seed = protocol["seeds"][0]
    labels = {"mlp": "Original MLP", "interaction": "MLP + interactions", "deepfm": "DeepFM"}
    colors = {"mlp": "#3569B5", "interaction": "#CB8435", "deepfm": "#168572"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for architecture in ["mlp", "interaction", "deepfm"]:
        trial = f"{architecture}/seed{first_seed}"
        with (args.run / trial / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        x = [int(row["epoch"]) for row in rows]
        y = [float(row["val_auc"]) for row in rows]
        axes[0].plot(x, y, color=colors[architecture], label=labels[architecture], linewidth=2)
        best = summary["results"][trial]
        axes[0].scatter([best["best_auc_epoch"]], [best["best_validation_auc"]], color=colors[architecture], s=40, zorder=5)
    axes[0].set(title=f"Validation AUC (seed {first_seed})", xlabel="Epoch", ylabel="AUC")
    axes[0].legend(loc="lower right", frameon=False)
    seeds = protocol["seeds"]
    for architecture in summary["validation_selection"]["confirmation_families"]:
        values = [summary["results"][f"{architecture}/seed{seed}"]["test"]["auc"] for seed in seeds]
        axes[1].plot(seeds, values, marker="o", color=colors[architecture], label=labels[architecture], linewidth=2)
        for seed, value in zip(seeds, values):
            axes[1].annotate(f"{value:.4f}", (seed, value), xytext=(0, 8 if architecture != "mlp" else -15),
                             textcoords="offset points", ha="center", color=colors[architecture], fontsize=9)
    axes[1].set(title="Test AUC across confirmation seeds", xlabel="Training seed", ylabel="AUC", xticks=seeds)
    axes[1].margins(x=.12, y=.25)
    axes[1].legend(loc="best", frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.suptitle("MovieLens-1M: AUC-selected model comparison", fontsize=15, fontweight="bold", y=.99)
    fig.text(.5, .015, "Same temporal split | 75,930 test ratings | Checkpoints selected before test | Exploratory reused test set",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, .05, 1, .94])
    fig.savefig(args.run / "auc_comparison.png", dpi=180)
    plt.close(fig)
    records = []
    for trial, result in summary["results"].items():
        records.append(dict(trial=trial, architecture=result["architecture"], seed=result["seed"],
                            best_auc_epoch=result["best_auc_epoch"], completed_epochs=result["completed_epochs"],
                            parameter_count=result["parameter_count"], validation_auc=result["validation"]["auc"],
                            test_auc=result["test"]["auc"], test_accuracy=result["test"]["accuracy_at_0_5"],
                            test_ndcg_at_10=result["test"]["ndcg_at_10"], test_log_loss=result["test"]["log_loss"]))
    with (args.run / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"REPORT_READY {args.run / 'auc_comparison.png'}")


if __name__ == "__main__":
    main()
