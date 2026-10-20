"""相同数据、轮数和优化器下比较 CNN / ResNet / Vision Transformer。"""
import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models import MODEL_SPECS, build_model
from train import CLASSES, MEAN, STD, evaluation_transform, seed_worker, stratified_split
from train import run_epoch, final_test, save_figures, plt


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def run_model(name, args):
    output = args.output / name
    output.mkdir()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    augment = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_data = datasets.CIFAR10(str(args.data_dir), train=True, transform=augment, download=False)
    val_data = datasets.CIFAR10(str(args.data_dir), train=True, transform=evaluation_transform(), download=False)
    train_indices, val_indices = stratified_split(train_data.targets, args.seed)
    split = {"train": train_indices, "validation": val_indices}
    json_write(output / "split_indices.json", split)
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                   persistent_workers=args.workers > 0, worker_init_fn=seed_worker)
    training = DataLoader(Subset(train_data, train_indices), shuffle=True,
                         generator=torch.Generator().manual_seed(args.seed), **options)
    validation = DataLoader(Subset(val_data, val_indices), shuffle=False, **options)
    device = torch.device("cuda")
    model = build_model(name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    config = {
        "model": name, "model_spec": MODEL_SPECS[name], "epochs": args.epochs,
        "batch_size": args.batch_size, "workers": args.workers, "lr": args.lr,
        "weight_decay": args.weight_decay, "seed": args.seed, "optimizer": "AdamW",
        "adamw_betas": [0.9, 0.999], "adamw_eps": 1e-8,
        "scheduler": "CosineAnnealingLR", "T_max": args.epochs, "eta_min": 0,
        "loss": "CrossEntropyLoss", "label_smoothing": 0, "early_stopping": False,
        "pretrained": False, "precision": "float32", "mean": MEAN, "std": STD,
        "augmentation": {"random_crop": 32, "padding": 4, "horizontal_flip_probability": 0.5},
        "train_count": len(train_indices), "validation_count": len(val_indices), "test_count": 10000,
        "split_sha256": split_hash, "parameter_count": sum(p.numel() for p in model.parameters()),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    json_write(output / "config.json", config)
    print(f"START {name} | parameters={config['parameter_count']:,} | epochs={args.epochs}", flush=True)
    baseline_loss, baseline_accuracy = run_epoch(model, validation, criterion, device)
    history, best_accuracy, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = run_epoch(model, training, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, validation, criterion, device)
        scheduler.step()
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError(f"Non-finite loss: {name}, epoch {epoch}")
        history.append(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
            val_loss=val_loss, val_accuracy=val_accuracy, lr=lr, seconds=time.perf_counter() - tick))
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[-1]))
            writer.writeheader()
            writer.writerows(history)
        print(f"{name} {epoch:02d}/{args.epochs} | train={train_accuracy:.2%}, "
              f"val={val_accuracy:.2%}, loss={val_loss:.4f} | {history[-1]['seconds']:.1f}s", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save({"model_state": model.state_dict(), "classes": CLASSES, "config": config,
                "best_epoch": epoch, "validation_accuracy": val_accuracy}, output / "best.pt")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_data = datasets.CIFAR10(str(args.data_dir), train=False, transform=evaluation_transform(), download=False)
    testing = DataLoader(test_data, shuffle=False, **options)
    predicted, labels, confidences, confusion, test_loss = final_test(model, testing, device)
    summary = {"model": name, "parameter_count": config["parameter_count"], "epochs": args.epochs, "seed": args.seed,
        "split_sha256": split_hash, "baseline_validation_accuracy": baseline_accuracy,
        "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
        "test_accuracy": float(np.trace(confusion) / confusion.sum()), "test_loss": test_loss,
        "test_correct": int(np.trace(confusion)), "test_count": int(confusion.sum()),
        "training_epoch_seconds": sum(row["seconds"] for row in history),
        "elapsed_seconds_before_plots": time.perf_counter() - started,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "per_class_accuracy": {label: float(confusion[i, i] / confusion[i].sum()) for i, label in enumerate(CLASSES)},
        "note": "One seed, shared fixed recipe; validation selects checkpoint; each model tested once. Not best-tuned architectures."}
    save_figures(output, history, test_data, predicted, labels, confidences, confusion, MODEL_SPECS[name]["label"])
    np.savetxt(output / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    summary["total_elapsed_seconds"] = time.perf_counter() - started
    json_write(output / "summary.json", summary)
    # 独立进程加载模型，验证 checkpoint 与特征提取；结束当前进程占用的模型内存。
    del model, optimizer, scheduler, checkpoint, training, validation, testing
    gc.collect()
    torch.cuda.empty_cache()
    with (output / "inference_check.json").open("w") as handle:
        subprocess.run([sys.executable, "infer_model.py", "--checkpoint", str(output / "best.pt"),
            "--image", str(output / "sample_test_image.png"), "--features-output", str(output / "sample_features.npy")],
            stdout=handle, check=True)
    check = json.loads((output / "inference_check.json").read_text())
    if check["top3"][0]["class"] != CLASSES[predicted[0]] or not check["features_finite"]:
        raise RuntimeError("Reload inference or feature check failed")
    print("FINAL " + json.dumps(summary), flush=True)
    return summary


def save_comparison(output, summaries):
    assert len({entry["split_sha256"] for entry in summaries}) == 1
    best = max(summaries, key=lambda entry: entry["best_validation_accuracy"])
    json_write(output / "comparison.json", {"selection_rule": "Highest validation accuracy",
        "selected_model": best["model"], "runs": summaries})
    columns = ["model", "epochs", "parameter_count", "best_epoch", "best_validation_accuracy",
               "test_accuracy", "training_epoch_seconds", "total_elapsed_seconds", "peak_gpu_allocated_mib"]
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    positions = np.arange(len(summaries))
    for delta, field, color, label in [(-0.18, "best_validation_accuracy", "#2977c9", "Validation"),
                                      (0.18, "test_accuracy", "#e38d2c", "Test")]:
        bars = axes[0].bar(positions + delta, [entry[field] * 100 for entry in summaries],
                           width=0.36, color=color, label=label)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    names = [MODEL_SPECS[entry["model"]]["label"] for entry in summaries]
    axes[0].set(xticks=positions, xticklabels=names, ylim=(0, 100), ylabel="Accuracy (%)")
    axes[0].legend(loc="lower right")
    seconds = [entry["training_epoch_seconds"] for entry in summaries]
    bars = axes[1].bar(positions, seconds, color="#52937c")
    axes[1].bar_label(bars, fmt="%.1f s", padding=3)
    axes[1].set(xticks=positions, xticklabels=names, ylabel="Train + validation epoch time (seconds)",
                ylim=(0, max(seconds) * 1.2))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"CIFAR-10 | {summaries[0]['epochs']} epochs | Shared recipe | Seed {summaries[0]['seed']} | From scratch")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(f"COMPARISON_UPDATED completed={len(summaries)} selected_by_validation={best['model']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=["smallcnn", "resnet18", "vit"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or len(set(args.models)) != len(args.models):
        parser.error("Invalid dimensions, epochs, workers, or repeated model names")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("lr must be positive and weight-decay nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for name in ["models.py", "compare_models.py", "train.py", "infer_model.py"]:
        shutil.copy2(Path(__file__).parent / name, source / name)
    summaries = []
    for name in args.models:
        summaries.append(run_model(name, args))
        save_comparison(args.output, summaries)
    print("ALL_MODELS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
