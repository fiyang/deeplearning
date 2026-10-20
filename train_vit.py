"""CIFAR-10 ViT：可选择 192/384 维，warmup + cosine、RandAugment + MixUp/CutMix。"""
import argparse
import csv
import gc
import hashlib
import json
import math
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
from torchvision.transforms import v2

from models import MODEL_SPECS, build_model
from train import CLASSES, MEAN, STD, evaluation_transform, seed_worker, stratified_split
from train import run_epoch, final_test, save_figures, plt


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def learning_rate(step, total_steps, warmup_steps, peak, start=1e-6, minimum=1e-5):
    """按每次参数更新调节：前五轮线性预热，末次更新到达最低学习率。"""
    if step < warmup_steps:
        return start + (peak - start) * step / max(1, warmup_steps - 1)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def mixing_transform():
    # 每个已打乱的 batch 随机选择一种混合方法，同时正确混合标签。
    return v2.RandomChoice([v2.MixUp(alpha=0.8, num_classes=10),
                            v2.CutMix(alpha=1.0, num_classes=10)])


def training_epoch(model, loader, optimizer, mixer, device, epoch, args):
    model.train()
    total_loss = torch.zeros((), device=device)
    target_mass = torch.zeros((), device=device)
    count = 0
    steps = len(loader)
    lr_first = None
    for index, (images, labels) in enumerate(loader):
        lr = learning_rate((epoch - 1) * steps + index, args.epochs * steps,
                           args.warmup_epochs * steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if lr_first is None:
            lr_first = lr
        # 在 CPU 上混合；随后送入 GPU。标签是 [batch, 10] 的概率分布。
        images, targets = mixer(images, labels)
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=args.precision == "bf16"):
            scores = model(images)
        # 概率标签与平滑损失用 FP32 计算，避免 BF16 舍入影响该目标函数。
        loss = nn.functional.cross_entropy(scores.float(), targets, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach() * len(labels)
        # 该量仅用于诊断混合标签，不能解释为普通图片分类准确率。
        target_mass += targets.gather(1, scores.detach().argmax(1, keepdim=True)).sum()
        count += len(labels)
    return total_loss.item() / count, target_mass.item() / count, lr_first, lr


def plot_history(output, history, model_name="vit_tiny12"):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train: mixed + smoothed targets")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation: hard targets")
    axes[0].set(ylabel="Loss (different training/eval objectives)")
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, [100 * row["val_accuracy"] for row in history], label="Validation (5,000)")
    clean = [row for row in history if row["clean_train_accuracy"] is not None]
    axes[1].plot([row["epoch"] for row in clean], [100 * row["clean_train_accuracy"] for row in clean],
                 marker="o", label="Clean train subset (5,000)")
    axes[1].set(ylabel="Accuracy (%)", ylim=(0, 100))
    axes[1].legend(fontsize=8)
    axes[2].plot(epochs, [row["lr_end"] for row in history], color="#52937c")
    axes[2].set(ylabel="Learning rate (last update per epoch)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    fig.suptitle(f"CIFAR-10 | {MODEL_SPECS[model_name]['label']} | From scratch | Validation selects checkpoint")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vit_tiny12", "vit_small12"], default="vit_tiny12")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (args.epochs > args.warmup_epochs >= 1 and args.batch_size > 1 and args.workers >= 0):
        parser.error("Require epochs > warmup-epochs >= 1, batch-size > 1, workers >= 0")
    if args.lr < 1e-5 or args.weight_decay < 0:
        parser.error("lr must be at least 1e-5; weight-decay must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported; use --precision fp32")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for name in ["train_vit.py", "models.py", "train.py", "infer_model.py"]:
        shutil.copy2(Path(__file__).parent / name, source / name)
    started = time.perf_counter()
    augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_data = datasets.CIFAR10(str(args.data_dir), train=True, transform=augment, download=False)
    clean_data = datasets.CIFAR10(str(args.data_dir), train=True, transform=evaluation_transform(), download=False)
    train_indices, val_indices = stratified_split(train_data.targets, args.seed)
    split = {"train": train_indices, "validation": val_indices}
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    json_write(args.output / "split_indices.json", split)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                   persistent_workers=args.workers > 0, worker_init_fn=seed_worker)
    training = DataLoader(Subset(train_data, train_indices), shuffle=True,
                         generator=torch.Generator().manual_seed(args.seed), **options)
    # 分别给评估 loader 提供 generator，防止额外评估改变后续训练的随机数流。
    def eval_loader(dataset):
        return DataLoader(dataset, shuffle=False, generator=torch.Generator().manual_seed(args.seed + 1), **options)
    validation = eval_loader(Subset(clean_data, val_indices))
    clean_training = eval_loader(Subset(clean_data, train_indices[::9]))
    model = build_model(args.model).to(device)
    # 与常见 ViT 训练方法一致：矩阵权重衰减，bias / LayerNorm / 位置与分类 token 不衰减。
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name in {"position", "class_token"} else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    mixer = mixing_transform()
    config = dict(model=args.model, model_spec=MODEL_SPECS[args.model],
        epochs=args.epochs, batch_size=args.batch_size, workers=args.workers, lr=args.lr,
        weight_decay=args.weight_decay, no_decay="bias, LayerNorm, position, class_token", seed=args.seed,
        optimizer="AdamW", adamw_betas=[0.9, 0.999], adamw_eps=1e-8,
        scheduler="per-update linear warmup + cosine", warmup_epochs=args.warmup_epochs,
        warmup_lr=1e-6, min_lr=1e-5, steps_per_epoch=len(training),
        loss="CrossEntropyLoss with mixed probability targets", label_smoothing=0.1,
        early_stopping=False, pretrained=False, precision=args.precision, evaluation_precision="fp32",
        mean=MEAN, std=STD, train_count=len(train_indices), validation_count=len(val_indices), test_count=10000,
        augmentation=dict(random_crop=32, padding=4, horizontal_flip_probability=0.5,
                          randaugment=dict(num_ops=2, magnitude=9, interpolation="bilinear"),
                          mixup_alpha=0.8, cutmix_alpha=1.0, mix_probability=1.0, cutmix_probability=0.5),
        split_sha256=split_hash, parameter_count=sum(p.numel() for p in model.parameters()),
        python=platform.python_version(), torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        cuda_build=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        note="CIFAR adaptation inspired by DeiT, not an exact paper reproduction; no teacher, EMA or stochastic depth.")
    json_write(args.output / "config.json", config)
    criterion = nn.CrossEntropyLoss()
    baseline_loss, baseline_accuracy = run_epoch(model, validation, criterion, device)
    print("START " + json.dumps(config), flush=True)
    history, best_accuracy, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        loss, mass, lr_start, lr_end = training_epoch(model, training, optimizer, mixer, device, epoch, args)
        val_loss, val_accuracy = run_epoch(model, validation, criterion, device)
        clean_loss, clean_accuracy = None, None
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            clean_loss, clean_accuracy = run_epoch(model, clean_training, criterion, device)
        if not np.isfinite([loss, val_loss, mass, val_accuracy]).all():
            raise RuntimeError(f"Non-finite metrics at epoch {epoch}")
        history.append(dict(epoch=epoch, train_loss=loss, train_soft_target_accuracy=mass,
            val_loss=val_loss, val_accuracy=val_accuracy, clean_train_loss=clean_loss,
            clean_train_accuracy=clean_accuracy, lr_start=lr_start, lr_end=lr_end,
            seconds=time.perf_counter() - tick))
        with (args.output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[-1]))
            writer.writeheader()
            writer.writerows(history)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save(dict(model_state=model.state_dict(), classes=CLASSES, config=config,
                            best_epoch=epoch, validation_accuracy=val_accuracy), args.output / "best.pt")
        print(f"EPOCH {epoch:03d}/{args.epochs} | val={val_accuracy:.2%} best={best_accuracy:.2%}@{best_epoch} "
              f"| train_loss={loss:.4f} val_loss={val_loss:.4f} lr={lr_end:.7f} "
              f"| {history[-1]['seconds']:.1f}s", flush=True)
        if epoch % 10 == 0:
            plot_history(args.output, history, args.model)
    torch.save(dict(model_state=model.state_dict(), classes=CLASSES, config=config,
                    epoch=args.epochs, validation_accuracy=history[-1]["val_accuracy"]), args.output / "last.pt")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    full_training = eval_loader(Subset(clean_data, train_indices))
    clean_loss, clean_accuracy = run_epoch(model, full_training, criterion, device)
    # 测试集只在训练结束、验证选定 checkpoint 后使用。
    test_data = datasets.CIFAR10(str(args.data_dir), train=False, transform=evaluation_transform(), download=False)
    testing = eval_loader(test_data)
    predicted, labels, confidences, confusion, test_loss = final_test(model, testing, device)
    summary = dict(model=args.model, parameter_count=config["parameter_count"], epochs=args.epochs, seed=args.seed,
        split_sha256=split_hash, baseline_validation_accuracy=baseline_accuracy,
        best_epoch=best_epoch, best_validation_accuracy=best_accuracy,
        clean_train_accuracy_at_best=clean_accuracy, clean_train_loss_at_best=clean_loss,
        test_accuracy=float(np.trace(confusion) / confusion.sum()), test_loss=test_loss,
        test_correct=int(np.trace(confusion)), test_count=int(confusion.sum()),
        training_epoch_seconds=sum(row["seconds"] for row in history),
        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        per_class_accuracy={label: float(confusion[i, i] / confusion[i].sum()) for i, label in enumerate(CLASSES)},
        note="One seed; multiple changes vs old baseline; checkpoint selected on validation; test evaluated once.")
    # 复用测试样例与混淆矩阵绘制；曲线单独绘制，明确区分软标签和普通准确率。
    plot_rows = [dict(row, train_accuracy=row["train_soft_target_accuracy"]) for row in history]
    save_figures(args.output, plot_rows, test_data, predicted, labels, confidences, confusion,
                 MODEL_SPECS[args.model]["label"])
    plot_history(args.output, history, args.model)
    np.savetxt(args.output / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    summary["total_elapsed_seconds"] = time.perf_counter() - started
    json_write(args.output / "summary.json", summary)
    del model, optimizer, checkpoint, training, validation, clean_training, full_training, testing
    gc.collect()
    torch.cuda.empty_cache()
    with (args.output / "inference_check.json").open("w") as handle:
        subprocess.run([sys.executable, str(Path(__file__).parent / "infer_model.py"),
            "--checkpoint", str(args.output / "best.pt"), "--image", str(args.output / "sample_test_image.png"),
            "--features-output", str(args.output / "sample_features.npy")], stdout=handle, check=True)
    check = json.loads((args.output / "inference_check.json").read_text())
    assert check["features_finite"] and check["feature_shape"] == [1, MODEL_SPECS[args.model]["feature_dim"]]
    assert check["top3"][0]["class"] == CLASSES[predicted[0]]
    print("FINAL " + json.dumps(summary), flush=True)
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
