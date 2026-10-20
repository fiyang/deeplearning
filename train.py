"""CIFAR-10 第一课：读取数据 -> 训练 -> 验证选模型 -> 一次最终测试。"""
import argparse
import csv
import json
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class SmallCNN(nn.Module):
    """卷积提取特征，池化缩小尺寸，全连接层输出 10 类的分数。"""
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for width in (32, 64, 128):
            layers += [
                nn.Conv2d(channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ]
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def evaluation_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def stratified_split(targets, seed):
    # 每类 500 张做验证，其余 4500 张训练。验证图片不参与梯度更新。
    generator = torch.Generator().manual_seed(seed)
    targets = torch.tensor(targets)
    train_indices, val_indices = [], []
    for label in range(10):
        indices = torch.where(targets == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)].tolist()
        val_indices.extend(indices[:500])
        train_indices.extend(indices[500:])
    assert not set(train_indices).intersection(val_indices)
    return train_indices, val_indices


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)  # 切换 BatchNorm / Dropout 的训练和评估行为。
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss = criterion(scores, labels)
            if training:
                loss.backward()  # 反向传播：计算每个参数对误差的影响。
                optimizer.step()  # 根据梯度更新参数。
            total_loss += loss.detach() * len(labels)
            correct += (scores.argmax(dim=1) == labels).sum()
            count += len(labels)
    return total_loss.item() / count, correct.item() / count


@torch.inference_mode()
def final_test(model, loader, device):
    model.eval()
    predictions, labels, confidences = [], [], []
    loss_sum = 0.0
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        scores = model(images)
        loss_sum += nn.functional.cross_entropy(scores, target, reduction="sum").item()
        confidence, predicted = scores.softmax(dim=1).max(dim=1)
        predictions.extend(predicted.cpu().tolist())
        labels.extend(target.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    return predictions, labels, confidences, confusion, loss_sum / len(labels)


def save_figures(output, history, test_data, predicted, labels, confidences, confusion, model_label="Small CNN"):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for key, title, axis in [("loss", "Cross-entropy loss", axes[0]), ("accuracy", "Accuracy (%)", axes[1])]:
        scale = 100 if key == "accuracy" else 1
        for split, color in [("train", "#2977c9"), ("val", "#e38d2c")]:
            axis.plot(epochs, [row[f"{split}_{key}"] * scale for row in history],
                      marker="o", label=split, color=color)
        axis.set(xlabel="Epoch", ylabel=title, xticks=epochs)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(f"CIFAR-10 | {model_label} | 45,000 train / 5,000 validation")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    matrix = axis.imshow(confusion, cmap="Blues")
    axis.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted class", ylabel="True class", title="Held-out test set: confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(10):
        for column in range(10):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center",
                      color="white" if confusion[row, column] > confusion.max() * 0.55 else "#222222", fontsize=9)
    fig.colorbar(matrix, ax=axis, shrink=0.8)
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)
    examples = []
    # 使用测试集固定前 16 张，不挑选表现好的样本。
    for index, axis in enumerate(axes.flat):
        axis.imshow(test_data.data[index], interpolation="nearest")
        axis.set_title(f"True: {CLASSES[labels[index]]}\nPred: {CLASSES[predicted[index]]} ({confidences[index]:.0%})",
                       color="#167344" if labels[index] == predicted[index] else "#b83638", fontsize=10)
        axis.axis("off")
        examples.append({"test_index": index, "true": CLASSES[labels[index]],
                         "predicted": CLASSES[predicted[index]], "confidence": confidences[index]})
    fig.suptitle("First 16 test images | Green: correct / Red: incorrect")
    fig.savefig(output / "predictions.png", dpi=150)
    plt.close(fig)
    from PIL import Image
    Image.fromarray(test_data.data[0]).save(output / "sample_test_image.png")
    (output / "prediction_examples.json").write_text(json.dumps(examples, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)  # 防止覆盖上次实验。
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU lesson.")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()

    augment = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    training_data = datasets.CIFAR10(str(args.data_dir), train=True, transform=augment, download=False)
    validation_data = datasets.CIFAR10(str(args.data_dir), train=True, transform=evaluation_transform(), download=False)
    train_indices, val_indices = stratified_split(training_data.targets, args.seed)
    (args.output / "split_indices.json").write_text(json.dumps({"train": train_indices, "validation": val_indices}))
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                   persistent_workers=args.workers > 0, worker_init_fn=seed_worker)
    train_loader = DataLoader(Subset(training_data, train_indices), shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), **options)
    val_loader = DataLoader(Subset(validation_data, val_indices), shuffle=False, **options)
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "python": platform.python_version(),
        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "parameter_count": sum(p.numel() for p in model.parameters()), "train_count": 45000,
        "validation_count": 5000, "test_count": 10000, "mean": MEAN, "std": STD})
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    baseline_loss, baseline_accuracy = run_epoch(model, val_loader, criterion, device)
    print(f"Before training: validation accuracy={baseline_accuracy:.2%}", flush=True)
    history, best_accuracy, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                   val_loss=val_loss, val_accuracy=val_accuracy, lr=learning_rate,
                   seconds=time.perf_counter() - epoch_started)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError("Non-finite loss detected.")
        history.append(row)
        with (args.output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch:02d}/{args.epochs} | train={train_accuracy:.2%}, loss={train_loss:.4f} | "
              f"val={val_accuracy:.2%}, loss={val_loss:.4f} | {row['seconds']:.1f}s", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save({"model_state": model.state_dict(), "classes": CLASSES, "config": config,
                        "best_epoch": epoch, "validation_accuracy": val_accuracy}, args.output / "best.pt")

    # 用验证集最优模型对官方测试集做一次最终评估；测试集不参与选模型。
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_data = datasets.CIFAR10(str(args.data_dir), train=False, transform=evaluation_transform(), download=False)
    test_loader = DataLoader(test_data, shuffle=False, **options)
    predictions, labels, confidences, confusion, test_loss = final_test(model, test_loader, device)
    summary = {"baseline_validation_accuracy": baseline_accuracy, "baseline_validation_loss": baseline_loss,
        "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
        "test_accuracy": float(np.trace(confusion) / confusion.sum()), "test_loss": test_loss,
        "test_correct": int(np.trace(confusion)), "test_count": int(confusion.sum()),
        "per_class_accuracy": {label: float(confusion[i, i] / confusion[i].sum()) for i, label in enumerate(CLASSES)},
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "elapsed_seconds_before_plots": time.perf_counter() - started,
        "note": "Single seeded run; best checkpoint selected using validation accuracy only."}
    save_figures(args.output, history, test_data, predictions, labels, confidences, confusion)
    summary["total_elapsed_seconds"] = time.perf_counter() - started
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savetxt(args.output / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    print("FINAL " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
