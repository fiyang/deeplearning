"""MovieLens-1M 联合训练：ID embedding + 单隐藏层 MLP，预测高评分并排序。"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn

from model import ShallowRecommender


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def auc(labels, scores):
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    values = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    counts = np.diff(np.r_[starts, len(values)])
    ranks = np.repeat(starts + (counts + 1) / 2, counts)
    return float((ranks[labels[order] == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def metrics(rows, labels, probabilities):
    assert np.isfinite(probabilities).all()
    p = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    ndcg, group_sizes = [], []
    order = np.argsort(rows[:, 0], kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(rows[order, 0])) + 1)
    for indices in groups:
        relevant = int(labels[indices].sum())
        # 全正/全负用户无法区分排序好坏，不纳入该排名指标。
        if not 0 < relevant < len(indices):
            continue
        ranked = indices[np.lexsort((rows[indices, 1], -p[indices]))][:10]
        discount = 1 / np.log2(np.arange(2, len(ranked) + 2))
        dcg = float((labels[ranked] * discount).sum())
        ideal = float((1 / np.log2(np.arange(2, min(relevant, 10) + 2))).sum())
        ndcg.append(dcg / ideal)
        group_sizes.append(len(indices))
    return {"count": len(labels), "high_rating_fraction": float(labels.mean()),
            "auc": auc(labels, probabilities),
            "log_loss": float(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean()),
            "accuracy_at_0_5": float(((probabilities >= .5) == labels).mean()),
            "ndcg_at_10": float(np.mean(ndcg)) if ndcg else None,
            "ranking_user_count": len(ndcg),
            "median_ranking_candidates": float(np.median(group_sizes)) if group_sizes else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ml-1m/ratings.dat"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.embedding_dim, args.hidden_dim, args.batch_size) < 1:
        parser.error("Counts and dimensions must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for name in ["train.py", "model.py"]:
        shutil.copy2(Path(__file__).parent / name, source / name)
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with args.data.open() as handle:
        data = np.fromiter((int(v) for line in handle for v in line.strip().split("::")), dtype=np.int64).reshape(-1, 4)
    assert data.shape == (1000209, 4) and np.isin(data[:, 2], [1, 2, 3, 4, 5]).all()
    times = np.sort(data[:, 3])
    cut1, cut2 = int(times[int(.7 * len(data))]), int(times[int(.85 * len(data))])
    train_mask = data[:, 3] < cut1
    val_mask = (data[:, 3] >= cut1) & (data[:, 3] < cut2)
    test_mask = data[:, 3] >= cut2
    train = data[train_mask]
    user_ids, item_ids = np.unique(train[:, 0]), np.unique(train[:, 1])
    user_lookup = np.zeros(int(data[:, 0].max()) + 1, dtype=np.int64)
    item_lookup = np.zeros(int(data[:, 1].max()) + 1, dtype=np.int64)
    user_lookup[user_ids] = np.arange(1, len(user_ids) + 1)
    item_lookup[item_ids] = np.arange(1, len(item_ids) + 1)
    warm = np.isin(data[:, 0], user_ids) & np.isin(data[:, 1], item_ids)
    validation, testing = data[val_mask & warm], data[test_mask & warm]
    assert len(validation) > 0 and len(testing) > 0
    assert train[:, 3].max() < validation[:, 3].min() and validation[:, 3].max() < testing[:, 3].min()
    split = {"train": np.flatnonzero(train_mask), "validation": np.flatnonzero(val_mask & warm),
             "test": np.flatnonzero(test_mask & warm)}
    np.savez_compressed(args.output / "split_indices.npz", **split)
    np.savez(args.output / "id_maps.npz", user_ids=user_ids, item_ids=item_ids)
    def inputs(rows):
        return (torch.tensor(user_lookup[rows[:, 0]], device=device),
                torch.tensor(item_lookup[rows[:, 1]], device=device))
    training_inputs = inputs(train)
    training_labels = torch.tensor((train[:, 2] >= 4).astype(np.float32), device=device)
    validation_inputs, test_inputs = inputs(validation), inputs(testing)
    val_labels = (validation[:, 2] >= 4).astype(np.int64)
    test_labels = (testing[:, 2] >= 4).astype(np.int64)
    model = ShallowRecommender(len(user_ids), len(item_ids), args.embedding_dim, args.hidden_dim).to(device)
    initial_embeddings = {name: getattr(model, f"{name}_embedding").weight.detach().cpu().clone()[1:]
                          for name in ["user", "item"]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(dataset="MovieLens-1M", data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  label="rating >= 4 is 1; rating 1..3 is 0; only observed ratings, no sampled unobserved negatives",
                  split="global chronological 70/15/15; equal timestamps kept in the same split",
                  validation_selection="highest validation AUC; patience counts epochs without >0.0001 improvement",
                  cut_timestamps=[cut1, cut2], train_count=len(train), validation_count=len(validation),
                  test_count=len(testing), validation_before_warm_filter=int(val_mask.sum()),
                  test_before_warm_filter=int(test_mask.sum()),
                  validation_cold_excluded=int((val_mask & ~warm).sum()), test_cold_excluded=int((test_mask & ~warm).sum()),
                  user_count=len(user_ids), item_count=len(item_ids), parameter_count=sum(p.numel() for p in model.parameters()),
                  hidden_layers=1, trainable_embeddings=True, device=str(device), torch=str(torch.__version__),
                  started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(args.output / "config.json", config)
    print("START " + json.dumps(config), flush=True)
    @torch.inference_mode()
    def predict(tensors):
        model.eval()
        result = []
        for offset in range(0, len(tensors[0]), 16384):
            result.append(model(tensors[0][offset:offset + 16384], tensors[1][offset:offset + 16384]).sigmoid().cpu().numpy())
        return np.concatenate(result)
    best_auc, best_epoch, meaningful_best, stale = -1.0, 0, -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        model.train()
        order = torch.randperm(len(train), device=device)
        loss_total = torch.zeros((), device=device)
        for indices in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(training_inputs[0][indices], training_inputs[1][indices]), training_labels[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            loss_total += loss.detach() * len(indices)
        val = metrics(validation, val_labels, predict(validation_inputs))
        row = dict(epoch=epoch, train_loss=loss_total.item() / len(train), val_auc=val["auc"],
                   val_log_loss=val["log_loss"], val_ndcg_at_10=val["ndcg_at_10"], seconds=time.perf_counter() - tick)
        history.append(row)
        if val["auc"] > best_auc:
            best_auc, best_epoch = val["auc"], epoch
            torch.save({"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "config": config, "best_epoch": best_epoch, "validation": val}, args.output / "best.pt")
        if val["auc"] > meaningful_best + 1e-4:
            meaningful_best, stale = val["auc"], 0
        else:
            stale += 1
        with (args.output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader(); writer.writerows(history)
        print(f"EPOCH {epoch:02d}/{args.epochs} | val_auc={val['auc']:.5f} best={best_auc:.5f}@{best_epoch} | val_ndcg10={val['ndcg_at_10']:.5f} | train_loss={row['train_loss']:.4f} | {row['seconds']:.1f}s", flush=True)
        if stale >= args.patience:
            print("EARLY_STOPPING", flush=True)
            break
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    scores = predict(test_inputs)
    test = metrics(testing, test_labels, scores)
    # 只使用训练集统计的平滑物品高评分率，作为简单排序基线。
    item_counts = np.bincount(item_lookup[train[:, 1]], minlength=len(item_ids) + 1)
    item_positive = np.bincount(item_lookup[train[:, 1]], weights=(train[:, 2] >= 4), minlength=len(item_ids) + 1)
    item_prior = (item_positive + 10 * float(training_labels.mean().item())) / (item_counts + 10)
    baseline = metrics(testing, test_labels, item_prior[item_lookup[testing[:, 1]]])
    np.savez_compressed(args.output / "user_embeddings.npz", user_ids=user_ids,
                        embeddings=model.user_embedding.weight.detach().cpu().numpy()[1:])
    np.savez_compressed(args.output / "item_embeddings.npz", item_ids=item_ids,
                        embeddings=model.item_embedding.weight.detach().cpu().numpy()[1:])
    np.savez_compressed(args.output / "test_predictions.npz", user_ids=testing[:, 0], item_ids=testing[:, 1],
                        ratings=testing[:, 2], labels=test_labels, scores=scores)
    # 独立 CPU 模型重载，与 GPU 导出的分数核对，并检查两张向量表确实更新过。
    restored = ShallowRecommender(len(user_ids), len(item_ids), args.embedding_dim, args.hidden_dim)
    restored.load_state_dict(torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)["model_state"])
    restored.eval()
    with torch.inference_mode():
        reloaded = restored(test_inputs[0][:32].cpu(), test_inputs[1][:32].cpu()).sigmoid().numpy()
    np.testing.assert_allclose(reloaded, scores[:32], atol=1e-6, rtol=1e-4)
    embedding_checks = {}
    for name in ["user", "item"]:
        vectors = np.load(args.output / f"{name}_embeddings.npz")["embeddings"]
        assert vectors.shape[1] == args.embedding_dim and np.isfinite(vectors).all()
        delta = np.linalg.norm(vectors - initial_embeddings[name].numpy(), axis=1)
        assert np.all(delta > 0)
        embedding_checks[name] = {"shape": list(vectors.shape), "updated_rows": int((delta > 0).sum()),
                                  "mean_l2_change": float(delta.mean())}
    summary = {"best_epoch": best_epoch, "completed_epochs": len(history), "best_validation_auc": best_auc,
               "parameter_count": config["parameter_count"], "test": test, "item_popularity_baseline": baseline,
               "verification": {"cpu_checkpoint_reload_matches": True, "embeddings": embedding_checks},
               "elapsed_seconds": time.perf_counter() - started,
               "limitations": "Single seed. Warm-user/warm-item prediction of high ratings among observed ratings. NDCG@10 only ranks this split's rated candidates for users with both high and lower ratings; not full-catalog recommendation, not CTR, and no cold-start result."}
    write_json(args.output / "summary.json", summary)
    print("FINAL " + json.dumps(summary), flush=True)
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
