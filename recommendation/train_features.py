"""Controlled ID / metadata / metadata+history ablation on the saved temporal split."""
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

from features import metadata, build_history, pair_history
from feature_model import make_model, restore
from train import metrics, write_json


@torch.inference_mode()
def predict(model, tensors, batch_size=8192):
    model.eval()
    return np.concatenate([model(*(value[start:start + batch_size] for value in tensors)).sigmoid().cpu().numpy()
                           for start in range(0, len(tensors[0]), batch_size)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--baseline-run", type=Path, default=Path("runs/mlp64-20261007-085012"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["id", "metadata", "history"], default=["id", "metadata", "history"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.embedding_dim, args.hidden_dim, args.history_length, args.batch_size) < 1:
        parser.error("Counts and dimensions must be positive")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Modes must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for name in ["train_features.py", "features.py", "feature_model.py", "score_features.py", "train.py", "model.py", "test_features.py"]:
        shutil.copy2(Path(__file__).parent / name, source / name)
    started = time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = json.loads((args.baseline_run / "config.json").read_text())
    rating_file = args.data_dir / "ratings.dat"
    data_hash = hashlib.sha256(rating_file.read_bytes()).hexdigest()
    assert data_hash == prior["data_sha256"], "Baseline dataset changed"
    with rating_file.open() as handle:
        data = np.fromiter((int(v) for line in handle for v in line.strip().split("::")), dtype=np.int64).reshape(-1, 4)
    split_file = args.baseline_run / "split_indices.npz"
    split = dict(np.load(split_file))
    ids = dict(np.load(args.baseline_run / "id_maps.npz"))
    user_ids, item_ids = ids["user_ids"], ids["item_ids"]
    user_lookup = np.zeros(int(data[:, 0].max()) + 1, dtype=np.int64)
    item_lookup = np.zeros(int(data[:, 1].max()) + 1, dtype=np.int64)
    user_lookup[user_ids] = np.arange(1, len(user_ids) + 1)
    item_lookup[item_ids] = np.arange(1, len(item_ids) + 1)
    rows = {name: data[indices] for name, indices in split.items()}
    assert rows["train"][:, 3].max() < rows["validation"][:, 3].min()
    assert rows["validation"][:, 3].max() < rows["test"][:, 3].min()
    assert len(np.unique(np.concatenate(list(split.values())))) == sum(map(len, split.values()))
    mapped = {name: (user_lookup[value[:, 0]], item_lookup[value[:, 1]]) for name, value in rows.items()}
    assert all(np.all(u > 0) and np.all(i > 0) for u, i in mapped.values())
    user_categorical, item_metadata, schema = metadata(args.data_dir, user_ids, item_ids)
    schema.update(history_length=args.history_length,
                  history_policy="Training rows: strictly earlier training timestamps; tied events excluded. Validation and test: frozen training-only history at the training cutoff; no held-out ratings or feedback.",
                  history_vector="Mean of shared movie embeddings for at most the last 20 past training ratings >= 4; zero when empty. Actual max length is history_length.",
                  history_statistics="Past user/item counts and Beta(1,1)-smoothed high-rating rates; 18 mean genre indicators among the user's past high ratings.")
    write_json(args.output / "feature_schema.json", schema)
    shutil.copy2(split_file, args.output / "split_indices.npz")
    shutil.copy2(args.baseline_run / "id_maps.npz", args.output / "id_maps.npz")
    np.savez_compressed(args.output / "feature_tables.npz", user_categorical=user_categorical, item_metadata=item_metadata)
    common = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    common.update(user_count=len(user_ids), item_count=len(item_ids), data_sha256=data_hash,
                  metadata_sha256={name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() for name in ["users.dat", "movies.dat"]},
                  split_sha256=hashlib.sha256(split_file.read_bytes()).hexdigest(),
                  counts={name: len(value) for name, value in rows.items()},
                  selection_metric="validation NDCG@10", hidden_layers=1, device=str(device), torch=str(torch.__version__),
                  label="Observed rating >= 4; no unobserved negative sampling; not CTR",
                  history_cutoff_exclusive=prior["cut_timestamps"][0],
                  validation_cold_excluded=prior["validation_cold_excluded"], test_cold_excluded=prior["test_cold_excluded"],
                  started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(args.output / "experiment_config.json", common)
    print("EXPERIMENT " + json.dumps(common), flush=True)
    history_state = None
    history_features = {}
    if "history" in args.modes:
        tick = time.perf_counter()
        print("BUILD_HISTORY: strictly earlier training events; freeze before validation/test", flush=True)
        h, numeric, history_state = build_history(rows["train"], *mapped["train"], item_metadata[:, :18],
                                                 len(user_ids), len(item_ids), args.history_length)
        history_features["train"] = (h, numeric)
        for name in ["validation", "test"]:
            history_features[name] = pair_history(history_state, *mapped[name])
        np.savez_compressed(args.output / "history_state.npz", **history_state)
        print(f"HISTORY_READY | {time.perf_counter() - tick:.1f}s", flush=True)
    base_inputs = {name: tuple(torch.as_tensor(v, dtype=torch.long, device=device) for v in values)
                   for name, values in mapped.items()}
    historical_inputs = {name: base_inputs[name] + (torch.as_tensor(value[0], dtype=torch.long, device=device),
                                                    torch.as_tensor(value[1], dtype=torch.float32, device=device))
                         for name, value in history_features.items()}
    train_labels = torch.as_tensor((rows["train"][:, 2] >= 4).astype(np.float32), device=device)
    val_labels = (rows["validation"][:, 2] >= 4).astype(np.int64)
    training_records = {}
    for mode in args.modes:
        mode_started = time.perf_counter()
        out = args.output / mode
        out.mkdir()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Batch order is identical across modes, independent of model initialization RNG consumption.
        generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
        config = dict(common, mode=mode)
        model = make_model(config, user_categorical, item_metadata).to(device)
        config.update(input_dim=2 * args.embedding_dim if mode == "id" else model.input_dim,
                      parameter_count=sum(p.numel() for p in model.parameters()))
        write_json(out / "config.json", config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        tensors = historical_inputs if mode == "history" else base_inputs
        loss_fn = nn.BCEWithLogitsLoss()
        best, meaningful_best, best_epoch, stale = -1., -1., 0, 0
        history, gradient_checks = [], {}
        print("START_MODE " + json.dumps(config), flush=True)
        for epoch in range(1, args.epochs + 1):
            tick = time.perf_counter()
            model.train()
            total = torch.zeros((), device=device)
            order = torch.randperm(len(train_labels), generator=generator, device=device)
            for indices in order.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(*(value[indices] for value in tensors["train"])), train_labels[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                loss.backward()
                if not gradient_checks:
                    for name, module in model.named_modules():
                        if isinstance(module, nn.Embedding):
                            gradient_checks[name] = bool(module.weight.grad[1:].abs().sum() > 0)
                    assert all(gradient_checks.values())
                nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                total += loss.detach() * len(indices)
            val = metrics(rows["validation"], val_labels, predict(model, tensors["validation"]))
            score = val["ndcg_at_10"]
            row = dict(epoch=epoch, train_loss=total.item() / len(train_labels), val_auc=val["auc"],
                       val_log_loss=val["log_loss"], val_ndcg_at_10=score, seconds=time.perf_counter() - tick)
            history.append(row)
            if score > best:
                best, best_epoch = score, epoch
                checkpoint = {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              "config": config, "best_epoch": best_epoch, "validation": val,
                              "user_ids": torch.from_numpy(user_ids), "item_ids": torch.from_numpy(item_ids),
                              "feature_schema": schema}
                if mode == "history":
                    checkpoint["history_state"] = {k: torch.from_numpy(v) for k, v in history_state.items()}
                torch.save(checkpoint, out / "best.pt")
            if score > meaningful_best + args.min_delta:
                meaningful_best, stale = score, 0
            else:
                stale += 1
            with (out / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            print(f"[{mode}] EPOCH {epoch:02d}/{args.epochs} | val_ndcg10={score:.5f} best={best:.5f}@{best_epoch} | val_auc={val['auc']:.5f} | train_loss={row['train_loss']:.4f} | {row['seconds']:.1f}s", flush=True)
            if stale >= args.patience:
                print(f"[{mode}] EARLY_STOPPING", flush=True)
                break
        training_records[mode] = dict(best_epoch=best_epoch, completed_epochs=len(history), best_validation_ndcg_at_10=best,
                                      training_seconds=time.perf_counter() - mode_started, gradient_checks=gradient_checks)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # All candidate checkpoints are fixed before any new test metrics are examined.
    results = {}
    test_labels = (rows["test"][:, 2] >= 4).astype(np.int64)
    for mode in args.modes:
        out = args.output / mode
        checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        model = restore(checkpoint).to(device)
        test_inputs = historical_inputs["test"] if mode == "history" else base_inputs["test"]
        scores = predict(model, test_inputs)
        test = metrics(rows["test"], test_labels, scores)
        cpu_model = restore(checkpoint)
        cpu_scores = predict(cpu_model, tuple(v[:128].cpu() for v in test_inputs))
        np.testing.assert_allclose(cpu_scores, scores[:128], atol=2e-6, rtol=1e-4)
        for name in ["user", "item"]:
            embedding = getattr(cpu_model, f"{name}_embedding").weight.detach().numpy()[1:]
            assert embedding.shape == (len(ids[f"{name}_ids"]), args.embedding_dim) and np.isfinite(embedding).all()
            np.savez_compressed(out / f"{name}_embeddings.npz", **{f"{name}_ids": ids[f"{name}_ids"], "embeddings": embedding})
        np.savez_compressed(out / "test_predictions.npz", user_ids=rows["test"][:, 0], item_ids=rows["test"][:, 1],
                            ratings=rows["test"][:, 2], labels=test_labels, scores=scores)
        result = dict(training_records[mode], input_dim=checkpoint["config"]["input_dim"],
                      parameter_count=checkpoint["config"]["parameter_count"], validation=checkpoint["validation"],
                      test=test, cpu_checkpoint_reload_matches=True)
        results[mode] = result
        write_json(out / "summary.json", result)
        print("RESULT " + json.dumps({"mode": mode, **result}), flush=True)
        del model, cpu_model
    summary = dict(results=results, counts=common["counts"], split_sha256=common["split_sha256"],
                   elapsed_seconds=time.perf_counter() - started,
                   limitations="Single seed, warm users/items only. NDCG ranks each user's observed test ratings, not the full catalog. Held-out histories are frozen at training cutoff. Test set previously used for an earlier experiment; this is an exploratory ablation, not a new independent benchmark.")
    write_json(args.output / "summary.json", summary)
    print("EXPERIMENT_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
