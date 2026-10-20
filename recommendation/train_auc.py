"""AUC-selected architecture experiment with validation-only seed confirmation."""
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

from auc_data import prepare
from auc_models import make_auc_model, restore_auc
from train import metrics, write_json


@torch.inference_mode()
def predict(model, tensors, batch_size=4096):
    model.eval()
    return np.concatenate([model(*(v[start:start + batch_size] for v in tensors)).sigmoid().cpu().numpy()
                           for start in range(0, len(tensors[0]), batch_size)])


def train_one(args, prepared, common, architecture, seed, device):
    key = f"{architecture}/seed{seed}"
    out = args.output / key
    out.mkdir(parents=True)
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    config = dict(common, architecture=architecture, seed=seed, mode="history")
    model = make_auc_model(config, prepared["categories"], prepared["item_metadata"]).to(device)
    config.update(input_dim=model.input_dim, parameter_count=sum(p.numel() for p in model.parameters()))
    write_json(out / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    inputs = prepared["tensors"]
    rows = prepared["rows"]
    labels = torch.as_tensor((rows["train"][:, 2] >= 4).astype(np.float32), device=device)
    val_labels = (rows["validation"][:, 2] >= 4).astype(np.int64)
    best_auc, best_ndcg, meaningful_best, stale = -1., -1., -1., 0
    auc_epoch, ndcg_epoch = 0, 0
    history, gradient_checks = [], {}
    print("START_TRIAL " + json.dumps({"trial": key, **config}), flush=True)
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        model.train()
        total = torch.zeros((), device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        for indices in torch.randperm(len(labels), generator=generator, device=device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(v[indices] for v in inputs["train"]))
            loss = loss_fn(logits, labels[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {key}")
            loss.backward()
            if not gradient_checks:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Embedding):
                        gradient_checks[name] = bool(module.weight.grad[1:].abs().sum() > 0)
                if architecture == "deepfm":
                    gradient_checks["numeric_embedding"] = bool(model.numeric_embedding.grad.abs().sum() > 0)
                assert all(gradient_checks.values()), gradient_checks
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach() * len(indices)
            correct += ((logits.detach() >= 0) == labels[indices]).sum()
        val = metrics(rows["validation"], val_labels, predict(model, inputs["validation"]))
        row = dict(epoch=epoch, train_loss=total.item() / len(labels), train_accuracy=correct.item() / len(labels),
                   val_auc=val["auc"], val_accuracy=val["accuracy_at_0_5"], val_log_loss=val["log_loss"],
                   val_ndcg_at_10=val["ndcg_at_10"], seconds=time.perf_counter() - tick)
        history.append(row)
        improved_auc, improved_ndcg = val["auc"] > best_auc, val["ndcg_at_10"] > best_ndcg
        if improved_auc or improved_ndcg:
            checkpoint = {"config": config, "epoch": epoch, "best_epoch": epoch, "validation": val,
                          "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "user_ids": torch.from_numpy(prepared["ids"]["user_ids"]),
                          "item_ids": torch.from_numpy(prepared["ids"]["item_ids"]),
                          "feature_schema": prepared["schema"],
                          "history_state": {k: torch.from_numpy(v) for k, v in prepared["history_state"].items()}}
            if improved_auc:
                best_auc, auc_epoch = val["auc"], epoch
                torch.save(dict(checkpoint, selected_by="validation_auc"), out / "best_auc.pt")
            if improved_ndcg:
                best_ndcg, ndcg_epoch = val["ndcg_at_10"], epoch
                torch.save(dict(checkpoint, selected_by="validation_ndcg_at_10"), out / "best_ndcg.pt")
        if val["auc"] > meaningful_best + args.min_delta:
            meaningful_best, stale = val["auc"], 0
        else:
            stale += 1
        with (out / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"[{key}] EPOCH {epoch:02d}/{args.epochs} | val_auc={val['auc']:.5f} best={best_auc:.5f}@{auc_epoch} | val_acc={100 * val['accuracy_at_0_5']:.2f}% ndcg10={val['ndcg_at_10']:.5f} | train_loss={row['train_loss']:.4f} train_acc={100 * row['train_accuracy']:.2f}% | {row['seconds']:.1f}s", flush=True)
        if stale >= args.patience:
            print(f"[{key}] EARLY_STOPPING", flush=True)
            break
    shutil.copy2(out / "best_auc.pt", out / "best.pt")
    record = dict(architecture=architecture, seed=seed, best_auc_epoch=auc_epoch, best_validation_auc=best_auc,
                  best_ndcg_epoch=ndcg_epoch, best_validation_ndcg=best_ndcg, completed_epochs=len(history),
                  parameter_count=config["parameter_count"], input_dim=config["input_dim"],
                  training_seconds=time.perf_counter() - started, gradient_checks=gradient_checks)
    write_json(out / "training_summary.json", record)
    print("TRAINING_RESULT " + json.dumps({"trial": key, **record}), flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return key, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--baseline-run", type=Path, default=Path("runs/mlp64-20261007-085012"))
    parser.add_argument("--reference-features", type=Path, default=Path("runs/features-20261007-092932"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.embedding_dim, args.hidden_dim, args.history_length, args.batch_size) < 1:
        parser.error("Counts and dimensions must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    files = ["auc_models.py", "auc_data.py", "train_auc.py", "score_auc.py", "test_auc.py", "start_auc.sh",
             "features.py", "feature_model.py", "model.py", "train.py", "test_features.py", "score_features.py"]
    source_hashes = {}
    for name in files:
        path = Path(__file__).parent / name
        shutil.copy2(path, source / name)
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.output / "source_sha256.json", source_hashes)
    started = time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = dict(initial_architectures=["mlp", "interaction", "deepfm"], seeds=args.seeds,
                    confirmation="MLP plus the challenger with highest first-seed validation AUC; test not inspected before confirmation",
                    final_family_selection="highest mean validation AUC among the confirmed families",
                    representative_checkpoint="first seed, fixed before evaluation; no best-test-seed selection",
                    primary_metric="validation_auc", secondary_metrics=["ndcg_at_10", "accuracy_at_0_5", "log_loss"])
    write_json(args.output / "protocol.json", protocol)
    print("PROTOCOL " + json.dumps(protocol), flush=True)
    prepared = prepare(args, device)
    common = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    common.update(prepared["details"], selection_metric="validation_auc", hidden_layers=1, dropout=0.,
                  source_sha256=source_hashes, device=str(device), torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  label="Observed rating >= 4, not CTR; threshold 0.5 for accuracy",
                  started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(args.output / "experiment_config.json", common)
    records = {}
    first_seed = args.seeds[0]
    for architecture in protocol["initial_architectures"]:
        key, record = train_one(args, prepared, common, architecture, first_seed, device)
        records[key] = record
    challenger = max(["interaction", "deepfm"], key=lambda name: records[f"{name}/seed{first_seed}"]["best_validation_auc"])
    selection = dict(challenger=challenger, chosen_using="first-seed validation AUC only",
                     validation_auc={name: records[f"{name}/seed{first_seed}"]["best_validation_auc"] for name in protocol["initial_architectures"]},
                     confirmation_families=["mlp", challenger], additional_seeds=args.seeds[1:])
    write_json(args.output / "validation_selection.json", selection)
    print("VALIDATION_SELECTION " + json.dumps(selection), flush=True)
    for seed in args.seeds[1:]:
        for architecture in selection["confirmation_families"]:
            key, record = train_one(args, prepared, common, architecture, seed, device)
            records[key] = record
    mean_validation = {name: float(np.mean([records[f"{name}/seed{s}"]["best_validation_auc"] for s in args.seeds]))
                       for name in selection["confirmation_families"]}
    selected_architecture = max(mean_validation, key=mean_validation.get)
    final_selection = dict(architecture=selected_architecture, mean_validation_auc=mean_validation,
                           representative_checkpoint=f"{selected_architecture}/seed{first_seed}/best_auc.pt",
                           selection_uses_test=False)
    write_json(args.output / "final_selection.json", final_selection)
    print("ALL_CHECKPOINTS_FIXED " + json.dumps(final_selection), flush=True)
    # First test metrics in this experiment occur only after all training and choices.
    results = {}
    rows = prepared["rows"]["test"]
    labels = (rows[:, 2] >= 4).astype(np.int64)
    for key, record in records.items():
        out = args.output / key
        checkpoint = torch.load(out / "best_auc.pt", map_location="cpu", weights_only=True)
        model = restore_auc(checkpoint).to(device)
        scores = predict(model, prepared["tensors"]["test"])
        test = metrics(rows, labels, scores)
        cpu_model = restore_auc(checkpoint)
        cpu_scores = predict(cpu_model, tuple(v[:128].cpu() for v in prepared["tensors"]["test"]))
        np.testing.assert_allclose(cpu_scores, scores[:128], atol=3e-6, rtol=1e-4)
        for name in ["user", "item"]:
            vectors = getattr(cpu_model, f"{name}_embedding").weight.detach().numpy()[1:]
            assert np.isfinite(vectors).all()
            np.savez_compressed(out / f"{name}_embeddings.npz", **{f"{name}_ids": prepared["ids"][f"{name}_ids"], "embeddings": vectors})
        np.savez_compressed(out / "test_predictions.npz", user_ids=rows[:, 0], item_ids=rows[:, 1],
                            ratings=rows[:, 2], labels=labels, scores=scores)
        result = dict(record, validation=checkpoint["validation"], test=test, cpu_checkpoint_reload_matches=True)
        write_json(out / "summary.json", result)
        results[key] = result
        print("RESULT " + json.dumps({"trial": key, **result}), flush=True)
        del model, cpu_model
    aggregates = {}
    for architecture in protocol["initial_architectures"]:
        trials = [r for r in results.values() if r["architecture"] == architecture]
        entry = dict(seed_count=len(trials), seeds=[r["seed"] for r in trials])
        for name in ["auc", "ndcg_at_10", "accuracy_at_0_5", "log_loss"]:
            values = [r["test"][name] for r in trials]
            entry[name] = dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        aggregates[architecture] = entry
    paired = {name: [results[f"{challenger}/seed{s}"]["test"][name] - results[f"mlp/seed{s}"]["test"][name]
                     for s in args.seeds] for name in ["auc", "ndcg_at_10", "accuracy_at_0_5", "log_loss"]}
    summary = dict(results=results, aggregates=aggregates, paired_challenger_minus_mlp=paired,
                   validation_selection=selection, final_selection=final_selection,
                   elapsed_seconds=time.perf_counter() - started, counts=prepared["details"]["counts"],
                   limitations="Warm IDs; observed-rating classification, not CTR. NDCG within rated candidates only. Same chronological split and previously inspected test set: exploratory results. Three seeds assess optimization variation, not independent datasets. DeepFM adapts input encoding and has more parameters; not a parameter-matched ablation or paper reproduction.")
    write_json(args.output / "summary.json", summary)
    print("EXPERIMENT_COMPLETE " + json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
