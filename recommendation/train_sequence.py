"""Paired mean/Transformer historical feature experiment, selected on validation."""
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

from auc_models import restore_auc
from sequence_data import prepare, sequence_for_users
from sequence_model import make_sequence_model, restore_sequence
from train import metrics, write_json
from train_auc import predict


def train_one(args, prepared, common, encoder, base, device):
    out = args.output / encoder
    out.mkdir()
    tick = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    config = dict(common, encoder=encoder)
    model = make_sequence_model(config, prepared["categories"], prepared["item_metadata"])
    model.warm_start(base)
    model.to(device)
    config.update(input_dim=model.input_dim, parameter_count=sum(p.numel() for p in model.parameters()))
    write_json(out / "config.json", config)
    # Both candidates start with exactly the same DeepFM scoring function.
    model.eval()
    sample = tuple(v[:128] for v in prepared["tensors"]["validation"])
    original = restore_auc(base).to(device)
    with torch.inference_mode():
        torch.testing.assert_close(model(*sample), original(*sample[:4]), atol=3e-6, rtol=1e-4)
    del original
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    labels = torch.tensor((prepared["rows"]["train"][:, 2] >= 4).astype(np.float32), device=device)
    val_labels = (prepared["rows"]["validation"][:, 2] >= 4).astype(np.int64)
    best_auc, best_ndcg, meaningful_best, stale = -1., -1., -1., 0
    auc_epoch, ndcg_epoch = 0, 0
    history, gradient_checks = [], {}
    global_step = 0
    print("START_TRIAL " + json.dumps(config), flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_tick = time.perf_counter()
        model.train()
        total = torch.zeros((), device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        for indices in torch.randperm(len(labels), generator=generator, device=device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(v[indices] for v in prepared["tensors"]["train"]))
            loss = loss_fn(logits, labels[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {encoder}")
            loss.backward()
            # New hidden columns start at zero; the encoder receives gradients from step 2.
            if global_step == 1:
                names = ["rating_embedding.weight", "time_embedding.weight", "position_embedding.weight",
                         "movie_feature_projection.weight"]
                if encoder == "transformer":
                    names += [f"encoder_layers.{i}.self_attn.in_proj_weight" for i in range(args.layers)]
                params = dict(model.named_parameters())
                gradient_checks = {name: bool(params[name].grad is not None and params[name].grad.abs().sum() > 0)
                                   for name in names}
                assert all(gradient_checks.values()), gradient_checks
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            total += loss.detach() * len(indices)
            correct += ((logits.detach() >= 0) == labels[indices]).sum()
        val = metrics(prepared["rows"]["validation"], val_labels,
                      predict(model, prepared["tensors"]["validation"], args.batch_size))
        row = dict(epoch=epoch, train_loss=total.item() / len(labels), train_accuracy=correct.item() / len(labels),
                   val_auc=val["auc"], val_accuracy=val["accuracy_at_0_5"], val_log_loss=val["log_loss"],
                   val_ndcg_at_10=val["ndcg_at_10"], seconds=time.perf_counter() - epoch_tick)
        history.append(row)
        improved_auc, improved_ndcg = val["auc"] > best_auc, val["ndcg_at_10"] > best_ndcg
        if improved_auc or improved_ndcg:
            checkpoint = dict(config=config, epoch=epoch, best_epoch=epoch, validation=val,
                              model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              user_ids=torch.from_numpy(prepared["ids"]["user_ids"]),
                              item_ids=torch.from_numpy(prepared["ids"]["item_ids"]), feature_schema=prepared["schema"],
                              history_state={k: torch.from_numpy(v) for k, v in prepared["history_state"].items()},
                              sequence_state={k: torch.from_numpy(v) for k, v in prepared["sequence_state"].items()})
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
        print(f"[{encoder}] EPOCH {epoch:02d}/{args.epochs} | val_auc={val['auc']:.5f} best={best_auc:.5f}@{auc_epoch} | val_acc={100*val['accuracy_at_0_5']:.2f}% ndcg10={val['ndcg_at_10']:.5f} | train_loss={row['train_loss']:.4f} | {row['seconds']:.1f}s", flush=True)
        if stale >= args.patience:
            print(f"[{encoder}] EARLY_STOPPING", flush=True)
            break
    shutil.copy2(out / "best_auc.pt", out / "best.pt")
    result = dict(encoder=encoder, seed=args.seed, best_auc_epoch=auc_epoch, best_validation_auc=best_auc,
                  best_ndcg_epoch=ndcg_epoch, best_validation_ndcg=best_ndcg, completed_epochs=len(history),
                  parameter_count=config["parameter_count"], input_dim=model.input_dim,
                  training_seconds=time.perf_counter() - tick, gradient_checks=gradient_checks,
                  initial_scores_match_base=True)
    write_json(out / "training_summary.json", result)
    print("TRAINING_RESULT " + json.dumps(result), flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def export_features(model, prepared, out, device, batch_size):
    users = np.arange(1, len(prepared["ids"]["user_ids"]) + 1)
    cutoff = prepared["details"]["history_cutoff_exclusive"]
    sequence = sequence_for_users(prepared["sequence_state"], users, cutoff)
    result = []
    model.eval()
    for start in range(0, len(users), batch_size):
        inputs = [torch.as_tensor(v[start:start + batch_size], device=device) for v in sequence]
        result.append(model.extract_interest(*inputs).cpu().numpy())
    features = np.concatenate(result)
    assert features.shape == (len(users), 64) and np.isfinite(features).all()
    np.savez_compressed(out / "user_interest_features.npz", user_ids=prepared["ids"]["user_ids"],
                        features=features, history_count=(sequence[0] > 0).sum(axis=1), cutoff=np.int64(cutoff))
    for name in ["user", "item"]:
        vectors = getattr(model, f"{name}_embedding").weight.detach().cpu().numpy()[1:]
        np.savez_compressed(out / f"{name}_embeddings.npz", **{f"{name}_ids": prepared["ids"][f"{name}_ids"], "embeddings": vectors})
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--baseline-run", type=Path, default=Path("runs/mlp64-20261007-085012"))
    parser.add_argument("--reference-features", type=Path, default=Path("runs/features-20261007-092932"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("runs/auc-20261007-100354/deepfm/seed42/best_auc.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.1)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.heads, args.layers, args.ff_dim) < 1:
        parser.error("Counts must be positive")
    if not 0 < args.sequence_length < 255 or 64 % args.heads or not 0 <= args.dropout < 1:
        parser.error("Invalid sequence length, attention heads, or dropout")
    args.embedding_dim, args.hidden_dim, args.history_length = 64, 64, 20
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = args.output / "source"
    source.mkdir()
    files = ["sequence_data.py", "sequence_model.py", "train_sequence.py", "score_sequence.py", "test_sequence.py", "start_sequence.sh",
             "auc_data.py", "auc_models.py", "train_auc.py", "features.py", "feature_model.py", "model.py", "train.py",
             "test_features.py", "test_auc.py", "prepare_data.py"]
    source_hashes = {}
    for name in files:
        path = Path(__file__).parent / name
        shutil.copy2(path, source / name)
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.output / "source_sha256.json", source_hashes)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = dict(encoders=["mean", "transformer"], selection="validation AUC; both trials finish before test evaluation",
                    comparison="Same 50 all-rating tokens and common pretrained DeepFM; all parameters fine-tuned",
                    baseline="Existing DeepFM retained as validation-selected fallback; no model replaces a deployed service",
                    seeds=[args.seed], primary_metric="validation_auc", source_sha256=source_hashes)
    write_json(args.output / "protocol.json", protocol)
    print("PROTOCOL " + json.dumps(protocol), flush=True)
    prepared = prepare(args, device)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    for name in ["user_ids", "item_ids"]:
        np.testing.assert_array_equal(base[name].numpy(), prepared["ids"][name])
    assert base["config"]["split_sha256"] == prepared["details"]["split_sha256"]
    baseline = restore_auc(base).to(device)
    baseline_val = metrics(prepared["rows"]["validation"], (prepared["rows"]["validation"][:, 2] >= 4).astype(int),
                           predict(baseline, prepared["tensors"]["validation"][:4], args.batch_size))
    assert abs(baseline_val["auc"] - base["validation"]["auc"]) < 1e-6
    del baseline
    common = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    common.update(prepared["details"], source_sha256=source_hashes, selection_metric="validation_auc",
                  base_sha256=hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest(),
                  device=str(device), torch=str(torch.__version__), gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), baseline_validation=baseline_val,
                  label="Observed rating >=4; not clicks or CTR", sequence_attention="bidirectional within strictly past history")
    write_json(args.output / "experiment_config.json", common)
    records = {name: train_one(args, prepared, common, name, base, device) for name in protocol["encoders"]}
    validation_scores = {name: row["best_validation_auc"] for name, row in records.items()}
    validation_scores["existing_deepfm"] = baseline_val["auc"]
    chosen = max(validation_scores, key=validation_scores.get)
    selection = dict(selected=chosen, validation_auc=validation_scores, selection_uses_test=False,
                     checkpoint=str(args.base_checkpoint) if chosen == "existing_deepfm" else f"{chosen}/best_auc.pt")
    write_json(args.output / "final_selection.json", selection)
    print("ALL_CHECKPOINTS_FIXED " + json.dumps(selection), flush=True)
    rows = prepared["rows"]["test"]
    labels = (rows[:, 2] >= 4).astype(np.int64)
    results = {}
    for encoder, record in records.items():
        out = args.output / encoder
        checkpoint = torch.load(out / "best_auc.pt", map_location="cpu", weights_only=True)
        model = restore_sequence(checkpoint).to(device)
        scores = predict(model, prepared["tensors"]["test"], args.batch_size)
        test = metrics(rows, labels, scores)
        features = export_features(model, prepared, out, device, args.batch_size)
        sample = tuple(v[:128] for v in prepared["tensors"]["test"])
        with torch.inference_mode():
            cached = torch.as_tensor(features[sample[0].cpu().numpy() - 1], device=device)
            cached_scores = model.score_with_interest(*sample[:4], cached).sigmoid().cpu().numpy()
        np.testing.assert_allclose(cached_scores, scores[:128], atol=1e-5, rtol=1e-4)
        cpu_model = restore_sequence(checkpoint)
        cpu_scores = predict(cpu_model, tuple(v.cpu() for v in sample), args.batch_size)
        np.testing.assert_allclose(cpu_scores, scores[:128], atol=1e-5, rtol=1e-4)
        np.savez_compressed(out / "test_predictions.npz", user_ids=rows[:, 0], item_ids=rows[:, 1], ratings=rows[:, 2], labels=labels, scores=scores)
        result = dict(record, validation=checkpoint["validation"], test=test,
                      exported_user_features_shape=list(features.shape),
                      cpu_gpu_max_score_error=float(np.abs(cpu_scores - scores[:128]).max()),
                      cached_interest_max_score_error=float(np.abs(cached_scores - scores[:128]).max()))
        write_json(out / "summary.json", result)
        results[encoder] = result
        print("RESULT " + json.dumps(result), flush=True)
        del model, cpu_model
    baseline = restore_auc(base).to(device)
    baseline_test = metrics(rows, labels, predict(baseline, prepared["tensors"]["test"][:4], args.batch_size))
    summary = dict(results=results, existing_deepfm=dict(validation=baseline_val, test=baseline_test), final_selection=selection,
                   transformer_minus_mean={name: results["transformer"]["test"][name] - results["mean"]["test"][name]
                                           for name in ["auc", "ndcg_at_10", "accuracy_at_0_5", "log_loss"]},
                   elapsed_seconds=time.perf_counter() - started, counts=prepared["details"]["counts"],
                   limitations="One seed, one previously inspected chronological split; exploratory. Warm IDs only; rating classification, not CTR. NDCG over rated candidates. Mean/Transformer share inputs but differ in parameter count; both warm-start an existing validation-selected DeepFM. Histories and clock frozen at train cutoff for validation/test.")
    write_json(args.output / "summary.json", summary)
    print("EXPERIMENT_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
