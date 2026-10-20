"""KuaiRand ESMM experiment: lock both checkpoints before any test evaluation."""
import argparse
import copy
import csv
from datetime import datetime
from pathlib import Path
import random
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from data import prepare, sha256, write_json, FIELDS
from metrics import funnel_metrics
from model import FunnelModel


@torch.inference_mode()
def predict(model, features, batch_size=8192):
    model.eval()
    return np.concatenate([model.probabilities(*(a[i:i+batch_size] for a in features)).cpu().numpy()
                           for i in range(0, len(features[0]), batch_size)])


def train_one(args, config, initial_state, features, labels, device):
    out = args.output / config["kind"]; out.mkdir()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = FunnelModel(config).to(device)
    model.load_state_dict(initial_state, strict=(config["kind"] == "esmm"))
    if config["kind"] == "independent":
        model.cvr_encoder.load_state_dict(model.encoder.state_dict())
    config = dict(config, parameter_count=sum(p.numel() for p in model.parameters()), input_dim=model.input_dim)
    write_json(out / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                 betas=(.9, .999), eps=1e-8)
    g = torch.Generator(device=device).manual_seed(args.seed + 1000)
    y = torch.as_tensor(labels["train"], device=device)
    best, patience_best, stale, history = -1., -1., 0, []
    gradient_check = {}; started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); tick = time.perf_counter(); total = torch.zeros((), device=device)
        for ix in torch.randperm(len(y), device=device, generator=g).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(a[ix] for a in features["train"]))
            loss = model.loss(logits, y[ix])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            if not gradient_check:
                for name in ["encoder", "ctr", "cvr"] + (["cvr_encoder"] if model.kind == "independent" else []):
                    gradient_check[name] = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in getattr(model, name).parameters())
                assert all(gradient_check.values()), gradient_check
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach() * len(ix)
        scores = predict(model, features["validation"])
        val = funnel_metrics(labels["validation"], scores)
        metric = val["ctcvr_click_and_like"]["auc"]
        row = dict(epoch=epoch, train_loss=total.item()/len(y),
                   val_ctr_auc=val["ctr"]["auc"], val_cvr_auc=val["cvr_like_given_click"]["auc"],
                   val_ctcvr_auc=metric, val_ctcvr_log_loss=val["ctcvr_click_and_like"]["log_loss"],
                   val_ctcvr_ap=val["ctcvr_click_and_like"]["average_precision"], seconds=time.perf_counter()-tick)
        history.append(row)
        if metric > best:
            best = metric
            torch.save(dict(config=config, epoch=epoch, validation=val,
                            selection="validation CTCVR AUC only",
                            model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}), out / "best.pt")
        if metric > patience_best + args.min_delta:
            patience_best, stale = metric, 0
        else:
            stale += 1
        with (out / "history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(history)
        print(f"[{model.kind}] EPOCH {epoch:02d}/{args.epochs} | CTR_AUC={row['val_ctr_auc']:.5f} CVR_AUC={row['val_cvr_auc']:.5f} CTCVR_AUC={metric:.5f} best={best:.5f} | loss={row['train_loss']:.5f} | {row['seconds']:.1f}s", flush=True)
        if stale >= args.patience:
            print(f"[{model.kind}] EARLY_STOPPING patience={args.patience}", flush=True)
            break
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    result = dict(kind=model.kind, best_epoch=checkpoint["epoch"], validation=checkpoint["validation"],
                  completed_epochs=len(history), training_seconds=time.perf_counter()-started,
                  gradient_check=gradient_check, parameter_count=config["parameter_count"],
                  checkpoint_sha256=sha256(out / "best.pt"))
    write_json(out / "training_summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("../data/kuairand/KuaiRand-Pure/data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--embedding-dim", type=int, default=18)
    p.add_argument("--hidden", type=int, nargs="+", default=[360, 200, 80])
    p.add_argument("--dropout", type=float, default=0.)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.embedding_dim, *args.hidden) < 1 or args.lr <= 0 or not 0 <= args.dropout < 1:
        p.error("Invalid hyperparameters")
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"; source.mkdir()
    hashes = {}
    for file in Path(__file__).parent.iterdir():
        if file.suffix in [".py", ".sh"]:
            shutil.copy2(file, source / file.name); hashes[file.name] = sha256(file)
    write_json(args.output / "source_sha256.json", hashes)
    torch.set_num_threads(4)
    np.random.seed(args.seed); random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print("PREPARING temporal splits and train-only feature vocabulary", flush=True)
    rows, features_cpu, labels, schema, manifest = prepare(args.data_dir, args.output)
    print("DATA_COUNTS " + str(manifest["counts"]), flush=True)
    features = {k: tuple(torch.as_tensor(v, dtype=torch.long, device=device) for v in pair) for k, pair in features_cpu.items()}
    click, like = labels["train"].T
    ctr_prior = float(click.mean()); cvr_prior = float(like[click == 1].mean())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(vocabulary_size=schema["vocabulary_size"], category_count=len(FIELDS),
                  ctr_prior=ctr_prior, cvr_prior=cvr_prior, kind="esmm", source_sha256=hashes,
                  feature_schema_sha256=sha256(args.output / "feature_schema.json"), data_sha256=manifest["sha256"],
                  torch=str(torch.__version__), device=str(device), gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(args.output / "experiment_config.json", config)
    protocol = dict(primary_metric="validation CTCVR AUC", model_selection="Both checkpoints selected independently on validation; no test-driven selection or tuning.",
                    labels=manifest["labels"], architectures=["independent", "esmm"], seeds=[args.seed],
                    independent="Separate embeddings and towers; CTR BCE on every impression + CVR BCE on clicked rows only; product used for CTCVR.",
                    esmm="Shared embeddings; CTR BCE + product-probability BCE on all impressions; NO direct CVR loss.",
                    initialization="Identical initial shared embedding and tower weights; independent CVR gets a cloned embedding; same shuffle seed.",
                    stopping="Best validation CTCVR AUC checkpoint, patience on improvements > min_delta", threshold=.5,
                    caveats=["Click may mean valid play. Joint label is click AND like, not every like and not purchase conversion.",
                             "KuaiRand-Pure has incomplete histories; this experiment uses no sequence encoder.",
                             "Anonymous user metadata is a released snapshot without as-of timestamps.",
                             "Independent baseline has two embedding tables, hence more parameters.",
                             "One seed; exploratory adaptation, not numerical reproduction of the Alibaba paper."])
    write_json(args.output / "protocol.json", protocol)
    initial = copy.deepcopy(FunnelModel(config).state_dict())
    records = {}
    for kind in protocol["architectures"]:
        records[kind] = train_one(args, dict(config, kind=kind), initial, features, labels, device)
    write_json(args.output / "checkpoints_fixed.json", {k: v["checkpoint_sha256"] for k, v in records.items()})
    print("CHECKPOINTS_FIXED: starting final standard and random-policy evaluations", flush=True)
    for kind, record in records.items():
        out = args.output / kind
        ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        model = FunnelModel(ckpt["config"]).to(device); model.load_state_dict(ckpt["model_state"])
        record["evaluation"] = {}
        for split in ["test", "random"]:
            scores = predict(model, features[split]); assert (scores[:, 2] <= scores[:, 0] + 1e-7).all()
            record["evaluation"][split] = funnel_metrics(labels[split], scores)
            np.savez_compressed(out / f"{split}_predictions.npz", scores=scores, labels=labels[split].astype(np.int8),
                                user_ids=rows[split][:, 0], item_ids=rows[split][:, 1], time_ms=rows[split][:, 4])
        cpu_model = FunnelModel(ckpt["config"]); cpu_model.load_state_dict(ckpt["model_state"])
        sample = tuple(v[:256].cpu() for v in features["test"])
        cpu_scores = predict(cpu_model, sample)
        gpu_scores = predict(model, tuple(v[:256] for v in features["test"]))
        np.testing.assert_allclose(cpu_scores, gpu_scores, atol=3e-6, rtol=1e-4)
        record["cpu_reload_max_difference"] = float(np.max(np.abs(cpu_scores - gpu_scores)))
        write_json(out / "summary.json", record)
        print("RESULT " + kind + " " + str(record["evaluation"]), flush=True)
    baselines = {}
    for split in ["test", "random"]:
        probability = np.tile([ctr_prior, cvr_prior, ctr_prior*cvr_prior], (len(labels[split]), 1))
        baselines[split] = funnel_metrics(labels[split], probability)
    summary = dict(results=records, constant_train_prior_baseline=baselines, counts=manifest["counts"],
                   elapsed_seconds=time.perf_counter()-started,
                   peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type == "cuda" else None,
                   caveats=protocol["caveats"])
    write_json(args.output / "summary.json", summary)
    print(f"TRAINING_COMPLETE {args.output} elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
