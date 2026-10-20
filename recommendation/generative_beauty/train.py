"""Compare popularity, SASRec-style full-softmax, and TIGER-style generation."""
import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.nn import functional as F

from download import sha256
from models import SASRec, Tiger
from prepare import save_json

ROOT = Path(__file__).resolve().parent
KS = (5, 10, 20)


def metrics(predicted, target):
    """One held-out target per request; invalid (-1) predictions never hit."""
    result = {}
    match = predicted == target[:, None]
    for k in KS:
        hits = match[:, :k]
        weights = 1 / np.log2(np.arange(hits.shape[1]) + 2)
        result[f"recall@{k}"] = float(hits.any(1).mean())
        result[f"ndcg@{k}"] = float((hits * weights).sum(1).mean())
    result["valid_result_fraction"] = float((predicted >= 0).mean())
    result["duplicate_rows"] = int(sum(len(set(row[row >= 0])) != int((row >= 0).sum()) for row in predicted))
    return result


def seen_mask(sequences, indices, split, n_items, device):
    mask = np.zeros((len(indices), n_items), dtype=np.bool_)
    stop = -2 if split == "val" else -1
    for row, index in enumerate(indices):
        mask[row, np.asarray(sequences[int(index)]["items"][:stop]) - 1] = True
    return torch.from_numpy(mask).to(device)


def amp(device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model, name, data, sequences, split, device, beam_size, batch_size=128, limit=None):
    n = len(data[f"{split}_target"])
    if limit is not None:
        n = min(n, limit)
    if model is not None:
        model.eval()
    n_items = len(data["popularity"])
    predictions, elapsed = [], time.monotonic()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        history = torch.from_numpy(data[f"{split}_history"][start:stop]).to(device)
        user = torch.from_numpy(data[f"{split}_user"][start:stop]).to(device)
        seen = seen_mask(sequences, data[f"{split}_sequence"][start:stop], split, n_items, device)
        target = torch.from_numpy(data[f"{split}_target"][start:stop]).to(device)
        if bool(seen.gather(1, target[:, None]).any()):
            raise ValueError("A held-out target was accidentally filtered as seen")
        with amp(device):
            if name == "tiger":
                predicted, _ = model.recommend(history, user, seen, beam_size)
            else:
                scores = (torch.from_numpy(data["popularity"].astype(np.float32)).to(device)[None]
                          .expand(len(history), -1).clone()) if name == "popularity" else model(history).float()
                scores = scores.masked_fill(seen, -torch.inf)
                values, predicted = scores.topk(min(beam_size, n_items), dim=-1)
                predicted = predicted.masked_fill(~torch.isfinite(values), -1)
        if bool((seen.gather(1, predicted.clamp_min(0)) & predicted.ge(0)).any()):
            raise AssertionError("Previously seen item returned")
        predictions.append(predicted[:, :max(KS)].cpu().numpy())
    predictions = np.concatenate(predictions)
    result = metrics(predictions, data[f"{split}_target"][:n])
    result.update({"requests": n, "seconds": time.monotonic() - elapsed, "split": split,
                   "ranking": "full_catalog_no_sampled_negatives", "beam_size": beam_size})
    return result, predictions


def create_model(name, codes):
    return Tiger(codes) if name == "tiger" else SASRec(len(codes))


def fit(name, codes, data, sequences, args, run, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = create_model(name, codes).to(device)
    n = min(1024, len(data["train_target"])) if args.smoke else len(data["train_target"])
    h, u, y = (torch.from_numpy(data[f"train_{key}"][:n]).to(device)
               for key in ("history", "user", "target"))
    lr = .001
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    steps_per_epoch = math.ceil(n / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(500, max(1, total_steps // 10))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    out = run / name
    out.mkdir()
    parameter_count = sum(p.numel() for p in model.parameters())
    cfg = {"name": name, "seed": args.seed, "parameter_count": parameter_count,
           "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "AdamW",
           "peak_lr": lr, "weight_decay": .01, "warmup_steps": warmup,
           "scheduler": "linear warmup then cosine to 10 percent peak LR", "grad_clip": 1.,
           "precision": "bf16 autocast, fp32 parameters", "width": 128,
           "encoder_layers": 4 if name == "tiger" else 2, "decoder_layers": 4 if name == "tiger" else 0,
           "history_items": 20, "dropout": .1, "eval_every": args.eval_every,
           "patience_evaluations": args.patience, "selection": "validation ndcg@10",
           "beam_size": args.beam_size, "smoke": args.smoke}
    save_json(out / "config.json", cfg)
    print("MODEL_START", json.dumps(cfg), flush=True)
    best, best_epoch, stale, history, update = -1., 0, 0, [], 0
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n, device=device)
        epoch_start, total_loss, examples = time.monotonic(), 0., 0
        for batch_no, indexes in enumerate(order.split(args.batch_size), 1):
            with amp(device):
                loss = model(h[indexes], u[indexes], y[indexes]) if name == "tiger" else F.cross_entropy(model(h[indexes]).float(), y[indexes])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss: {name} epoch={epoch} update={update}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite gradient")
            opt.step()
            scheduler.step()
            total_loss += float(loss.detach()) * len(indexes)
            examples += len(indexes)
            update += 1
            if batch_no == 1 or batch_no % 100 == 0:
                print(f"STEP {name} epoch={epoch}/{args.epochs} batch={batch_no}/{steps_per_epoch} loss={float(loss.detach()):.5f} lr={opt.param_groups[0]['lr']:.7f}", flush=True)
        row = {"epoch": epoch, "update": update, "train_loss": total_loss / examples,
               "lr": opt.param_groups[0]["lr"], "training_seconds": time.monotonic() - epoch_start}
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val, _ = evaluate(model, name, data, sequences, "val", device, args.beam_size,
                              batch_size=args.eval_batch_size, limit=64 if args.smoke else None)
            row.update({"val_" + k: v for k, v in val.items() if isinstance(v, (float, int))})
            if val["ndcg@10"] > best:
                best, best_epoch, stale = val["ndcg@10"], epoch, 0
                torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch,
                            "validation": val}, out / "best.pt")
                save_json(out / "best_validation.json", val)
            else:
                stale += 1
        history.append(row)
        fields = sorted({k for r in history for k in r})
        with open(out / "history.csv", "w") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(history)
        status = {"stage": "training", "model": name, "epoch": epoch, "best_epoch": best_epoch,
                  "best_val_ndcg@10": best, "last": row}
        save_json(run / "status.json", status)
        print("EPOCH", name, json.dumps(row), f"best={best:.6f}@{best_epoch}", flush=True)
        if stale >= args.patience:
            print("EARLY_STOP", name, epoch, flush=True)
            break
    result = {"best_epoch": best_epoch, "epochs_run": epoch,
              "seconds": time.monotonic() - start, "parameter_count": parameter_count}
    save_json(out / "training_summary.json", result)
    del model, opt, h, u, y
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--beam-size", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run-dir", type=Path)
    args = p.parse_args()
    if args.beam_size < max(KS):
        p.error("beam-size must be at least 20")
    if args.smoke:
        args.epochs = 1
    run = args.run_dir or ROOT / "runs" / ("beauty-" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    run.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        raise RuntimeError("Full run requires CUDA")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = dict(np.load(ROOT / "data/dataset.npz"))
    codes = np.load(ROOT / "data/semantic_ids.npy")
    sequences = json.loads((ROOT / "data/sequences.json").read_text())
    items = json.loads((ROOT / "data/items.json").read_text())
    source = run / "source"
    source.mkdir()
    for f in ROOT.iterdir():
        if f.suffix in (".py", ".sh", ".md"):
            shutil.copy2(f, source / f.name)
    for name in ["dataset_info.json", "quantizer_info.json", "download_manifest.json"]:
        shutil.copy2(ROOT / "data" / name, run / name)
    (run / "environment.txt").write_text(subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True))
    save_json(run / "config.json", {**vars(args), "run_dir": str(run), "device": str(device),
                                   "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                                   "source_sha256": {f.name: sha256(f) for f in source.iterdir()}})
    print("RUN_DIR", run, flush=True)
    start = time.monotonic()
    try:
        pop_val, _ = evaluate(None, "popularity", data, sequences, "val", device, args.beam_size,
                              batch_size=512, limit=64 if args.smoke else None)
        save_json(run / "popularity_validation.json", pop_val)
        print("POPULARITY_VALIDATION", json.dumps(pop_val), flush=True)
        training = {name: fit(name, codes, data, sequences, args, run, device)
                    for name in ("sasrec", "tiger")}
        if args.smoke:
            for name in ("sasrec", "tiger"):
                model = create_model(name, codes).to(device)
                checkpoint = torch.load(run / name / "best.pt", map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["state_dict"])
                restored, _ = evaluate(model, name, data, sequences, "val", device,
                                       args.beam_size, args.eval_batch_size, limit=64)
                for key in ["recall@10", "ndcg@10", "valid_result_fraction"]:
                    assert abs(restored[key] - checkpoint["validation"][key]) < 1e-8
                print("CHECKPOINT_RELOAD_OK", name, flush=True)
                del model
            save_json(run / "status.json", {"stage": "smoke_complete", "test_evaluated": False})
            print("SMOKE_COMPLETE (test untouched)", flush=True)
            return
        # Test is evaluated only after both models have been selected on validation.
        results = {}
        for name in ("popularity", "sasrec", "tiger"):
            model = None
            if name != "popularity":
                model = create_model(name, codes).to(device)
                checkpoint = torch.load(run / name / "best.pt", map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["state_dict"])
            result, pred = evaluate(model, name, data, sequences, "test", device, args.beam_size,
                                    batch_size=args.eval_batch_size if name == "tiger" else 512)
            results[name] = result
            np.savez_compressed(run / f"{name}_test_predictions.npz", predicted=pred,
                                target=data["test_target"])
            print("TEST", name, json.dumps(result), flush=True)
            if name == "tiger":
                examples = []
                for index in range(5):
                    seq = sequences[int(data["test_sequence"][index])]["items"]
                    examples.append({"history": [items[i - 1] for i in seq[:-1][-20:]],
                                     "target": items[seq[-1] - 1],
                                     "top10": [{**items[int(i)], "semantic_id": codes[int(i)].tolist()}
                                               for i in pred[index, :10] if i >= 0]})
                save_json(run / "example_recommendations.json", examples)
            del model
            torch.cuda.empty_cache()
        summary = {"results": results, "training": training, "seed": args.seed,
                   "elapsed_seconds": time.monotonic() - start,
                   "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "note": "One seed, shortened educational implementation; not an official reproduction. Full catalog; held-out test evaluated once per model."}
        save_json(run / "summary.json", summary)
        save_json(run / "status.json", {"stage": "complete"})
        print("FINAL", json.dumps(summary), flush=True)
        print("TRAINING_COMPLETE", flush=True)
    except Exception as exc:
        save_json(run / "status.json", {"stage": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
