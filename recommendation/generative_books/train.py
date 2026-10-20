"""Full-data, equal-example-budget comparison; test only after model selection."""
import argparse
from contextlib import nullcontext
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

from download import ROOT, save_json, sha256
from models import create_model

KS = (5, 10, 20)


def metrics(predicted, targets):
    match = predicted == targets[:, None]
    result = {}
    for k in KS:
        hit = match[:, :k]
        result[f"recall@{k}"] = float(hit.any(1).mean())
        result[f"ndcg@{k}"] = float((hit / np.log2(np.arange(hit.shape[1])+2)).sum(1).mean())
    result["valid_result_fraction"] = float((predicted >= 0).mean())
    result["duplicate_rows"] = int(sum(len(set(p[p >= 0])) != (p >= 0).sum() for p in predicted))
    return result


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


class Dataset:
    def __init__(self, directory, device, history=20):
        self.directory, self.device, self.history = Path(directory), device, history
        self.cpu_items = np.load(self.directory / "sequence_items.npy", mmap_mode="r")
        self.cpu_offsets = np.load(self.directory / "sequence_offsets.npy", mmap_mode="r")
        self.items = torch.tensor(self.cpu_items, dtype=torch.long, device=device)
        self.offsets = torch.tensor(self.cpu_offsets, dtype=torch.long, device=device)
        for key in ("user_hash", "train_position", "train_user", "val_position", "test_position", "popularity"):
            setattr(self, key, torch.tensor(np.load(self.directory / f"{key}.npy"), dtype=torch.long, device=device))
        self.users = len(self.user_hash)
        self.n_items = len(self.popularity)

    def batch(self, split, indices):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        owner = self.train_user[indices] if split == "train" else indices
        position = getattr(self, f"{split}_position")[indices]
        start = torch.maximum(self.offsets[owner], position-self.history)
        hpos = start[:, None] + torch.arange(self.history, device=self.device)
        history = self.items[hpos.clamp_max(len(self.items)-1)].masked_fill(hpos >= position[:, None], 0)
        return history, self.user_hash[owner], self.items[position]-1

    def seen(self, split, users):
        stop_offset = 2 if split == "val" else 1
        mask = np.zeros((len(users), self.n_items), dtype=np.bool_)
        for row, user in enumerate(users):
            begin, end = self.cpu_offsets[user:user+2]
            mask[row, self.cpu_items[begin:end-stop_offset]-1] = True
        return torch.from_numpy(mask).to(self.device)


@torch.no_grad()
def evaluate(model, data, split, user_indices, batch_size=64, beam_size=20):
    if model is not None:
        model.eval()
    predictions, targets, started = [], [], time.monotonic()
    torch.cuda.synchronize() if data.device.type == "cuda" else None
    for start in range(0, len(user_indices), batch_size):
        users = user_indices[start:start+batch_size]
        history, user, target = data.batch(split, users)
        seen = data.seen(split, users)
        if bool(seen.gather(1, target[:, None]).any()):
            raise AssertionError("Held-out target filtered as history")
        with amp(data.device):
            if model is None:
                score = data.popularity.float()[None].expand(len(users), -1).masked_fill(seen, -torch.inf)
                value, pred = score.topk(min(beam_size, data.n_items), -1)
                pred = pred.masked_fill(~torch.isfinite(value), -1)
            else:
                pred, _ = model.recommend(history, user, seen, beam_size)
        if bool((seen.gather(1, pred.clamp_min(0)) & pred.ge(0)).any()):
            raise AssertionError("Model returned a seen item")
        predictions.append(pred[:, :max(KS)].cpu().numpy())
        targets.append(target.cpu().numpy())
        if (start // batch_size + 1) % 500 == 0:
            print("EVAL_PROGRESS", split, min(start+batch_size, len(user_indices)), len(user_indices), flush=True)
    predicted, target = np.concatenate(predictions), np.concatenate(targets)
    result = metrics(predicted, target)
    if result["duplicate_rows"]:
        raise AssertionError("Duplicate recommendation within a request")
    result.update(requests=len(target), seconds=time.monotonic()-started, split=split,
                  beam_size=beam_size, ranking="full_catalog_no_sampled_negatives",
                  seen_filter="all earlier items, not only model input history",
                  users_sha256=__import__("hashlib").sha256(np.asarray(user_indices, dtype=np.int64).tobytes()).hexdigest())
    return result, predicted, target


def atomic_checkpoint(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def fit(name, codes, data, val_users, args, run):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = create_model(name, codes).to(data.device)
    out = run / name
    out.mkdir(exist_ok=True)
    n = min(args.smoke_train_examples, len(data.train_position)) if args.smoke else len(data.train_position)
    batches = math.ceil(n/args.batch_size)
    total = batches*args.epochs
    warmup = min(1000, max(1, total//10))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    def schedule(step):
        if step < warmup:
            return (step+1)/warmup
        progress = min((step-warmup)/max(1, total-warmup), 1)
        return .1 + .45*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    config = {"name": name, "parameter_count": sum(p.numel() for p in model.parameters()),
              "architecture": model.net.config.to_dict(), "seed": args.seed, "epochs": args.epochs,
              "batch_size": args.batch_size, "train_examples_per_epoch": n, "optimizer": "AdamW",
              "lr": args.lr, "weight_decay": .01, "warmup_steps": warmup,
              "scheduler": "linear warmup then cosine to 0.1x peak", "clip": 1., "precision": "BF16",
              "history_items": 20, "validation_users": len(val_users), "beam_size": args.beam_size,
              "selection": "fixed validation users ndcg@10", "tokenizer": "shared 3x512 RQ-VAE plus 1024-slot collision token",
              "token_cardinalities": list(model.sizes),
              "target_loss": "4 target tokens only; full vocabulary cross entropy; teacher forcing",
              "smoke": args.smoke}
    save_json(out / "config.json", config)
    print("MODEL_START", json.dumps(config), flush=True)
    best, best_epoch, stale, history, update, first_epoch = -1., 0, 0, [], 0, 1
    if args.resume and (out / "last.pt").exists():
        checkpoint = torch.load(out / "last.pt", map_location=data.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best, best_epoch, stale = checkpoint["best"], checkpoint["best_epoch"], checkpoint["stale"]
        history, update, first_epoch = checkpoint["history"], checkpoint["update"], checkpoint["epoch"]+1
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        if data.device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        print("RESUMED", name, first_epoch, flush=True)
    started = time.monotonic()
    for epoch in range(first_epoch, args.epochs+1):
        model.train()
        generator = torch.Generator(device=data.device).manual_seed(args.seed+epoch)
        order = torch.randperm(n, generator=generator, device=data.device)
        # Smoke uses spread-out examples, not just the first few users.
        if args.smoke:
            pool = torch.linspace(0, len(data.train_position)-1, n, device=data.device).long()
            order = pool[order]
        loss_sum, count, epoch_start = 0., 0, time.monotonic()
        for batch_no, ids in enumerate(order.split(args.batch_size), 1):
            h, u, y = data.batch("train", ids)
            with amp(data.device):
                loss = model(h, u, y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite loss in {name} epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Nonfinite gradient")
            opt.step()
            scheduler.step()
            count += len(ids)
            loss_sum += float(loss.detach())*len(ids)
            update += 1
            if batch_no == 1 or batch_no % 200 == 0:
                progress = {"stage": "training", "model": name, "epoch": epoch, "epochs": args.epochs,
                            "batch": batch_no, "batches": batches, "update": update,
                            "loss": float(loss.detach()), "lr": opt.param_groups[0]["lr"],
                            "examples_per_second": count/(time.monotonic()-epoch_start)}
                save_json(run / "status.json", progress)
                print("STEP", json.dumps(progress), flush=True)
        row = {"epoch": epoch, "update": update, "train_loss": loss_sum/count,
               "lr": opt.param_groups[0]["lr"], "training_seconds": time.monotonic()-epoch_start}
        save_json(run / "status.json", {"stage": "validation", "model": name, "epoch": epoch})
        val, _, _ = evaluate(model, data, "val", val_users, args.eval_batch_size, args.beam_size)
        row["validation"] = val
        if val["ndcg@10"] > best:
            best, best_epoch, stale = val["ndcg@10"], epoch, 0
            atomic_checkpoint(out / "best.pt", {"model": model.state_dict(), "epoch": epoch,
                                               "config": config, "validation": val})
            save_json(out / "best_validation.json", val)
        else:
            stale += 1
        history.append(row)
        save_json(out / "history.json", history)
        atomic_checkpoint(out / "last.pt", {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                            "scheduler": scheduler.state_dict(), "epoch": epoch,
                                            "best": best, "best_epoch": best_epoch, "stale": stale,
                                            "update": update, "history": history,
                                            "cpu_rng": torch.get_rng_state(),
                                            "cuda_rng": torch.cuda.get_rng_state() if data.device.type == "cuda" else None})
        print("EPOCH", name, json.dumps(row), "best", best_epoch, flush=True)
        if stale >= args.patience:
            print("EARLY_STOP", name, epoch, flush=True)
            break
    summary = {"best_epoch": best_epoch, "epochs_run": len(history),
               "parameter_count": config["parameter_count"], "this_invocation_seconds": time.monotonic()-started}
    save_json(out / "training_summary.json", summary)
    del model, opt, scheduler
    if data.device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--val-users", type=int, default=20000)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-examples", type=int, default=2048)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.beam_size < 20 or args.val_users < 1 or args.epochs < 1:
        parser.error("beam >=20, validation users >=1, epochs >=1 required")
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    if args.smoke:
        args.epochs, args.val_users = 1, 64
    run = args.run_dir or ROOT / "runs" / ("books-" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    run.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        raise RuntimeError("Full experiment requires CUDA")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = Dataset(args.data_dir, device)
    codes = np.load(args.data_dir / "semantic_ids.npy")
    rng = np.random.default_rng(args.seed)
    val_users = np.sort(rng.choice(data.users, min(args.val_users, data.users), replace=False)).astype(np.int64)
    source = run / "source"
    signature = {name: sha256(args.data_dir / name) for name in
                 ("dataset_info.json", "quantizer_info.json", "semantic_ids.npy")}
    config = {**vars(args), "run_dir": str(run), "data_dir": str(args.data_dir), "device": str(device), "data_signature": signature,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}
    if args.resume:
        old = json.loads((run / "config.json").read_text())
        for key in ("epochs", "batch_size", "eval_batch_size", "val_users", "beam_size", "lr", "patience", "seed", "smoke", "data_dir", "data_signature"):
            if old[key] != config[key]:
                raise RuntimeError(f"Resume configuration mismatch: {key}")
        current = {f.name: sha256(f) for f in ROOT.iterdir() if f.suffix in (".py", ".sh")}
        if current != old["source_sha256"]:
            raise RuntimeError("Source changed; start a new experiment instead of resuming")
        if (run / "summary.json").exists():
            raise RuntimeError("This run is already complete")
    else:
        source.mkdir()
        for f in ROOT.iterdir():
            if f.suffix in (".py", ".sh", ".md"):
                shutil.copy2(f, source / f.name)
        config["source_sha256"] = {f.name: sha256(f) for f in ROOT.iterdir() if f.suffix in (".py", ".sh")}
        save_json(run / "config.json", config)
        for name in ("dataset_info.json", "quantizer_info.json", "download_manifest.json"):
            shutil.copy2(args.data_dir / name, run / name)
        np.save(run / "validation_users.npy", val_users)
        (run / "environment.txt").write_text(subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True))
    print("RUN_DIR", run, flush=True)
    started = time.monotonic()
    try:
        pop, _, _ = evaluate(None, data, "val", val_users, args.eval_batch_size, args.beam_size)
        save_json(run / "popularity_validation.json", pop)
        print("POPULARITY_VALIDATION", json.dumps(pop), flush=True)
        training = {}
        for name in ("tiger", "decoder_only"):
            done = run / name / "training_summary.json"
            training[name] = json.loads(done.read_text()) if args.resume and done.exists() else fit(name, codes, data, val_users, args, run)
        if args.smoke:
            for name in ("tiger", "decoder_only"):
                model = create_model(name, codes).to(device)
                ckpt = torch.load(run / name / "best.pt", map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                result, _, _ = evaluate(model, data, "val", val_users, args.eval_batch_size, args.beam_size)
                for key in ("recall@10", "ndcg@10", "valid_result_fraction"):
                    assert abs(result[key]-ckpt["validation"][key]) < 1e-8
                print("CHECKPOINT_RELOAD_OK", name, flush=True)
                del model
            save_json(run / "status.json", {"stage": "smoke_complete", "test_evaluated": False})
            print("SMOKE_COMPLETE (test untouched)", flush=True)
            return
        results = {}
        for name in ("popularity", "tiger", "decoder_only"):
            save_json(run / "status.json", {"stage": "final_test", "model": name})
            result_path = run / f"{name}_test.json"
            if args.resume and result_path.exists():
                results[name] = json.loads(result_path.read_text())
                continue
            model = None
            if name != "popularity":
                model = create_model(name, codes).to(device)
                ckpt = torch.load(run / name / "best.pt", map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])
            result, predicted, target = evaluate(model, data, "test", np.arange(data.users), args.eval_batch_size, args.beam_size)
            results[name] = result
            np.savez_compressed(run / f"{name}_test_predictions.npz", predicted=predicted, target=target)
            save_json(result_path, result)
            print("TEST", name, json.dumps(result), flush=True)
            del model
            torch.cuda.empty_cache()
        summary = {"results": results, "training": training, "seed": args.seed,
                   "elapsed_seconds": time.monotonic()-started,
                   "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                   "note": "Single seed; all training examples each epoch; fixed validation user subset; all users tested; same full catalog and semantic IDs; educational architecture comparison, not official paper reproduction."}
        save_json(run / "summary.json", summary)
        save_json(run / "status.json", {"stage": "complete"})
        print("FINAL", json.dumps(summary), flush=True)
        print("TRAINING_COMPLETE", flush=True)
    except Exception as exc:
        save_json(run / "status.json", {"stage": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
