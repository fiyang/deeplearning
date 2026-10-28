"""Continue the completed 14.28M TIGER from validation-selected best weights.

Derived from tiger_growth/train_growth.py; architecture and old experiments stay unchanged.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from model_factory import create_wide_model

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def lr_factor(step, total, warmup):
    if step < warmup:
        return .1 + .9 * step / warmup
    fraction = min(1., max(0., (step - warmup) / max(1, total - warmup)))
    return .1 + .45 * (1 + math.cos(math.pi * fraction))


def clip_checked(model):
    # Reject before clipping mutates gradients and before optimizer.step().
    return torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)


def numerical_report(model):
    report = []
    for name, param in model.named_parameters():
        row = {"name": name, "parameter_finite": bool(torch.isfinite(param).all()),
               "parameter_max_abs": float(param.detach().abs().max())}
        if param.grad is not None:
            grad = param.grad.detach()
            finite = torch.isfinite(grad)
            row.update(gradient_nonfinite=int((~finite).sum()),
                       gradient_max_abs=float(grad[finite].abs().max()) if bool(finite.any()) else None,
                       gradient_norm_float64=float(grad.double().norm()) if bool(finite.all()) else None)
        report.append(row)
    return report


def check_finite_state(model, optimizer):
    tensors = list(model.parameters())
    tensors.extend(v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
    if not all(bool(torch.isfinite(v).all()) for v in tensors):
        raise RuntimeError("Nonfinite model or optimizer state")


def stop_requested(signum, frame):
    raise TimeoutError("Training interrupted; resume from the last completed epoch")


def main():
    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--additional-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-examples", type=int, default=16384)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Checkpoint boundary for interrupted-run verification")
    args = parser.parse_args()
    if min(args.additional_epochs, args.batch_size, args.eval_batch_size, args.warmup_steps, args.patience, args.smoke_train_examples) < 1 or args.lr <= 0:
        parser.error("Epochs, batch sizes, warmup, patience and learning rate must be positive")
    if args.smoke:
        args.additional_epochs = 2
    parent, run = args.parent_run.resolve(), args.run_dir.resolve()
    parent_config = json.loads((parent / "config.json").read_text())
    if json.loads((parent / "status.json").read_text())["stage"] != "complete":
        raise RuntimeError("Continuation requires a completed parent run")
    if parent_config.get("parameter_count") != 14281344:
        raise RuntimeError("Continuation requires the 14.28M parent")
    for name, expected in parent_config["source_sha256"].items():
        if digest(parent / "source" / name) != expected:
            raise RuntimeError(f"Parent training source changed: {name}")
    parent_source = parent / "inherited_source"
    inherited_hashes = parent_config["inherited_source_sha256"]
    data_dir = Path(parent_config["data_dir"])
    for name, value in parent_config["data_signature"].items():
        if digest(data_dir / name) != value:
            raise RuntimeError(f"Data signature changed: {name}")
    for name, value in inherited_hashes.items():
        if digest(parent_source / name) != value:
            raise RuntimeError(f"Parent source signature changed: {name}")
    dataset_info = json.loads((data_dir / "dataset_info.json").read_text())
    for name, value in dataset_info["data_sha256"].items():
        if digest(data_dir / (name + ".npy")) != value:
            raise RuntimeError(f"Data array signature changed: {name}")
    parent_checkpoint = parent / "best.pt"
    own_hashes = {p.name: digest(p) for p in HERE.iterdir() if p.suffix in (".py", ".sh")}
    config = {"parent_run": str(parent), "parent_checkpoint_sha256": digest(parent_checkpoint),
              "data_dir": str(data_dir), "data_signature": parent_config["data_signature"],
              "inherited_source_sha256": inherited_hashes, "source_sha256": own_hashes,
              "variant": "tiger_width256_ff2048_continuation", "parameter_count": 14281344,
              "architecture": {"d_model": 256, "d_ff": 2048, "encoder_layers": 4,
                               "decoder_layers": 4, "num_heads": 6, "d_kv": 64},
              "initialization": "strict load of all parent best model tensors; no further growth",
              "parent_training_source_sha256": parent_config["source_sha256"],
              "additional_epochs": args.additional_epochs, "lr": args.lr, "minimum_lr": args.lr * .1,
              "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
              "warmup_steps": args.warmup_steps, "patience": args.patience, "seed": args.seed,
              "smoke": args.smoke, "smoke_train_examples": args.smoke_train_examples,
              "beam_size": parent_config["beam_size"], "precision": "BF16",
              "weight_decay": .01, "clip": 1., "optimizer": "AdamW reset (parent best.pt contains model only)",
              "scheduler": "warmup from 0.1x peak to peak, then cosine to 0.1x peak",
              "validation_users_sha256": digest(parent / "validation_users.npy")}
    if args.resume:
        saved = json.loads((run / "config.json").read_text())
        if saved != config:
            raise RuntimeError("Resume configuration or source mismatch")
        if (run / "summary.json").exists():
            raise RuntimeError("Continuation is already complete")
    else:
        run.mkdir(parents=True, exist_ok=False)
        shutil.copytree(parent_source, run / "inherited_source")
        shutil.copytree(parent / "source", run / "parent_training_source")
        (run / "source").mkdir()
        for p in HERE.iterdir():
            if p.suffix in (".py", ".sh"):
                shutil.copy2(p, run / "source" / p.name)
    for name, value in config["inherited_source_sha256"].items():
        if digest(run / "inherited_source" / name) != value:
            raise RuntimeError(f"Inherited snapshot changed: {name}")
    sys.path.insert(0, str(run / "inherited_source"))
    from train import Dataset, evaluate, amp, atomic_checkpoint
    from models import create_model
    from download import save_json
    if not args.resume:
        save_json(run / "config.json", config)
    if not torch.cuda.is_available():
        raise RuntimeError("This refinement experiment requires CUDA")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = Dataset(data_dir, device)
    codes = np.load(data_dir / "semantic_ids.npy")
    val_users = np.load(parent / "validation_users.npy")
    if args.smoke:
        val_users = val_users[:64]
    np.save(run / "validation_users.npy", val_users)
    model = create_wide_model(codes, create_model).to(device)
    checkpoint = torch.load(parent_checkpoint, map_location=device, weights_only=False)
    parent_epoch = checkpoint["epoch"]
    parent_validation = checkpoint["validation"]
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    if sum(p.numel() for p in model.parameters()) != config["parameter_count"]:
        raise RuntimeError("Unexpected continuation parameter count")
    n = min(args.smoke_train_examples, len(data.train_position)) if args.smoke else len(data.train_position)
    batches = math.ceil(n / args.batch_size)
    total_steps = batches * args.additional_epochs
    warmup = min(args.warmup_steps, max(1, total_steps // 10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total_steps, warmup))
    best, best_epoch, stale, update, first, history = -1., parent_epoch, 0, 0, 1, []
    save_json(run / "status.json", {"stage": "initializing", "model": "tiger", "smoke": args.smoke})
    started = time.monotonic()
    context = {}
    try:
        if args.resume:
            state = torch.load(run / "last.pt", map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            best, best_epoch, stale = state["best"], state["best_epoch"], state["stale"]
            history, update, first = state["history"], state["update"], state["refinement_epoch"] + 1
            torch.set_rng_state(state["cpu_rng"].cpu())
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
            del state
            check_finite_state(model, optimizer)
            print("RESUMED", json.dumps({"next_refinement_epoch": first, "next_cumulative_epoch": parent_epoch + first,
                                          "update": update, "next_lr": optimizer.param_groups[0]["lr"]}), flush=True)
        else:
            check_finite_state(model, optimizer)
            baseline, _, _ = evaluate(model, data, "val", val_users, args.eval_batch_size, config["beam_size"])
            if not args.smoke:
                for key in ("recall@10", "ndcg@10", "valid_result_fraction"):
                    if abs(baseline[key] - parent_validation[key]) > 1e-8:
                        raise RuntimeError(f"Parent validation reload mismatch: {key}")
            save_json(run / "initialization_check.json", {"parent_epoch": parent_epoch,
                "parent_checkpoint_sha256": config["parent_checkpoint_sha256"],
                "baseline_validation": baseline,
                "full_validation_reload_verified": not args.smoke,
                "note": "Strict full model load, identical architecture; optimizer reset."})
            best = baseline["ndcg@10"]
            atomic_checkpoint(run / "best.pt", {"model": model.state_dict(), "epoch": parent_epoch,
                                                "refinement_epoch": 0, "validation": baseline, "config": config})
            save_json(run / "baseline_validation.json", baseline)
            save_json(run / "best_validation.json", baseline)
            print("PARENT_CHECKPOINT_RELOAD_OK", parent_epoch, json.dumps(baseline), flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("CONTINUATION_TRAINING_START", json.dumps({**config, "parent_epoch": parent_epoch,
              "maximum_cumulative_epochs": parent_epoch + args.additional_epochs,
              "train_examples_per_epoch": n, "effective_warmup_steps": warmup,
              "parameter_count": sum(p.numel() for p in model.parameters())}), flush=True)
        for refinement_epoch in range(first, args.additional_epochs + 1):
            if stale >= args.patience:
                break
            epoch = parent_epoch + refinement_epoch
            model.train()
            generator = torch.Generator(device=device).manual_seed(args.seed + epoch)
            order = torch.randperm(n, generator=generator, device=device)
            if args.smoke:
                order = torch.linspace(0, len(data.train_position) - 1, n, device=device).long()[order]
            loss_sum, norm_sum, norm_max, count, epoch_start = 0., 0., 0., 0, time.monotonic()
            for batch, ids in enumerate(order.split(args.batch_size), 1):
                context = {"refinement_epoch": refinement_epoch, "cumulative_epoch": epoch, "batch": batch,
                           "batches": batches, "update_before_batch": update, "lr_used": optimizer.param_groups[0]["lr"]}
                rng_before = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
                h, user, target = data.batch("train", ids)
                optimizer.zero_grad(set_to_none=True)
                try:
                    with amp(device):
                        loss = model(h, user, target)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite loss")
                    loss.backward()
                    norm = clip_checked(model)
                    optimizer.step()
                    if batch == 1 or batch % 200 == 0:
                        check_finite_state(model, optimizer)
                except RuntimeError as exc:
                    save_json(run / "numerical_failure.json", {**context, "error": repr(exc), "layers": numerical_report(model)})
                    atomic_checkpoint(run / "failure_snapshot.pt", {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "train_indices": ids.cpu(), "rng_before_batch": rng_before, "context": context})
                    raise
                scheduler.step()
                update += 1
                count += len(ids)
                loss_sum += float(loss.detach()) * len(ids)
                grad_norm = float(norm)
                norm_sum += grad_norm
                norm_max = max(norm_max, grad_norm)
                if batch == 1 or batch % 200 == 0:
                    progress = {**context, "stage": "training", "model": "tiger", "smoke": args.smoke,
                        "epochs": parent_epoch + args.additional_epochs, "update": update,
                        "loss": float(loss.detach()), "gradient_norm_before_clip": grad_norm,
                        "gradient_norm_max_this_epoch": norm_max, "next_lr": optimizer.param_groups[0]["lr"],
                        "examples_per_second": count / (time.monotonic() - epoch_start),
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}
                    save_json(run / "status.json", progress)
                    print("STEP", json.dumps(progress), flush=True)
            check_finite_state(model, optimizer)
            row = {"epoch": epoch, "refinement_epoch": refinement_epoch, "update": update,
                   "train_loss": loss_sum / count, "gradient_norm_mean": norm_sum / batches,
                   "gradient_norm_max": norm_max, "next_lr": optimizer.param_groups[0]["lr"],
                   "training_seconds": time.monotonic() - epoch_start,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}
            save_json(run / "status.json", {"stage": "validation", "cumulative_epoch": epoch})
            val, _, _ = evaluate(model, data, "val", val_users, args.eval_batch_size, config["beam_size"])
            row["validation"] = val
            if val["ndcg@10"] > best:
                best, best_epoch, stale = val["ndcg@10"], epoch, 0
                atomic_checkpoint(run / "best.pt", {"model": model.state_dict(), "epoch": epoch,
                    "refinement_epoch": refinement_epoch, "validation": val, "config": config})
                save_json(run / "best_validation.json", val)
            else:
                stale += 1
            history.append(row)
            save_json(run / "history.json", history)
            atomic_checkpoint(run / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "refinement_epoch": refinement_epoch, "epoch": epoch,
                "update": update, "best": best, "best_epoch": best_epoch, "stale": stale, "history": history,
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()})
            print("EPOCH", json.dumps(row), "best", best_epoch, "stale", stale, flush=True)
            if args.stop_after_epoch is not None and refinement_epoch >= args.stop_after_epoch:
                save_json(run / "status.json", {"stage": "checkpoint_boundary_stop", "cumulative_epoch": epoch})
                print("CHECKPOINT_BOUNDARY_STOP", flush=True)
                return
            if stale >= args.patience:
                print("EARLY_STOP", epoch, flush=True)
                break
        selected = torch.load(run / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(selected["model"])
        reproduced, _, _ = evaluate(model, data, "val", val_users, args.eval_batch_size, config["beam_size"])
        for key in ("recall@10", "ndcg@10", "valid_result_fraction"):
            if abs(reproduced[key] - selected["validation"][key]) > 1e-8:
                raise RuntimeError(f"Best checkpoint reload mismatch: {key}")
        print("BEST_CHECKPOINT_RELOAD_OK", selected["epoch"], flush=True)
        if args.smoke:
            save_json(run / "status.json", {"stage": "smoke_complete", "test_evaluated": False})
            print("SMOKE_COMPLETE (test untouched)", flush=True)
            return
        save_json(run / "status.json", {"stage": "final_test", "model": "tiger"})
        result_path = run / "tiger_test.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result, predicted, target = evaluate(model, data, "test", np.arange(data.users), args.eval_batch_size, config["beam_size"])
            np.savez_compressed(run / "tiger_test_predictions.npz", predicted=predicted, target=target)
            save_json(result_path, result)
        parent_test = json.loads((parent / "tiger_test.json").read_text())
        for key in ("requests", "beam_size", "ranking", "seen_filter", "users_sha256"):
            if result[key] != parent_test[key]:
                raise RuntimeError(f"Parent test protocol mismatch: {key}")
        comparison = {key: {"parent": parent_test[key], "continued": result[key],
                      "delta": result[key]-parent_test[key],
                      "relative_percent": (result[key]/parent_test[key]-1)*100}
                      for key in ("recall@10", "ndcg@10", "recall@20")}
        summary = {"model": "tiger_wide_continued", "parameter_count": config["parameter_count"],
            "results": result, "best_epoch": best_epoch, "parent_test_comparison": comparison,
            "parent_epochs": parent_epoch, "refinement_epochs_run": len(history),
            "cumulative_epochs_run": parent_epoch + len(history), "best_validation": reproduced,
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "note": "Single seed; same 14.28M architecture continued from parent validation best with reset optimizer and a fresh warmup/cosine schedule. Parent baseline participates in model selection; test is evaluated after selection."}
        save_json(run / "summary.json", summary)
        save_json(run / "status.json", {"stage": "complete"})
        print("FINAL", json.dumps(summary), flush=True)
        print("TRAINING_COMPLETE", flush=True)
    except Exception as exc:
        save_json(run / "status.json", {"stage": "failed", "error": repr(exc), **context})
        raise


if __name__ == "__main__":
    main()
