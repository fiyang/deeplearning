"""顺序比较增加 ViT 宽度与增加训练预算；用验证集选模型。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parent
JOBS = [
    {"name": "width384-300epochs", "model": "vit_small12", "epochs": 300},
    {"name": "width192-600epochs", "model": "vit_tiny12", "epochs": 600},
]
FIXED = {"batch_size": 256, "workers": 4, "lr": 2.5e-4, "weight_decay": 0.05,
         "warmup_epochs": 5, "precision": "bf16", "seed": 42}
SNAPSHOT = ["models.py", "train_vit.py", "train.py", "infer_model.py", "check_models.py",
            "check_vit_training.py", "check_vit_compatibility.py", "run_vit_scaling.py"]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def check_baseline(path):
    config, summary = read_json(path / "config.json"), read_json(path / "summary.json")
    expected = {**FIXED, "model": "vit_tiny12", "epochs": 300, "pretrained": False,
                "optimizer": "AdamW", "label_smoothing": 0.1, "early_stopping": False}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Baseline {key}: expected {value!r}, found {config.get(key)!r}")
    if config["split_sha256"] != summary["split_sha256"] or summary["test_count"] != 10000:
        raise ValueError("Baseline split/test metadata inconsistent")
    if not (path / "source" / "models.py").is_file():
        raise FileNotFoundError("Missing baseline source/models.py for compatibility check")
    return config, summary


def save_results(output, rows):
    best = max(rows, key=lambda row: row["best_validation_accuracy"])
    report = {"selection_rule": "Highest validation accuracy; test is not used for model selection",
              "selected_experiment": best["experiment"], "runs": rows,
              "note": "One seed. Wider model retains head dimension 64 and MLP ratio 4. Longer training stretches cosine schedule."}
    write_json(output / "comparison.json", report)
    keys = ["experiment", "model", "epochs", "parameter_count", "best_epoch",
            "best_validation_accuracy", "test_accuracy", "training_epoch_seconds"]
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path,
                        default=PROJECT / "runs/vit-tiny12-300epochs-20260930-165232-8538")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    output = (args.output or PROJECT / f"runs/vit-scaling-{stamp}-{os.getpid()}").resolve()
    baseline = args.baseline.resolve()
    plan = {"baseline": str(baseline), "output": str(output), "jobs": JOBS,
            "fixed_hyperparameters": FIXED, "from_scratch": True, "execution": "sequential"}
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("PLAN_ONLY: no training, no output files, GPU not checked", flush=True)
        return
    baseline_config, baseline_summary = check_baseline(baseline)
    # 不占用正在执行其他任务的 GPU，也不停止现有进程。
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if active.strip():
        raise RuntimeError("GPU has an existing compute process; retry after it finishes")
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("These experiments require a CUDA GPU with BF16 support")
    os.chdir(PROJECT)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name in SNAPSHOT:
        shutil.copy2(PROJECT / name, source / name)
    plan["source_sha256"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in SNAPSHOT}
    write_json(output / "plan.json", plan)
    checks = [
        [sys.executable, str(source / "check_models.py")],
        [sys.executable, str(source / "check_vit_compatibility.py"), "--baseline-source", str(baseline / "source/models.py")],
        [sys.executable, str(source / "check_vit_training.py"), "--model", "vit_small12", "--batch-size", "256"],
        [sys.executable, str(PROJECT / "prepare_data.py")],
    ]
    for command in checks:
        subprocess.run(command, check=True)
    rows = [dict(baseline_summary, experiment="baseline-width192-300epochs", path=str(baseline))]
    save_results(output, rows)
    for job in JOBS:
        destination = output / job["name"]
        command = [sys.executable, "-u", str(source / "train_vit.py"), "--model", job["model"],
                   "--output", str(destination), "--data-dir", str(PROJECT / "data"), "--epochs", str(job["epochs"])]
        for key, value in FIXED.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        print(f"EXPERIMENT_START {job['name']} | log={destination}.log", flush=True)
        with destination.with_suffix(".log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        destination.with_suffix(".exit").write_text(f"{result.returncode}\n")
        if result.returncode:
            raise RuntimeError(f"Experiment failed: {job['name']}; see {destination}.log")
        config, summary = read_json(destination / "config.json"), read_json(destination / "summary.json")
        unchanged = list(FIXED) + ["augmentation", "mean", "std", "optimizer", "adamw_betas", "adamw_eps",
                                 "no_decay", "label_smoothing", "pretrained", "early_stopping", "split_sha256"]
        for key in unchanged:
            if config[key] != baseline_config[key]:
                raise ValueError(f"Unexpected training recipe change: {key}")
        if summary["split_sha256"] != baseline_summary["split_sha256"]:
            raise ValueError("Data split differs from baseline")
        check = read_json(destination / "inference_check.json")
        if not check["features_finite"] or check["feature_shape"] != [1, config["model_spec"]["feature_dim"]]:
            raise ValueError("Reloaded features failed verification")
        rows.append(dict(summary, experiment=job["name"], path=str(destination)))
        save_results(output, rows)
        print("EXPERIMENT_COMPLETE " + json.dumps(rows[-1]), flush=True)
    print("ALL_SCALING_EXPERIMENTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
