"""Evaluate a completed model when its comparison partner failed; preserve run status."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--model", choices=("tiger", "decoder_only"), required=True)
args = parser.parse_args()
run = args.run_dir.resolve()

def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

config = json.loads((run / "config.json").read_text())
source = run / "source"
for name, expected in config["source_sha256"].items():
    assert digest(source / name) == expected, f"Saved source mismatch: {name}"
directory = Path(config["data_dir"])
for name, expected in config["data_signature"].items():
    assert digest(directory / name) == expected, f"Data signature mismatch: {name}"

lock = (run.parent / "pipeline.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
summary = json.loads((run / args.model / "training_summary.json").read_text())
result_path = run / f"{args.model}_test.json"
if result_path.exists():
    print("EXISTING_TEST", result_path.read_text(), flush=True)
    sys.exit(0)
sys.path.insert(0, str(source))
import numpy as np
import torch
from train import Dataset, evaluate
from models import create_model
from download import save_json

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda")
data = Dataset(directory, device)
model = create_model(args.model, np.load(directory / "semantic_ids.npy")).to(device)
checkpoint_path = run / args.model / "best.pt"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
assert checkpoint["epoch"] == summary["best_epoch"]
assert all(bool(torch.isfinite(t).all()) for t in checkpoint["model"].values())
model.load_state_dict(checkpoint["model"])
audit = {"model": args.model, "checkpoint": str(checkpoint_path),
         "checkpoint_sha256": digest(checkpoint_path), "best_epoch": checkpoint["epoch"],
         "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
         "note": "Completed model evaluated independently after comparison partner failed; original failed run status is preserved."}
audit_path = run / f"{args.model}_independent_evaluation.json"
save_json(audit_path, {**audit, "stage": "validation_reload_check"})
val, _, _ = evaluate(model, data, "val", np.load(run / "validation_users.npy"),
                     config["eval_batch_size"], config["beam_size"])
for key in ("recall@10", "ndcg@10", "valid_result_fraction"):
    assert abs(val[key] - checkpoint["validation"][key]) < 1e-8, f"Validation mismatch: {key}"
print("VALIDATION_RELOAD_OK", args.model, checkpoint["epoch"], flush=True)
save_json(audit_path, {**audit, "stage": "final_test", "validation_reload": val})
result, predicted, target = evaluate(model, data, "test", np.arange(data.users),
                                     config["eval_batch_size"], config["beam_size"])
np.savez_compressed(run / f"{args.model}_test_predictions.npz", predicted=predicted, target=target)
save_json(result_path, result)
save_json(audit_path, {**audit, "stage": "complete", "validation_reload": val,
                      "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
print("TEST", args.model, json.dumps(result), flush=True)
print("INDEPENDENT_EVALUATION_COMPLETE", flush=True)
