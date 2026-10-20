"""Verify saved trial selection, source/data identity, exports and CLI inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline-run", type=Path, default=Path("runs/mlp64-20261007-085012"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "source_sha256.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == digest
    assert (run / "split_indices.npz").read_bytes() == (args.baseline_run / "split_indices.npz").read_bytes()
    log = (run.parent / (run.name + ".log")).read_text()
    assert log.index("ALL_CHECKPOINTS_FIXED") < log.index("\nRESULT ")
    reference = np.load(args.baseline_run / "test_predictions.npz")
    verification = dict(source_snapshot_matches=True, split_matches=True,
                        all_checkpoints_fixed_before_test=True, trials={})
    for key in summary["results"]:
        out = run / key
        current = np.load(out / "test_predictions.npz")
        for name in ["user_ids", "item_ids", "ratings", "labels"]:
            np.testing.assert_array_equal(reference[name], current[name])
        checkpoint = torch.load(out / "best_auc.pt", map_location="cpu", weights_only=True)
        ndcg_checkpoint = torch.load(out / "best_ndcg.pt", map_location="cpu", weights_only=True)
        assert checkpoint["selected_by"] == "validation_auc"
        assert ndcg_checkpoint["selected_by"] == "validation_ndcg_at_10"
        with (out / "metrics.csv").open() as handle:
            history = list(csv.DictReader(handle))
        assert checkpoint["validation"]["auc"] == max(float(row["val_auc"]) for row in history)
        assert ndcg_checkpoint["validation"]["ndcg_at_10"] == max(float(row["val_ndcg_at_10"]) for row in history)
        assert (out / "best_auc.pt").read_bytes() == (out / "best.pt").read_bytes()
        for name in ["user", "item"]:
            exported = np.load(out / f"{name}_embeddings.npz")
            np.testing.assert_array_equal(exported[f"{name}_ids"], checkpoint[f"{name}_ids"].numpy())
            np.testing.assert_array_equal(exported["embeddings"], checkpoint["model_state"][f"{name}_embedding.weight"][1:].numpy())
        output = subprocess.check_output([sys.executable, str(root / "score_auc.py"), "--checkpoint", str(out / "best_auc.pt"),
                                          "--user-id", "1164", "--item-ids", "2987", "715", "1249", "2996", "728", "587", "589", "1", "6", "3006"], text=True)
        (out / "example_ranking.json").write_text(output)
        example = json.loads(output)
        mask = current["user_ids"] == 1164
        expected = {int(i): float(p) for i, p in zip(current["item_ids"][mask], current["scores"][mask])}
        difference = max(abs(expected[row["item_id"]] - row["score"]) for row in example["ranking"])
        assert difference < 3e-6, (key, difference)
        verification["trials"][key] = dict(same_test_rows=True, selected_checkpoints_match_metrics=True,
                                           exported_embeddings_match=True, cli_max_probability_difference=difference,
                                           best_auc_sha256=hashlib.sha256((out / "best_auc.pt").read_bytes()).hexdigest())
    (run / "verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
