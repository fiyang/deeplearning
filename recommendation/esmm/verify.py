"""Rebuild raw data/features, verify stored predictions and independent ROC AUC."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from data import prepare, sha256, write_json
from metrics import funnel_metrics
from score import restore
from train import predict


def roc_area(y, p):
    # Independent of rank-sum implementation in metrics.py.
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind="stable"); y = y[order]; p = p[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p)-1]
    tp = np.r_[0, np.cumsum(y)[ends]]
    fp = np.r_[0, (ends+1) - np.cumsum(y)[ends]]
    return float(np.trapezoid(tp/tp[-1], fp/fp[-1]))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("../data/kuairand/KuaiRand-Pure/data"))
    args = p.parse_args(); torch.set_num_threads(4)
    summary = json.loads((args.run/"summary.json").read_text())
    fixed = json.loads((args.run/"checkpoints_fixed.json").read_text())
    manifest = json.loads((args.run/"data_manifest.json").read_text())
    config = json.loads((args.run/"experiment_config.json").read_text())
    for name, expected in config["source_sha256"].items():
        assert sha256(args.run/"source"/name) == expected
    for name, expected in manifest["sha256"].items():
        assert sha256(args.data_dir/name) == expected
    with tempfile.TemporaryDirectory(prefix="verify-esmm-") as temp:
        rows, feats, labels, schema, rebuilt = prepare(args.data_dir, temp)
        assert rebuilt["counts"] == manifest["counts"]
        assert schema == json.loads((args.run/"feature_schema.json").read_text())
        original_split = np.load(args.run/"split_indices.npz")
        rebuilt_split = np.load(Path(temp)/"split_indices.npz")
        for k in original_split.files:
            np.testing.assert_array_equal(original_split[k], rebuilt_split[k])
        report = dict(raw_hashes_match=True, sources_match=True, rebuilt_features_and_splits_match=True,
                      verifier_sha256=sha256(__file__),
                      train_max_time_ms=int(rows["train"][:,4].max()), validation_min_time_ms=int(rows["validation"][:,4].min()),
                      validation_max_time_ms=int(rows["validation"][:,4].max()), test_min_time_ms=int(rows["test"][:,4].min()), trials={})
        for kind in ["independent", "esmm"]:
            checkpoint = args.run/kind/"best.pt"
            assert sha256(checkpoint) == fixed[kind]
            model, builder = restore(checkpoint, args.data_dir)
            result = {}
            for split in ["test", "random"]:
                saved = np.load(args.run/kind/f"{split}_predictions.npz")
                np.testing.assert_array_equal(saved["labels"], labels[split])
                for key, col in [("user_ids", 0), ("item_ids", 1), ("time_ms", 4)]:
                    np.testing.assert_array_equal(saved[key], rows[split][:, col])
                scores = saved["scores"]
                np.testing.assert_allclose(scores[:, 2], scores[:, 0]*scores[:, 1], atol=1e-7, rtol=1e-6)
                calculated = funnel_metrics(labels[split], scores)
                assert calculated == summary["results"][kind]["evaluation"][split]
                click, like = labels[split].T
                for name, y, s in [("ctr", click, scores[:, 0]), ("cvr_like_given_click", like[click==1], scores[click==1, 1]),
                                   ("ctcvr_click_and_like", click*like, scores[:, 2])]:
                    assert abs(roc_area(y, s)-calculated[name]["auc"]) < 1e-10
                ix = np.linspace(0, len(scores)-1, min(1024, len(scores)), dtype=int)
                impression = rows[split][ix].copy(); impression[:, 5:7] = 0
                input_pair = builder.transform(impression)
                for a, b in zip(input_pair, feats[split]):
                    np.testing.assert_array_equal(a, b[ix])
                rebuilt_scores = predict(model, tuple(torch.tensor(a, dtype=torch.long) for a in input_pair))
                np.testing.assert_allclose(rebuilt_scores, scores[ix], atol=3e-6, rtol=1e-4)
                result[split] = dict(count=len(scores), independent_roc_auc_matches=True,
                                     predictions_match_raw_rows=True, inference_uses_no_outcome=True,
                                     cpu_reload_checked_rows=len(ix), cpu_max_error=float(abs(rebuilt_scores-scores[ix]).max()))
            report["trials"][kind] = result
        write_json(args.run/"verification.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
