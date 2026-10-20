"""Audit completed sequence artifacts and compare independent CLI feature extraction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from score_sequence import score_pairs
from sequence_data import sequence_for_users
from sequence_model import restore_sequence
from train import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.run
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "experiment_config.json").read_text())
    for name, digest in json.loads((root / "source_sha256.json").read_text()).items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((root / "split_indices.npz").read_bytes()).hexdigest() == config["split_sha256"]
    # Independently rebuild each final 50-event window by sorting each user's raw training events.
    data_file = Path(__file__).parent / "data/ml-1m/ratings.dat"
    assert hashlib.sha256(data_file.read_bytes()).hexdigest() == config["data_sha256"]
    with data_file.open() as handle:
        rows = np.fromiter((int(v) for line in handle for v in line.strip().split("::")), dtype=np.int64).reshape(-1, 4)
    ids = dict(np.load(root / "id_maps.npz"))
    split = dict(np.load(root / "split_indices.npz"))
    train = rows[split["train"]]
    order = np.lexsort((train[:, 1], train[:, 3], train[:, 0]))
    ordered = train[order]
    users, starts, counts = np.unique(ordered[:, 0], return_index=True, return_counts=True)
    state = dict(np.load(root / "sequence_state.npz"))
    length = config["sequence_length"]
    for raw_user, start, count in zip(users, starts, counts):
        events = ordered[max(start, start + count - length):start + count][::-1]
        user = np.searchsorted(ids["user_ids"], raw_user) + 1
        n = len(events)
        np.testing.assert_array_equal(state["items"][user, :n], np.searchsorted(ids["item_ids"], events[:, 1]) + 1)
        np.testing.assert_array_equal(state["ratings"][user, :n], events[:, 2])
        np.testing.assert_array_equal(state["times"][user, :n], events[:, 3])
        assert not state["items"][user, n:].any()
    assert state["times"].max() < config["history_cutoff_exclusive"]
    audit = dict(source_hashes_match=True, split_hash_matches=True, raw_data_hash_matches=True,
                 independently_rebuilt_sequence_users=len(users), trials={})
    for encoder in ["mean", "transformer"]:
        out = root / encoder
        checkpoint = torch.load(out / "best_auc.pt", map_location="cpu", weights_only=True)
        history = list(csv.DictReader((out / "metrics.csv").open()))
        chosen = max(history, key=lambda row: float(row["val_auc"]))
        assert int(chosen["epoch"]) == checkpoint["epoch"]
        assert abs(float(chosen["val_auc"]) - checkpoint["validation"]["auc"]) < 1e-12
        assert all(summary["results"][encoder]["gradient_checks"].values())
        assert checkpoint["model_state"]["hidden.weight"][:, -64:].abs().sum() > 0
        for name in state:
            np.testing.assert_array_equal(checkpoint["sequence_state"][name].numpy(), state[name])
        predictions = np.load(out / "test_predictions.npz")
        np.testing.assert_array_equal(predictions["user_ids"], rows[split["test"], 0])
        np.testing.assert_array_equal(predictions["item_ids"], rows[split["test"], 1])
        np.testing.assert_array_equal(predictions["labels"], rows[split["test"], 2] >= 4)
        vectors = np.load(out / "user_interest_features.npz")
        np.testing.assert_array_equal(vectors["user_ids"], ids["user_ids"])
        assert vectors["features"].shape == (len(users), 64)
        assert np.isfinite(vectors["features"]).all()
        assert np.std(vectors["features"], axis=0).mean() > 1e-3
        example_users = np.unique(predictions["user_ids"])
        example_users = example_users[[0, len(example_users) // 2, -1]]
        checks = []
        for raw_user in example_users:
            indices = np.flatnonzero(predictions["user_ids"] == raw_user)[:8]
            candidates = predictions["item_ids"][indices].tolist()
            scores, feature, count = score_pairs(checkpoint, int(raw_user), candidates)
            expected = predictions["scores"][indices]
            np.testing.assert_allclose(scores, expected, atol=1e-5, rtol=1e-4)
            index = np.searchsorted(vectors["user_ids"], raw_user)
            # CUDA's fused Transformer inference kernel differs slightly from the
            # CPU/unfused FP32 path. Bound vector error separately from score error.
            expected_feature = vectors["features"][index]
            relative_l2 = float(np.linalg.norm(feature - expected_feature) / max(np.linalg.norm(expected_feature), 1e-12))
            assert np.max(np.abs(feature - expected_feature)) < 1e-3
            assert relative_l2 < 1e-3
            assert count == vectors["history_count"][index]
            checks.append(dict(user_id=int(raw_user), candidates=candidates, max_score_error=float(np.abs(scores-expected).max()),
                               max_feature_error=float(np.abs(feature-expected_feature).max()), relative_feature_l2_error=relative_l2))
        audit["trials"][encoder] = dict(best_epoch=checkpoint["epoch"], selected_by=checkpoint["selected_by"],
                                       checkpoint_sha256=hashlib.sha256((out / "best_auc.pt").read_bytes()).hexdigest(),
                                       feature_sha256=hashlib.sha256((out / "user_interest_features.npz").read_bytes()).hexdigest(),
                                       mean_feature_std=float(np.std(vectors["features"], axis=0).mean()),
                                       inference_checks=checks)
        if encoder == "transformer" and torch.cuda.is_available():
            user = np.searchsorted(ids["user_ids"], example_users[0]) + 1
            sequence = sequence_for_users(state, np.array([user]), config["history_cutoff_exclusive"])
            old_fastpath = torch.backends.mha.get_fastpath_enabled()
            outputs = {}
            try:
                with torch.inference_mode():
                    for fastpath, device in [(False, "cpu"), (False, "cuda"), (True, "cuda")]:
                        torch.backends.mha.set_fastpath_enabled(fastpath)
                        model = restore_sequence(checkpoint).to(device)
                        outputs[f"{device}_fast{fastpath}"] = model.extract_interest(
                            *(torch.as_tensor(v, device=device) for v in sequence)).cpu().numpy()[0]
            finally:
                torch.backends.mha.set_fastpath_enabled(old_fastpath)
            np.testing.assert_allclose(outputs["cpu_fastFalse"], outputs["cuda_fastFalse"], atol=2e-5, rtol=1e-4)
            np.testing.assert_allclose(outputs["cuda_fastTrue"], vectors["features"][user-1], atol=2e-5, rtol=1e-4)
            audit["cuda_kernel_check"] = dict(user_id=int(example_users[0]),
                unfused_cpu_gpu_max_error=float(np.max(np.abs(outputs["cpu_fastFalse"]-outputs["cuda_fastFalse"]))),
                fused_vs_unfused_max_error=float(np.max(np.abs(outputs["cuda_fastTrue"]-outputs["cuda_fastFalse"]))),
                fused_matches_export=True)
    write_json(root / "verification.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
