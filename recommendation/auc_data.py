"""Reuse the established temporal split and strict-past feature policy."""
import hashlib
import json
import shutil
import time

import numpy as np
import torch

from features import metadata, build_history, pair_history
from train import write_json


def prepare(args, device):
    prior = json.loads((args.baseline_run / "config.json").read_text())
    rating_file = args.data_dir / "ratings.dat"
    data_hash = hashlib.sha256(rating_file.read_bytes()).hexdigest()
    assert data_hash == prior["data_sha256"]
    with rating_file.open() as handle:
        data = np.fromiter((int(v) for line in handle for v in line.strip().split("::")), dtype=np.int64).reshape(-1, 4)
    split_file = args.baseline_run / "split_indices.npz"
    split = dict(np.load(split_file))
    ids = dict(np.load(args.baseline_run / "id_maps.npz"))
    users, items = ids["user_ids"], ids["item_ids"]
    user_lookup = np.zeros(int(data[:, 0].max()) + 1, dtype=np.int64)
    item_lookup = np.zeros(int(data[:, 1].max()) + 1, dtype=np.int64)
    user_lookup[users] = np.arange(1, len(users) + 1)
    item_lookup[items] = np.arange(1, len(items) + 1)
    rows = {name: data[indices] for name, indices in split.items()}
    assert rows["train"][:, 3].max() < rows["validation"][:, 3].min()
    assert rows["validation"][:, 3].max() < rows["test"][:, 3].min()
    assert len(np.unique(np.concatenate(list(split.values())))) == sum(map(len, split.values()))
    mapped = {name: (user_lookup[value[:, 0]], item_lookup[value[:, 1]]) for name, value in rows.items()}
    assert all(np.all(u > 0) and np.all(i > 0) for u, i in mapped.values())
    categories, item_metadata, schema = metadata(args.data_dir, users, items)
    schema.update(history_length=args.history_length,
                  history_policy="Strictly earlier training timestamps; tied events excluded. Validation/test frozen at training cutoff, no held-out feedback.")
    print("BUILD_HISTORY: reusing strict-past training-only policy", flush=True)
    tick = time.perf_counter()
    h, numeric, state = build_history(rows["train"], *mapped["train"], item_metadata[:, :18],
                                      len(users), len(items), args.history_length)
    historical = {"train": (h, numeric)}
    for name in ["validation", "test"]:
        historical[name] = pair_history(state, *mapped[name])
    if args.reference_features is not None:
        reference = args.reference_features
        assert split_file.read_bytes() == (reference / "split_indices.npz").read_bytes()
        previous_tables = np.load(reference / "feature_tables.npz")
        np.testing.assert_array_equal(previous_tables["user_categorical"], categories)
        np.testing.assert_array_equal(previous_tables["item_metadata"], item_metadata)
        previous_history = np.load(reference / "history_state.npz")
        for name, value in state.items():
            np.testing.assert_array_equal(previous_history[name], value)
        print("PASS: metadata and frozen histories exactly match the previous feature experiment", flush=True)
    print(f"HISTORY_READY | {time.perf_counter() - tick:.1f}s", flush=True)
    shutil.copy2(split_file, args.output / "split_indices.npz")
    shutil.copy2(args.baseline_run / "id_maps.npz", args.output / "id_maps.npz")
    np.savez_compressed(args.output / "feature_tables.npz", user_categorical=categories, item_metadata=item_metadata)
    np.savez_compressed(args.output / "history_state.npz", **state)
    write_json(args.output / "feature_schema.json", schema)
    tensors = {name: (torch.as_tensor(mapped[name][0], dtype=torch.long, device=device),
                      torch.as_tensor(mapped[name][1], dtype=torch.long, device=device),
                      torch.as_tensor(historical[name][0], dtype=torch.long, device=device),
                      torch.as_tensor(historical[name][1], dtype=torch.float32, device=device)) for name in rows}
    details = dict(user_count=len(users), item_count=len(items), data_sha256=data_hash,
                   split_sha256=hashlib.sha256(split_file.read_bytes()).hexdigest(),
                   metadata_sha256={name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                                    for name in ["users.dat", "movies.dat"]},
                   counts={name: len(value) for name, value in rows.items()},
                   history_cutoff_exclusive=prior["cut_timestamps"][0],
                   validation_cold_excluded=prior["validation_cold_excluded"], test_cold_excluded=prior["test_cold_excluded"])
    return dict(rows=rows, ids=ids, categories=categories, item_metadata=item_metadata, schema=schema,
                history_state=state, tensors=tensors, details=details)
