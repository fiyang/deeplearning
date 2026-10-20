"""Strict-past all-rating sequences, with tied timestamps sharing position codes."""
import numpy as np
import torch

from auc_data import prepare as prepare_base
from train import write_json

TIME_BUCKETS = 32


def encode_sequence(items, ratings, times, query_times):
    valid = items > 0
    if np.ndim(query_times) == 0:
        query_times = np.full(len(items), query_times, dtype=np.int64)
    delta = query_times[:, None] - times
    assert np.all(delta[valid] > 0), "Current/tied/future events cannot enter the history"
    buckets = np.zeros(items.shape, dtype=np.uint8)
    buckets[valid] = np.minimum(np.floor(np.log2(1 + delta[valid] / 3600)).astype(np.int64) + 1,
                                TIME_BUCKETS - 1)
    # Newest distinct timestamp is 1; all events at that timestamp share 1.
    positions = np.ones(items.shape, dtype=np.uint8)
    positions[:, 1:] += np.cumsum(times[:, 1:] != times[:, :-1], axis=1).astype(np.uint8)
    positions[~valid] = 0
    return items.copy(), ratings.copy(), buckets, positions


def build_sequences(rows, users, items, user_count, length=50):
    assert 0 < length < 255
    recent_items = np.zeros((user_count + 1, length), dtype=np.int32)
    recent_ratings = np.zeros((user_count + 1, length), dtype=np.uint8)
    recent_times = np.zeros((user_count + 1, length), dtype=np.int64)
    outputs = (np.zeros((len(rows), length), dtype=np.int32),
               np.zeros((len(rows), length), dtype=np.uint8),
               np.zeros((len(rows), length), dtype=np.uint8),
               np.zeros((len(rows), length), dtype=np.uint8))
    order = np.lexsort((rows[:, 1], rows[:, 0], rows[:, 3]))
    bounds = np.r_[0, np.flatnonzero(np.diff(rows[order, 3])) + 1, len(rows)]
    for start, end in zip(bounds[:-1], bounds[1:]):
        indices = order[start:end]
        u = users[indices]
        encoded = encode_sequence(recent_items[u], recent_ratings[u], recent_times[u], rows[indices, 3])
        for output, value in zip(outputs, encoded):
            output[indices] = value
        # Capture every tied event before any tied rating updates history.
        for row_index in indices:
            user = users[row_index]
            for array in [recent_items, recent_ratings, recent_times]:
                array[user, 1:] = array[user, :-1].copy()
            recent_items[user, 0] = items[row_index]
            recent_ratings[user, 0] = rows[row_index, 2]
            recent_times[user, 0] = rows[row_index, 3]
    state = dict(items=recent_items, ratings=recent_ratings, times=recent_times)
    return outputs, state


def sequence_for_users(state, users, cutoff):
    return encode_sequence(state["items"][users], state["ratings"][users], state["times"][users], cutoff)


def prepare(args, device):
    data = prepare_base(args, device)
    users = data["ids"]["user_ids"]
    items = data["ids"]["item_ids"]
    train = data["rows"]["train"]
    train_users = np.searchsorted(users, train[:, 0]) + 1
    train_items = np.searchsorted(items, train[:, 1]) + 1
    print("BUILD_SEQUENCES: all past ratings; equal-time events share position; no held-out feedback", flush=True)
    sequences, state = build_sequences(train, train_users, train_items, len(users), args.sequence_length)
    by_split = {"train": sequences}
    cutoff = data["details"]["history_cutoff_exclusive"]
    for name in ["validation", "test"]:
        mapped_users = np.searchsorted(users, data["rows"][name][:, 0]) + 1
        by_split[name] = sequence_for_users(state, mapped_users, cutoff)
    for name, values in by_split.items():
        data["tensors"][name] += tuple(torch.as_tensor(value, device=device) for value in values)
    np.savez_compressed(args.output / "sequence_state.npz", **state)
    data["sequence_state"] = state
    data["schema"].update(sequence_length=args.sequence_length, sequence_ratings="all 1..5 stars; current label excluded",
                          sequence_fields=["movie_id", "rating", "log_time_gap_bucket", "timestamp_group_recency_position"],
                          sequence_time_buckets=TIME_BUCKETS,
                          sequence_time_policy="Train gap measured at current training event; validation/test clock frozen at training cutoff",
                          sequence_ties="Same timestamp shares position; deterministic movie-ID order only for storage/truncation; not treated as true within-tie order",
                          sequence_target="One label per history window, not per-token next-item prediction")
    write_json(args.output / "feature_schema.json", data["schema"])
    counts = (sequences[0] > 0).sum(axis=1)
    data["details"].update(training_sequence_empty_count=int((counts == 0).sum()),
                           training_sequence_mean_length=float(counts.mean()),
                           training_sequence_median_length=float(np.median(counts)))
    print("SEQUENCES_READY", flush=True)
    return data
