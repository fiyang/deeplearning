"""Extract a user's 64-d history feature once and score known candidate movies."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sequence_data import sequence_for_users
from sequence_model import restore_sequence


@torch.inference_mode()
def score_pairs(checkpoint, user_id, item_ids):
    user_map = {int(v): i + 1 for i, v in enumerate(checkpoint["user_ids"])}
    item_map = {int(v): i + 1 for i, v in enumerate(checkpoint["item_ids"])}
    if user_id not in user_map:
        raise ValueError(f"Unknown user: {user_id}")
    missing = [v for v in item_ids if v not in item_map]
    if missing:
        raise ValueError(f"Unknown movies: {missing}")
    if not item_ids:
        raise ValueError("At least one candidate movie is required")
    model = restore_sequence(checkpoint)
    user = user_map[user_id]
    state = {k: v.numpy() for k, v in checkpoint["sequence_state"].items()}
    sequence = sequence_for_users(state, np.array([user]), checkpoint["config"]["history_cutoff_exclusive"])
    interest = model.extract_interest(*(torch.from_numpy(v) for v in sequence))
    users = torch.full((len(item_ids),), user, dtype=torch.long)
    items = torch.tensor([item_map[v] for v in item_ids], dtype=torch.long)
    history = checkpoint["history_state"]
    numeric = torch.cat([history["user_numeric"][users, :2], history["item_numeric"][items], history["user_numeric"][users, 2:]], dim=1)
    scores = model.score_with_interest(users, items, history["recent_items"][users].long(), numeric,
                                       interest.expand(len(items), -1)).sigmoid().numpy()
    return scores, interest[0].numpy(), int((sequence[0] > 0).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--item-ids", type=int, nargs="+", required=True)
    parser.add_argument("--include-interest", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        scores, interest, count = score_pairs(checkpoint, args.user_id, args.item_ids)
    except ValueError as error:
        parser.error(str(error))
    result = dict(encoder=checkpoint["config"]["encoder"], user_id=args.user_id, history_count=count,
                  feature_dim=len(interest), history_cutoff_exclusive=checkpoint["config"]["history_cutoff_exclusive"],
                  ranking=sorted([dict(item_id=v, score=float(s)) for v, s in zip(args.item_ids, scores)], key=lambda r: (-r["score"], r["item_id"])))
    if args.include_interest:
        result["interest_feature"] = interest.tolist()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
