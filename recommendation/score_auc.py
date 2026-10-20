"""Run the AUC-selected models using checkpoint-contained metadata and histories."""
import argparse
import json
from pathlib import Path

import torch

from auc_models import restore_auc


def score_pairs(checkpoint, user_id, item_ids):
    user_map = {int(value): index + 1 for index, value in enumerate(checkpoint["user_ids"])}
    item_map = {int(value): index + 1 for index, value in enumerate(checkpoint["item_ids"])}
    if user_id not in user_map:
        raise ValueError(f"Unknown user: {user_id}")
    missing = [value for value in item_ids if value not in item_map]
    if missing:
        raise ValueError(f"Unknown movies: {missing}")
    users = torch.full((len(item_ids),), user_map[user_id], dtype=torch.long)
    items = torch.tensor([item_map[value] for value in item_ids], dtype=torch.long)
    state = checkpoint["history_state"]
    numeric = torch.cat([state["user_numeric"][users, :2], state["item_numeric"][items],
                         state["user_numeric"][users, 2:]], dim=1)
    tensors = (users, items, state["recent_items"][users].long(), numeric)
    with torch.inference_mode():
        return restore_auc(checkpoint)(*tensors).sigmoid().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--item-ids", nargs="+", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        scores = score_pairs(checkpoint, args.user_id, args.item_ids)
    except ValueError as error:
        parser.error(str(error))
    titles = {}
    movie_file = Path(__file__).resolve().parent / "data/ml-1m/movies.dat"
    if movie_file.exists():
        for line in movie_file.read_text(encoding="latin-1").splitlines():
            item_id, title, _ = line.split("::")
            titles[int(item_id)] = title
    ranking = sorted([dict(item_id=item, title=titles.get(item), score=float(score)) for item, score in zip(args.item_ids, scores)],
                     key=lambda row: (-row["score"], row["item_id"]))
    print(json.dumps(dict(architecture=checkpoint["config"]["architecture"], seed=checkpoint["config"]["seed"],
                          selected_by=checkpoint["selected_by"], user_id=args.user_id, ranking=ranking), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
