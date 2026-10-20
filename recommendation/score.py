"""Load the learned embeddings and shallow network to rank specified movies."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import ShallowRecommender


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--item-ids", nargs="+", type=int, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    users = np.load(args.run_dir / "user_embeddings.npz")
    items = np.load(args.run_dir / "item_embeddings.npz")
    user_map = {int(value): index for index, value in enumerate(users["user_ids"])}
    item_map = {int(value): index for index, value in enumerate(items["item_ids"])}
    if args.user_id not in user_map:
        parser.error(f"User {args.user_id} did not appear in training; no learned embedding is available")
    missing = [value for value in args.item_ids if value not in item_map]
    if missing:
        parser.error(f"Movies did not appear in training: {missing}")
    model = ShallowRecommender(config["user_count"], config["item_count"],
                              config["embedding_dim"], config["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    user_index = user_map[args.user_id]
    item_indices = [item_map[value] for value in args.item_ids]
    # 直接使用导出的向量，演示两张 embedding 表与浅层网络分开加载。
    u = torch.from_numpy(users["embeddings"][user_index]).repeat(len(item_indices), 1)
    i = torch.from_numpy(items["embeddings"][item_indices])
    with torch.inference_mode():
        scores = model.output(torch.relu(model.hidden(torch.cat([u, i], dim=-1)))).sigmoid().flatten()
        from_ids = model(torch.full((len(item_indices),), user_index + 1),
                         torch.tensor(item_indices) + 1).sigmoid()
    torch.testing.assert_close(scores, from_ids, atol=1e-6, rtol=1e-5)
    titles = {}
    movie_file = Path(__file__).resolve().parent / "data/ml-1m/movies.dat"
    if movie_file.exists():
        for line in movie_file.read_text(encoding="latin-1").splitlines():
            item_id, title, _ = line.split("::")
            titles[int(item_id)] = title
    ranked = sorted([{"item_id": item_id, "title": titles.get(item_id), "score": float(score)}
                     for item_id, score in zip(args.item_ids, scores)], key=lambda row: (-row["score"], row["item_id"]))
    print(json.dumps({"user_id": args.user_id, "label": "rating >= 4 among observed ratings", "ranking": ranked},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
