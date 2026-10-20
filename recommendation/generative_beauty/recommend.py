"""Load one selected checkpoint and generate actual item IDs for an observed history."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from download import sha256
from models import Tiger

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--user-index", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()
    if not 1 <= args.top_k <= 20:
        p.error("top-k must be between 1 and 20")
    info = json.loads((args.run_dir / "quantizer_info.json").read_text())
    if sha256(ROOT / "data/semantic_ids.npy") != info["semantic_ids_sha256"]:
        raise ValueError("The current semantic IDs differ from the checkpoint's frozen IDs")
    codes = np.load(ROOT / "data/semantic_ids.npy")
    sequences = json.loads((ROOT / "data/sequences.json").read_text())
    items = json.loads((ROOT / "data/items.json").read_text())
    seq = sequences[args.user_index]
    # Demonstrate on the training history, without using the user's held-out actions.
    observed = seq["items"][:-2]
    history = observed[-20:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model = Tiger(codes).to(device).eval()
    ckpt = torch.load(args.run_dir / "tiger/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    seen = torch.zeros(1, len(items), dtype=torch.bool, device=device)
    seen[0, torch.tensor(observed, device=device) - 1] = True
    predicted, scores = model.recommend(torch.tensor([history + [0] * (20 - len(history))], device=device),
                                        torch.tensor([seq["user_hash"]], device=device), seen)
    result = {"history": [{"asin": items[i - 1]["asin"], "title": items[i - 1]["title"]} for i in history],
              "recommendations": [{"asin": items[int(i)]["asin"], "title": items[int(i)]["title"],
                                   "semantic_id": codes[int(i)].tolist(), "log_probability": float(s)}
                                  for i, s in zip(predicted[0, :args.top_k], scores[0, :args.top_k]) if i >= 0],
              "note": "Log probabilities rank generated IDs; they are not calibrated CTR/CVR."}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
