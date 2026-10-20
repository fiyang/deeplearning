"""Prepare chronological examples, frozen content embeddings, and unique semantic IDs."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer

from download import sha256
from models import RQVAE

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def prepare_data(history=20):
    data = ROOT / "data"
    raw = data / "raw"
    users = defaultdict(list)
    pairs = set()
    duplicates = 0
    with gzip.open(raw / "reviews_Beauty_5.json.gz", "rt") as f:
        for line_index, line in enumerate(f):
            r = json.loads(line)
            pair = (r["reviewerID"], r["asin"])
            if pair in pairs:
                duplicates += 1
                continue
            pairs.add(pair)
            users[pair[0]].append((int(r["unixReviewTime"]), line_index, pair[1]))
    users = {u: sorted(v) for u, v in sorted(users.items()) if len(v) >= 5}
    asins = sorted({v[2] for seq in users.values() for v in seq})
    item_map = {a: i + 1 for i, a in enumerate(asins)}
    meta = {}
    with gzip.open(raw / "meta_Beauty.json.gz", "rt") as f:
        for line in f:
            r = ast.literal_eval(line)
            if r["asin"] in item_map:
                meta[r["asin"]] = r
    items = []
    for a in asins:
        r = meta.get(a, {})
        categories = sorted({str(c) for path in r.get("categories", []) for c in path})
        fields = {"asin": a, "title": str(r.get("title", "")),
                  "brand": str(r.get("brand", "")), "price": str(r.get("price", "")),
                  "categories": categories}
        fields["text"] = ". ".join([fields["title"], "Brand: " + fields["brand"],
                                       "Category: " + ", ".join(categories), "Price: " + fields["price"]])
        items.append(fields)
    sets = {s: {"history": [], "target": [], "user": [], "sequence": []}
            for s in ("train", "val", "test")}
    sequences, popularity = [], np.zeros(len(asins), dtype=np.int64)
    ties = 0
    for u, events in users.items():
        seq = [item_map[r[2]] for r in events]
        uid = int.from_bytes(hashlib.sha256(u.encode()).digest()[:8], "big") % 2000
        index = len(sequences)
        sequences.append({"user": u, "user_hash": uid, "items": seq,
                          "timestamps": [e[0] for e in events]})
        np.add.at(popularity, np.array(seq[:-2]) - 1, 1)
        ties += sum(events[i][0] == events[i - 1][0] for i in range(1, len(events)))
        for pos in range(1, len(seq)):
            split = "test" if pos == len(seq) - 1 else "val" if pos == len(seq) - 2 else "train"
            h = seq[max(0, pos - history):pos]
            sets[split]["history"].append(h + [0] * (history - len(h)))
            sets[split]["target"].append(seq[pos] - 1)
            sets[split]["user"].append(uid)
            sets[split]["sequence"].append(index)
    arrays = {f"{s}_{k}": np.asarray(v, dtype=np.int64)
              for s, fields in sets.items() for k, v in fields.items()}
    arrays["popularity"] = popularity
    np.savez_compressed(data / "dataset.npz", **arrays)
    save_json(data / "items.json", items)
    save_json(data / "sequences.json", sequences)
    info = {"dataset": "Amazon Reviews 2014 Beauty 5-core", "users": len(users),
            "items": len(asins), "interactions": sum(len(v) for v in users.values()),
            "examples": {s: len(v["target"]) for s, v in sets.items()},
            "train_interactions": int(popularity.sum()), "dropped_duplicate_user_item": duplicates,
            "missing_metadata": len(asins) - len(meta), "adjacent_timestamp_ties": ties,
            "history_items": history,
            "split": "Per-user leave-last-two-out; NOT a global chronological split; timestamp ties use source line order.",
            "target": "Next reviewed item, using all ratings, NOT click/conversion labels.",
            "catalog": "Fixed transductive catalog; content encoder and quantizer see catalog metadata but no held-out user interactions.",
            "ranking": "Full catalog eligible; remove every previously reviewed item for that user; no sampled negatives.",
            "raw_sha256": {p.name: sha256(p) for p in raw.glob("*.gz")},
            "split_sha256": sha256(data / "dataset.npz")}
    save_json(data / "dataset_info.json", info)
    print("DATASET", json.dumps(info, ensure_ascii=False), flush=True)


def prepare_codes(steps=3000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    data = ROOT / "data"
    emb_path = data / "embeddings.npy"
    if not emb_path.exists():
        items = json.loads((data / "items.json").read_text())
        model = SentenceTransformer(str(data / "sentence-t5-base"), device="cuda", local_files_only=True)
        model.max_seq_length = 128
        print("CONTENT_ENCODER", model, flush=True)
        x = model.encode([i["text"] for i in items], batch_size=128,
                         normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        np.save(emb_path, x.astype(np.float32))
        del model
        torch.cuda.empty_cache()
    x = torch.from_numpy(np.load(emb_path)).cuda()
    assert x.ndim == 2 and x.shape[1] == 768 and bool(torch.isfinite(x).all())
    # A shared translation and scalar preserve distances but improve optimization scale.
    center = x.mean(0, keepdim=True)
    x = (x - center) * (x.shape[1] ** .5)
    model = RQVAE(input_dim=x.shape[1]).cuda()
    pretrain_opt = torch.optim.Adam(model.parameters(), lr=.001)
    for step in range(500):
        batch = x[torch.randint(len(x), (1024,), device=x.device)]
        loss = torch.nn.functional.mse_loss(model.decoder(model.encoder(batch)), batch)
        pretrain_opt.zero_grad(set_to_none=True)
        loss.backward()
        pretrain_opt.step()
        if (step + 1) % 100 == 0:
            print("AUTOENCODER_WARMUP", step + 1, float(loss.detach()), flush=True)
    with torch.no_grad():
        residual = model.encoder(x).cpu().numpy()
        for level, cb in enumerate(model.codebooks):
            km = MiniBatchKMeans(n_clusters=256, batch_size=4096, n_init=3,
                                max_iter=100, random_state=seed + level)
            labels = km.fit_predict(residual)
            cb.copy_(torch.from_numpy(km.cluster_centers_).to(x.device))
            residual = residual - km.cluster_centers_[labels]
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rows = []
    start = time.monotonic()
    for step in range(1, steps + 1):
        batch = x[torch.randint(len(x), (1024,), device=x.device)]
        loss, rec, _ = model(batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite RQ-VAE loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "loss": float(loss), "reconstruction": float(rec),
                   "seconds": time.monotonic() - start}
            rows.append(row)
            print("RQ_VAE", json.dumps(row), flush=True)
    model.eval()
    with torch.no_grad():
        codes = torch.cat([model(b)[2] for b in x.split(1024)]).cpu().numpy()
        full_loss = float(torch.stack([model(b)[0] for b in x.split(1024)]).mean())
    counter, unique = Counter(), []
    for code in codes:
        prefix = tuple(code)
        unique.append([*code.tolist(), counter[prefix]])
        counter[prefix] += 1
    unique = np.asarray(unique, dtype=np.int64)
    if unique[:, 3].max() >= 256:
        raise RuntimeError("Excessive SID collisions: fix quantizer instead of silently truncating IDs")
    assert len(set(map(tuple, unique))) == len(unique)
    np.save(data / "semantic_ids.npy", unique)
    torch.save({"state_dict": model.state_dict(), "content_center": center.cpu(),
                "content_scale": x.shape[1] ** .5}, data / "rqvae.pt")
    with open(data / "rqvae_history.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    stats = {"seed": seed, "steps": steps, "optimizer": "Adam", "lr": .001,
             "autoencoder_warmup_steps": 500, "content_centered": True,
             "content_scale": x.shape[1] ** .5,
             "batch_size": 1024, "latent_dim": 32, "codebooks": 3, "codebook_size": 256,
             "commitment_beta": .25, "full_catalog_loss": full_loss,
             "used_codes": [int(len(np.unique(codes[:, j]))) for j in range(3)],
             "unique_prefixes": len(counter), "max_collision_group": max(counter.values()),
             "collision_fraction": 1 - len(counter) / len(codes), "id_length_with_disambiguation": 4,
             "semantic_ids_sha256": sha256(data / "semantic_ids.npy"),
             "embeddings_sha256": sha256(emb_path)}
    save_json(data / "quantizer_info.json", stats)
    print("QUANTIZER_COMPLETE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--data-only", action="store_true")
    args = p.parse_args()
    if not (ROOT / "data/dataset.npz").exists():
        prepare_data()
    if not args.data_only and not (ROOT / "data/semantic_ids.npy").exists():
        prepare_codes(args.steps)
