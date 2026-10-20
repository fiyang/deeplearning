"""Full Books data with compact prefix indices; no sampled training subset."""
import argparse
import ast
from collections import Counter, defaultdict
import gc
import gzip
import hashlib
import html
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer

from download import ROOT, save_json, sha256
from models import RQVAE

DATA = ROOT / "data"
CONTENT_MODEL = ROOT.parent / "generative_beauty/data/sentence-t5-base"
ARRAY_NAMES = ("sequence_items", "sequence_offsets", "sequence_timestamps", "user_hash",
               "train_position", "train_user", "val_position", "test_position", "popularity")


def unique_semantic_ids(codes, collision_size=1024):
    count, sid = Counter(), []
    for row in codes:
        key = tuple(row)
        sid.append([*row.tolist(), count[key]])
        count[key] += 1
    sid = np.asarray(sid, dtype=np.int64)
    if sid[:, 3].max() >= collision_size:
        raise RuntimeError(f"Collision group too large ({sid[:,3].max()+1}); repair tokenizer before training")
    if len(np.unique(sid, axis=0)) != len(sid):
        raise AssertionError("Non-unique item identifiers")
    return sid, count


def prefix_examples(sequences):
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in sequences])]).astype(np.int64)
    if any(len(s) < 5 for s in sequences):
        raise ValueError("Every retained sequence must have at least five distinct items")
    positions = np.concatenate([np.arange(offsets[u]+1, offsets[u+1]-2, dtype=np.int64)
                                for u in range(len(sequences))])
    owners = np.repeat(np.arange(len(sequences), dtype=np.int32), np.diff(offsets)-3)
    return offsets, positions, owners, offsets[1:]-2, offsets[1:]-1


def prepare_data():
    users = defaultdict(list)
    item_map, asins = {}, []
    raw_count = 0
    with gzip.open(DATA / "raw/reviews_Books_5.json.gz", "rt") as f:
        for line_index, line in enumerate(f):
            row = json.loads(line)
            asin = row["asin"]
            if asin not in item_map:
                item_map[asin] = len(asins)+1
                asins.append(asin)
            users[row["reviewerID"]].append((int(row["unixReviewTime"]), line_index, item_map[asin]))
            raw_count += 1
            if raw_count % 500000 == 0:
                print("READ_REVIEWS", raw_count, flush=True)
    sequences, timestamps, hashes, keys = [], [], [], []
    duplicate, dropped_users, ties = 0, 0, 0
    for user in sorted(users):
        events, seen = [], set()
        for event in sorted(users[user]):
            if event[2] in seen:
                duplicate += 1
                continue
            seen.add(event[2])
            events.append(event)
        if len(events) < 5:
            dropped_users += 1
            continue
        sequences.append([e[2] for e in events])
        timestamps.append([e[0] for e in events])
        hashes.append(int.from_bytes(hashlib.sha256(user.encode()).digest()[:8], "big") % 2000)
        keys.append(user)
        ties += sum(events[i][0] == events[i-1][0] for i in range(1, len(events)))
    del users
    gc.collect()
    offsets, train_position, train_user, val_position, test_position = prefix_examples(sequences)
    flat = np.concatenate(sequences).astype(np.int32)
    time_flat = np.concatenate(timestamps).astype(np.int64)
    popularity = np.bincount(flat-1, minlength=len(asins)).astype(np.int64)
    np.add.at(popularity, flat[val_position]-1, -1)
    np.add.at(popularity, flat[test_position]-1, -1)
    arrays = dict(sequence_items=flat, sequence_offsets=offsets, sequence_timestamps=time_flat,
                  user_hash=np.asarray(hashes, dtype=np.int64), train_position=train_position,
                  train_user=train_user, val_position=val_position, test_position=test_position,
                  popularity=popularity)
    for name, array in arrays.items():
        np.save(DATA / f"{name}.npy", array)
    save_json(DATA / "user_ids.json", keys)
    del sequences, timestamps, arrays, keys
    gc.collect()
    items = [{"asin": asin, "title": "", "brand": "", "price": "", "categories": [],
              "description": "", "text": ""} for asin in asins]
    found = set()
    with gzip.open(DATA / "raw/meta_Books.json.gz", "rt") as f:
        for line_index, line in enumerate(f):
            row = ast.literal_eval(line)
            asin = row.get("asin")
            if asin in item_map:
                categories = sorted({str(c) for path in row.get("categories", []) for c in path})
                title = html.unescape(str(row.get("title", ""))).strip()
                description = html.unescape(str(row.get("description", "")))[:1000]
                fields = {"asin": asin, "title": title, "brand": str(row.get("brand", "")),
                          "price": str(row.get("price", "")), "categories": categories,
                          "description": description}
                fields["text"] = ". ".join([title, "Category: " + ", ".join(categories),
                                               "Brand: " + fields["brand"], "Price: " + fields["price"],
                                               description])
                items[item_map[asin]-1] = fields
                found.add(asin)
            if (line_index+1) % 250000 == 0:
                print("READ_METADATA", line_index+1, "matched", len(found), flush=True)
    # Keep the entire observed catalog. Missing content remains explicitly marked.
    for item in items:
        if not item["text"].strip():
            item["text"] = "Book. Metadata unavailable."
    save_json(DATA / "items.json", items)
    info = {"dataset": "Amazon Reviews 2014 Books 5-core", "raw_interactions": raw_count,
            "users": len(hashes), "items": len(asins), "interactions": len(flat),
            "train_interactions": int(popularity.sum()), "train_examples": len(train_position),
            "validation_examples": len(val_position), "test_examples": len(test_position),
            "dropped_duplicate_user_item": duplicate, "dropped_users_under_5": dropped_users,
            "missing_metadata": len(asins)-len(found), "missing_titles": sum(not i["title"] for i in items),
            "adjacent_timestamp_ties": ties, "history_items": 20,
            "split": "per-user chronological leave-last-two-out; ties break by source row; not global time split",
            "target": "next reviewed item at any rating, not click/conversion or positive-rating prediction",
            "candidate_catalog": "all observed items, excluding ALL historical items before the target",
            "content": "title, category, brand, price, description; no reviews or co-purchase graphs",
            "transductive": "known complete catalog metadata for tokenization; held-out interactions unused",
            "data_sha256": {name: sha256(DATA / f"{name}.npy") for name in ARRAY_NAMES},
            "items_sha256": sha256(DATA / "items.json")}
    save_json(DATA / "dataset_info.json", info)
    print("DATA_COMPLETE", json.dumps(info), flush=True)


def tokenize(seed=42, steps=10000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    emb_path = DATA / "embeddings.npy"
    if not emb_path.exists():
        items = json.loads((DATA / "items.json").read_text())
        encoder = SentenceTransformer(str(CONTENT_MODEL), device="cuda", local_files_only=True)
        encoder.max_seq_length = 128
        x = encoder.encode([i["text"] for i in items], batch_size=128, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
        np.save(emb_path, x.astype(np.float32))
        del encoder, items, x
        torch.cuda.empty_cache()
    x = torch.from_numpy(np.load(emb_path)).cuda()
    assert x.ndim == 2 and x.shape[1] == 768 and bool(torch.isfinite(x).all())
    center = x.mean(0, keepdim=True)
    x = (x-center) * (768 ** .5)
    model = RQVAE().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for step in range(1000):
        batch = x[torch.randint(len(x), (1024,), device=x.device)]
        loss = torch.nn.functional.mse_loss(model.decoder(model.encoder(batch)), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1) % 200 == 0:
            print("AUTOENCODER", step+1, float(loss.detach()), flush=True)
    with torch.no_grad():
        sample = torch.randperm(len(x), device=x.device)[:min(200000, len(x))]
        residual = torch.cat([model.encoder(b) for b in x[sample].split(4096)]).cpu().numpy()
        for level, codebook in enumerate(model.codebooks):
            km = MiniBatchKMeans(n_clusters=512, batch_size=4096, n_init=3, max_iter=100,
                                random_state=seed+level)
            labels = km.fit_predict(residual)
            codebook.copy_(torch.from_numpy(km.cluster_centers_).cuda())
            residual -= km.cluster_centers_[labels]
            print("CODEBOOK_INITIALIZED", level, flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history, started = [], time.monotonic()
    for step in range(1, steps+1):
        batch = x[torch.randint(len(x), (1024,), device=x.device)]
        loss, rec, _ = model(batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite quantizer loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            row = {"step": step, "loss": float(loss.detach()), "reconstruction": float(rec.detach()),
                   "seconds": time.monotonic()-started}
            history.append(row)
            save_json(DATA / "quantizer_history.json", history)
            print("RQ_VAE", json.dumps(row), flush=True)
    model.eval()
    with torch.no_grad():
        code_batches, squared_error = [], 0.
        for batch in x.split(2048):
            _, rec, codes = model(batch)
            squared_error += float(rec)*len(batch)
            code_batches.append(codes.cpu().numpy())
    codes = np.concatenate(code_batches)
    np.save(DATA / "base_codes.npy", codes)
    torch.save({"state_dict": model.state_dict(), "content_center": center.cpu(),
                "content_scale": 768**.5}, DATA / "rqvae.pt")
    sid, count = unique_semantic_ids(codes, collision_size=1024)
    np.save(DATA / "semantic_ids.npy", sid)
    info = {"seed": seed, "steps": steps, "warmup_steps": 1000, "optimizer": "Adam", "lr": .001,
            "batch_size": 1024, "latent_dim": 64, "codebooks": 3, "codebook_size": 512,
            "commitment_beta": .25, "reconstruction_mse": squared_error/len(codes),
            "used_codes": [int(len(np.unique(codes[:, j]))) for j in range(3)],
            "unique_three_code_prefixes": len(count), "max_collision_group": max(count.values()),
            "items": len(codes), "id_length": 4, "all_ids_unique": True,
            "collision_token_size": 1024, "token_cardinalities": [512,512,512,1024],
            "semantic_ids_sha256": sha256(DATA / "semantic_ids.npy"),
            "embeddings_sha256": sha256(emb_path),
            "sentence_t5_model_manifest": str(CONTENT_MODEL.parent / "download_manifest.json")}
    save_json(DATA / "quantizer_info.json", info)
    print("TOKENIZER_COMPLETE", json.dumps(info), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--tokenize-only", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if not args.tokenize_only and not (DATA / "dataset_info.json").exists():
        prepare_data()
    if not args.data_only and not (DATA / "quantizer_info.json").exists():
        tokenize()


if __name__ == "__main__":
    main()
