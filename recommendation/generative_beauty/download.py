"""Download the public Amazon 2014 Beauty subset and a Sentence-T5 mirror."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download(url, path, expected_sha=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not expected_sha or sha256(path) == expected_sha):
        return {"url": url, "bytes": path.stat().st_size, "sha256": sha256(path)}
    tmp = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        try:
            print("DOWNLOAD", url, flush=True)
            with urllib.request.urlopen(url, timeout=120) as src, open(tmp, "wb") as dst:
                total = 0
                reported = time.monotonic()
                while b := src.read(1024 * 1024):
                    dst.write(b)
                    total += len(b)
                    if time.monotonic() - reported > 15:
                        print(path.name, total, "bytes", flush=True)
                        reported = time.monotonic()
                if src.headers.get("Content-Length"):
                    assert total == int(src.headers["Content-Length"])
            digest = sha256(tmp)
            if expected_sha and digest != expected_sha:
                raise ValueError(f"Checksum mismatch: {path}")
            tmp.replace(path)
            print("DOWNLOADED", path.name, total, digest, flush=True)
            return {"url": url, "bytes": total, "sha256": digest}
        except Exception as exc:
            print("DOWNLOAD_RETRY", attempt + 1, repr(exc), flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    manifest = {}
    raw = ROOT / "data/raw"
    for name in ["reviews_Beauty_5.json.gz", "meta_Beauty.json.gz"]:
        manifest[name] = download(
            "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/" + name,
            raw / name,
        )
    repo = "sentence-transformers/sentence-t5-base"
    api = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
    listing = json.load(urllib.request.urlopen(api, timeout=60))
    files = listing["Data"]["Files"]
    model_dir = ROOT / "data/sentence-t5-base"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "mirror_file_listing.json").write_text(json.dumps(listing, indent=2))
    paths = {x["Path"] for x in files}
    safe = "model.safetensors" in paths
    for f in files:
        name = f["Path"]
        if f.get("Type") == "tree" or name.startswith(("onnx/", "openvino/")):
            continue
        wanted = name.endswith((".json", ".model", ".txt", ".md", ".safetensors"))
        if name == "pytorch_model.bin" and not safe:
            wanted = True
        if name == "2_Dense/pytorch_model.bin" and "2_Dense/model.safetensors" not in paths:
            wanted = True
        if not wanted or name.startswith("."):
            continue
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe model path")
        manifest["model/" + name] = download(
            f"https://modelscope.cn/models/{repo}/resolve/master/{name}", model_dir / name,
            f.get("Sha256") or f.get("SHA256"),
        )
    (ROOT / "data/download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
