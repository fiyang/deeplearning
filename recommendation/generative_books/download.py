"""Download only the official Amazon 2014 Books 5-core and product metadata."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import concurrent.futures
import fcntl
import urllib.request

ROOT = Path(__file__).resolve().parent
FILES = {
    "reviews_Books_5.json.gz": 3223678899,
    "meta_Books.json.gz": 825767823,
}
BASE = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"


def download_ranges(path, size, workers=8):
    """Bounded HTTP range download. Retain any prefix from the earlier serial download."""
    part = path.with_suffix(path.suffix + ".part")
    pieces = path.with_name(path.name + ".ranges")
    pieces.mkdir(exist_ok=True)
    manifest = pieces / "ranges.json"
    if manifest.exists():
        layout = json.loads(manifest.read_text())
        if layout["size"] != size or layout["url"] != BASE + path.name:
            raise RuntimeError("Range resume source mismatch")
        prefix = layout["prefix"]
    else:
        prefix = part.stat().st_size if part.exists() else 0
        if prefix > size:
            raise RuntimeError("Partial download exceeds expected size")
        layout = {"size": size, "prefix": prefix, "url": BASE + path.name,
                  "ranges": [[a, min(a+(16<<20),size)-1] for a in range(prefix,size,16<<20)]}
        save_json(manifest, layout)
    if prefix and (not part.exists() or part.stat().st_size != prefix):
        raise RuntimeError("Partial prefix changed since range download began")

    def fetch(bounds):
        a,b = bounds
        chunk = pieces / f"{a}-{b}.bin"
        if chunk.exists() and chunk.stat().st_size == b-a+1:
            return chunk
        temp = chunk.with_suffix(".tmp")
        for attempt in range(6):
            try:
                request = urllib.request.Request(BASE+path.name, headers={"Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(request,timeout=120) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {a}-{b}/{size}":
                        raise RuntimeError("Server did not honor exact requested byte range")
                    with open(temp,"wb") as output:
                        shutil.copyfileobj(response,output,1<<20)
                if temp.stat().st_size != b-a+1:
                    raise RuntimeError("Incomplete byte range")
                temp.replace(chunk)
                return chunk
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(3*(attempt+1),15))
    completed, start = prefix, time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch,bounds): bounds for bounds in layout["ranges"]}
        while pending:
            done,_ = concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                chunk = future.result()
                completed += chunk.stat().st_size
                del pending[future]
            status = {"file":path.name,"verified_range_bytes":completed,"total_bytes":size,
                      "percent":round(100*completed/size,2),"elapsed_seconds":round(time.monotonic()-start,1),
                      "workers":workers,"remaining_ranges":len(pending)}
            save_json(ROOT / "data/download_status.json",status)
            print("DOWNLOAD_PROGRESS",json.dumps(status),flush=True)
    assembled = path.with_suffix(path.suffix+".assembled")
    with open(assembled,"wb") as output:
        if prefix:
            with open(part,"rb") as source:
                shutil.copyfileobj(source,output,8<<20)
        for a,b in layout["ranges"]:
            with open(pieces/f"{a}-{b}.bin","rb") as source:
                shutil.copyfileobj(source,output,8<<20)
    if assembled.stat().st_size != size:
        raise RuntimeError("Assembled byte count mismatch")
    assembled.replace(path)
    # Only this downloader's verified temporary pieces are removed.
    if part.exists():
        part.unlink()
    for a,b in layout["ranges"]:
        (pieces/f"{a}-{b}.bin").unlink()
    manifest.unlink()
    pieces.rmdir()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    lock = open(raw / ".download.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    required = sum(size for name, size in FILES.items() if not (raw / name).exists())
    if shutil.disk_usage(raw).free < required + 15 * 2**30:
        raise RuntimeError("Insufficient free space: downloads plus 15 GiB working reserve required")
    manifest = []
    for name, size in FILES.items():
        path = raw / name
        if not path.exists():
            print("DOWNLOAD_START", name, size, flush=True)
            download_ranges(path,size)
        if path.stat().st_size != size:
            raise RuntimeError(f"Wrong existing length: {name}")
        entry = {"file": name, "url": BASE + name, "bytes": size, "sha256": sha256(path)}
        manifest.append(entry)
        save_json(ROOT / "data/download_manifest.json", manifest)
        print("DOWNLOAD_VERIFIED", json.dumps(entry), flush=True)
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
