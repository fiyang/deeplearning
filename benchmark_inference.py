"""实际 HTTP 压测；复用连接，分别记录端到端和服务内部计时。无需第三方依赖。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import statistics
import threading
import time
from urllib.parse import urlsplit


def percentiles(values):
    values = sorted(values)
    if not values:
        return None
    def at(q):
        x = (len(values) - 1) * q
        lo = int(x)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (x - lo)
    return {"mean": statistics.mean(values), "p50": at(.5), "p95": at(.95), "p99": at(.99)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8008")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--context", required=True, help="GPU and network conditions for this run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < args.concurrency or args.concurrency < 1 or args.warmup < 0:
        parser.error("Require requests >= concurrency >= 1 and warmup >= 0")
    if args.output.exists():
        raise FileExistsError(args.output)
    url = urlsplit(args.url)
    if url.scheme != "http" or url.path not in {"", "/"}:
        parser.error("Use an HTTP base URL for the SSH-protected service")
    payload = args.image.read_bytes()
    def connection():
        return http.client.HTTPConnection(url.hostname, url.port or 80, timeout=30)
    def predict(client):
        start = time.perf_counter()
        client.request("POST", "/predict", body=payload, headers={"Content-Type": "application/octet-stream"})
        response = client.getresponse()
        body = response.read()
        elapsed = (time.perf_counter() - start) * 1000
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:200]!r}")
        data = json.loads(body)
        return {"http_ms": elapsed, "timing_ms": data["timing_ms"], "class": data["top_k"][0]["class"]}
    with closing(connection()) as client:
        client.request("GET", "/health")
        response = client.getresponse()
        assert response.status == 200
        health = json.loads(response.read())
        for _ in range(args.warmup):
            predict(client)
    barrier = threading.Barrier(args.concurrency + 1)
    def worker(index):
        samples, errors = [], []
        count = args.requests // args.concurrency + (index < args.requests % args.concurrency)
        with closing(connection()) as client:
            barrier.wait()
            for _ in range(count):
                try:
                    samples.append(predict(client))
                except Exception as error:
                    errors.append(str(error))
                    client.close()
        return samples, errors
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        tasks = [pool.submit(worker, i) for i in range(args.concurrency)]
        start = time.perf_counter()
        barrier.wait()
        results = [task.result() for task in tasks]
        seconds = time.perf_counter() - start
    samples = [row for rows, _ in results for row in rows]
    errors = [error for _, failures in results for error in failures]
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "context": args.context,
              "url": args.url, "image_bytes": len(payload), "batch_size": 1,
              "requests": args.requests, "concurrency": args.concurrency, "warmup": args.warmup,
              "successes": len(samples), "errors": errors, "elapsed_seconds": seconds,
              "requests_per_second": len(samples) / seconds,
              "http_latency_ms": percentiles([x["http_ms"] for x in samples]),
              "timing_ms": {key: percentiles([x["timing_ms"][key] for x in samples
                                             if x["timing_ms"][key] is not None])
                            for key in samples[0]["timing_ms"]} if samples else {},
              "health": health, "samples": samples,
              "method": "One fixed PNG; warm service; HTTP keep-alive; closed-loop fixed concurrency. No dynamic batching. service_total excludes response serialization and network."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in {"samples", "health"}}, indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} failed requests; see report")


if __name__ == "__main__":
    main()
