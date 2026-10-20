"""按类别固定随机抽样，比较 HTTP 推理准确率与冻结源码的离线预测。"""
import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import http.client
import importlib.util
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8008")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    if not 1 <= args.per_class <= 1000:
        parser.error("per-class must be in 1..1000")
    url = urlsplit(args.url)
    if url.scheme != "http" or url.path not in {"", "/"}:
        parser.error("Use an HTTP base URL")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.run / "best.pt"
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config, classes = checkpoint["config"], checkpoint["classes"]
    dataset = datasets.CIFAR10(str(args.data_dir), train=False, download=False)
    assert dataset.classes == classes and len(dataset) == 10000
    labels = np.asarray(dataset.targets)
    rng = np.random.default_rng(args.seed)
    indices = np.concatenate([rng.choice(np.flatnonzero(labels == label), args.per_class, replace=False)
                              for label in range(10)])
    rng.shuffle(indices)
    selection = {"dataset": "CIFAR-10 original test set", "seed": args.seed,
                 "per_class": args.per_class, "indices": indices.tolist()}
    # 在看到任何预测前保存抽样名单；不按预测正确与否挑选样本。
    (args.output / "sample_indices.json").write_text(json.dumps(selection, indent=2))
    records, failures, all_scores = [], [], []
    with closing(http.client.HTTPConnection(url.hostname, url.port or 80, timeout=30)) as client:
        client.request("GET", "/health")
        response = client.getresponse()
        assert response.status == 200
        health = json.loads(response.read())
        assert health["checkpoint_sha256"] == digest and health["model"] == config["model"]
        for index in indices:
            image, target = dataset[int(index)]
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            client.request("POST", "/predict?top_k=10", body=buffer.getvalue(),
                           headers={"Content-Type": "image/png"})
            response = client.getresponse()
            body = response.read()
            if response.status != 200:
                failures.append({"index": int(index), "status": response.status,
                                 "body": body.decode(errors="replace")[:200]})
                continue
            prediction = json.loads(body)
            assert prediction["model"] == config["model"] and len(prediction["top_k"]) == 10
            score_map = {p["class"]: p["score"] for p in prediction["top_k"]}
            scores = [score_map[label] for label in classes]
            assert np.isfinite(scores).all() and abs(sum(scores) - 1) < 1e-5
            predicted = classes.index(prediction["top_k"][0]["class"])
            records.append({"test_index": int(index), "true": classes[target],
                            "predicted": classes[predicted], "correct": predicted == target,
                            "top1_score": prediction["top_k"][0]["score"],
                            "top3_correct": classes[target] in [p["class"] for p in prediction["top_k"][:3]]})
            all_scores.append(scores)
            if len(records) % 100 == 0:
                print(f"HTTP_CHECK {len(records)}/{len(indices)}", flush=True)
    (args.output / "http_predictions.json").write_text(json.dumps({"records": records, "failures": failures}, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} HTTP failures; predictions saved")
    # 使用训练时保存的 models.py，独立加载同一权重；先完成 HTTP，再离线计算，避免竞争 GPU。
    spec = importlib.util.spec_from_file_location("frozen_accuracy_models", args.run / "source/models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = module.build_model(config["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])
    offline_scores = []
    with torch.inference_mode():
        for offset in range(0, len(indices), 32):
            tensor = torch.stack([transform(dataset[int(i)][0]) for i in indices[offset:offset + 32]])
            offline_scores.extend(model(tensor.to(device)).softmax(1).cpu().tolist())
    api = np.asarray(all_scores)
    offline = np.asarray(offline_scores)
    correct = sum(row["correct"] for row in records)
    agreement = int((api.argmax(1) == offline.argmax(1)).sum())
    per_class = {label: {"correct": sum(r["correct"] for r in records if r["true"] == label),
                         "count": args.per_class} for label in classes}
    confusion = [[sum(r["true"] == a and r["predicted"] == b for r in records) for b in classes]
                 for a in classes]
    report = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "sample_plan": selection, "health": health, "http_successes": len(records),
              "http_failures": failures, "correct": correct, "count": len(indices),
              "accuracy": correct / len(indices), "top3_accuracy": sum(r["top3_correct"] for r in records) / len(indices),
              "offline_accuracy": float((offline.argmax(1) == labels[indices]).mean()),
              "offline_top1_agreement_count": agreement,
              "max_probability_difference": float(np.abs(api - offline).max()),
              "per_class": per_class, "confusion_matrix": confusion,
              "reference_full_test_accuracy": json.loads((args.run / "summary.json").read_text())["test_accuracy"],
              "note": "Fixed stratified sample of the existing CIFAR-10 test set; an API deployment check, not a new independent test set or a test of arbitrary uploaded photos."}
    (args.output / "accuracy_summary.json").write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 每类按抽样顺序展示第一张，另展示前六个错误，错误样例明确标注。
    selected = [next(r for r in records if r["true"] == label) for label in classes]
    errors = [r for r in records if not r["correct"]][:6]
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax in axes.flat:
        ax.axis("off")
    for number, row in enumerate(selected + errors):
        ax = axes.flat[number]
        ax.imshow(dataset[row["test_index"]][0], interpolation="nearest")
        prefix = "Sample" if number < 10 else "Error"
        ax.set_title(f"{prefix} #{row['test_index']} | True: {row['true']}\nPred: {row['predicted']} ({row['top1_score']:.1%})",
                     color="#116149" if row["correct"] else "#b91c1c", fontsize=9)
    fig.suptitle(f"Live HTTP inference | {correct}/{len(indices)} correct ({correct / len(indices):.1%})\nFirst sampled image per class + first six errors", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .945))
    fig.savefig(args.output / "prediction_examples.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k not in {"sample_plan", "health", "confusion_matrix"}}, indent=2), flush=True)
    assert agreement == len(indices), "HTTP and offline top-1 predictions differ; inspect result files"
    np.testing.assert_allclose(api, offline, atol=1e-5, rtol=1e-4)
    print("INFERENCE_ACCURACY_CHECK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
