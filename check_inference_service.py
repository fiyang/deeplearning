"""用离线预测与保存特征验证真实 HTTP 服务，并检查并发和非法输入。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8008")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def request(path, body=None):
        req = urllib.request.Request(args.url + path, data=body,
                                     headers={"Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)
    status, health = request("/health")
    assert status == 200 and health["status"] == "ready"
    expected = json.loads((args.run / "inference_check.json").read_text())
    expected_features = np.load(args.run / "sample_features.npy")[0]
    image = (args.run / "sample_test_image.png").read_bytes()
    status, result = request("/predict?include_features=true", image)
    assert status == 200 and result["model"] == expected["model"]
    assert len(result["features"]) == expected["feature_shape"][1]
    np.testing.assert_allclose(result["features"], expected_features, atol=1e-5, rtol=1e-4)
    assert [p["class"] for p in result["top_k"]] == [p["class"] for p in expected["top3"]]
    np.testing.assert_allclose([p["score"] for p in result["top_k"]],
                               [p["score"] for p in expected["top3"]], atol=1e-5, rtol=1e-4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda _: request("/predict", image), range(12)))
    for code, prediction in concurrent:
        assert code == 200 and prediction["top_k"] == result["top_k"]
    invalid = {}
    for label, path, body, expected_status in [
        ("empty", "/predict", b"", 400), ("not_image", "/predict", b"not a picture", 400),
        ("oversized_body", "/predict", b"x" * (8 * 1024**2 + 1), 413),
        ("invalid_top_k", "/predict?top_k=11", image, 422),
    ]:
        status, response = request(path, body)
        assert status == expected_status, (label, status, response)
        invalid[label] = status
    for mode, format_ in [("L", "PNG"), ("RGB", "JPEG"), ("RGBA", "PNG"), ("RGB", "WEBP")]:
        buffer = BytesIO()
        with Image.open(BytesIO(image)) as picture:
            picture.convert(mode).save(buffer, format=format_)
        status, response = request("/predict?top_k=10&include_features=true", buffer.getvalue())
        assert status == 200
        assert np.isfinite(response["features"]).all()
        assert abs(sum(p["score"] for p in response["top_k"]) - 1) < 1e-5
    report = {"status": "passed", "health": health, "reference_run": str(args.run),
              "checks": ["offline class and scores match", "offline feature vector matches",
                         "12 concurrent calls are deterministic", "invalid input rejected",
                         "grayscale/JPEG/RGBA/WebP handled"],
              "invalid_input_status": invalid, "sample_prediction": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": report["checks"], "status": "passed", "prediction": result["top_k"]}))
    print("INFERENCE_SERVICE_CHECKS_PASSED")


if __name__ == "__main__":
    main()
