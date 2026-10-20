"""从对比实验的 checkpoint 预测图片，并可导出分类器之前的特征向量。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import build_model, extract_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--features-output", type=Path)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    name = checkpoint["config"].get("model", "smallcnn")
    model = build_model(name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    config = checkpoint["config"]
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"])])
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = extract_features(model, tensor, name).cpu().numpy()
        values, indices = model(tensor).softmax(dim=1)[0].topk(3)
    if args.features_output:
        if args.features_output.exists():
            raise FileExistsError(args.features_output)
        with args.features_output.open("xb") as handle:
            np.save(handle, features)
    print(json.dumps({"model": name, "feature_shape": list(features.shape),
        "features_finite": bool(np.isfinite(features).all()),
        "top3": [{"class": checkpoint["classes"][index], "score": value}
                 for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())]}, indent=2))


if __name__ == "__main__":
    main()
