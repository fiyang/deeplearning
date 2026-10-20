"""加载保存的模型，对一张图片进行预测。"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from train import SmallCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = SmallCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    config = checkpoint["config"]
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                    transforms.Normalize(config["mean"], config["std"])])
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        values, indices = model(tensor).softmax(dim=1)[0].topk(3)
    print(json.dumps({"image": str(args.image), "top3": [
        {"class": checkpoint["classes"][index], "score": value}
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())]}, indent=2))


if __name__ == "__main__":
    main()
