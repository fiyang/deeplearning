"""CIFAR-10 ViT 推理服务：加载一次权重，通过 HTTP 接收一张图片。"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time
import warnings

from fastapi import FastAPI, HTTPException, Query, Request
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
import torch
from torchvision import transforms
import uvicorn

from models import MODEL_SPECS, build_model

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000


class Predictor:
    def __init__(self, checkpoint_path, device):
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.config = checkpoint["config"]
        self.name = self.config["model"]
        if self.name not in {"vit_tiny12", "vit_small12"}:
            raise ValueError("This service supports the two 12-layer ViT models")
        self.classes = checkpoint["classes"]
        self.model = build_model(self.name)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize(self.config["mean"], self.config["std"])])
        self.metadata = {
            "status": "ready", "model": self.name, "precision": "fp32",
            "feature_dim": MODEL_SPECS[self.name]["feature_dim"],
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
            "best_epoch": checkpoint["best_epoch"],
            "validation_accuracy": checkpoint["validation_accuracy"],
            "classes": self.classes, "input_size": [3, 32, 32],
            "pid": os.getpid(), "device": str(self.device), "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
        }
        # 模型常驻显存；启动时预热，常规延迟不包含冷启动。
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 32, 32, device=self.device)
            for _ in range(20):
                self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def health(self):
        data = dict(self.metadata)
        if self.device.type == "cuda":
            data.update(gpu_allocated_mib=torch.cuda.memory_allocated(self.device) / 1024**2,
                        gpu_reserved_mib=torch.cuda.memory_reserved(self.device) / 1024**2,
                        gpu_peak_allocated_mib=torch.cuda.max_memory_allocated(self.device) / 1024**2)
        return data

    @torch.inference_mode()
    def predict(self, payload, top_k, include_features):
        begin = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(payload)) as image:
                    if image.format not in {"PNG", "JPEG", "WEBP"}:
                        raise HTTPException(415, "Use PNG, JPEG or WebP")
                    if image.width * image.height > MAX_PIXELS:
                        raise HTTPException(413, "Image exceeds 16 million pixels")
                    image.load()
                    original_size = list(image.size)
                    tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as error:
            raise HTTPException(400, "Invalid or oversized image") from error
        preprocessed = time.perf_counter()
        inputs = tensor.to(self.device)
        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        features = self.model.forward_features(inputs)
        logits = self.model.classifier(features)
        if self.device.type == "cuda":
            end.record()
            end.synchronize()
            model_cuda_ms = start.elapsed_time(end)
        else:
            model_cuda_ms = None
        inferred = time.perf_counter()
        values, indices = logits.float().softmax(1)[0].topk(top_k)
        predictions = [{"class": self.classes[i], "score": v}
                       for v, i in zip(values.cpu().tolist(), indices.cpu().tolist())]
        result = {"model": self.name, "original_size": original_size, "top_k": predictions,
                  "feature_dim": int(features.shape[1]), "precision": "fp32"}
        if include_features:
            result["features"] = features[0].cpu().tolist()
        result["timing_ms"] = {
            "preprocess": (preprocessed - begin) * 1000,
            "inference_wall": (inferred - preprocessed) * 1000,
            "model_cuda": model_cuda_ms,
            "postprocess": (time.perf_counter() - inferred) * 1000,
        }
        return result


def create_app(checkpoint_path, device="cuda"):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = Predictor(checkpoint_path, device)
        app.state.gpu_lock = asyncio.Lock()
        yield
        del app.state.predictor

    app = FastAPI(title="CIFAR-10 ViT inference", version="1.0", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        return request.app.state.predictor.health()

    @app.post("/predict", openapi_extra={"requestBody": {"required": True, "content": {
        "image/png": {"schema": {"type": "string", "format": "binary"}},
        "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
    }}})
    async def predict(request: Request, top_k: int = Query(3, ge=1, le=10),
                      include_features: bool = False):
        started = time.perf_counter()
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > MAX_BYTES:
                raise HTTPException(413, "Upload exceeds 8 MiB")
            payload.extend(chunk)
        if not payload:
            raise HTTPException(400, "Upload an image as the request body")
        received = time.perf_counter()
        # 单个常驻模型按请求串行执行；并发请求排队，不复制 GPU 模型。
        async with request.app.state.gpu_lock:
            acquired = time.perf_counter()
            result = await run_in_threadpool(request.app.state.predictor.predict,
                                            bytes(payload), top_k, include_features)
        result["timing_ms"].update(
            read_body=(received - started) * 1000, queue_wait=(acquired - received) * 1000,
            service_total=(time.perf_counter() - started) * 1000)
        return result

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    uvicorn.run(create_app(args.checkpoint, args.device), host="127.0.0.1", port=args.port,
                workers=1, access_log=False, limit_concurrency=64, timeout_keep_alive=30)
