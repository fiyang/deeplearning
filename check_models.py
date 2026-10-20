"""检查新模型能学习、按图片独立推理、导出特征并正确恢复权重。"""
import io
import json
import torch
from models import MODEL_SPECS, build_model, extract_features


def main():
    torch.set_num_threads(4)
    # 结构检查使用完整 FP32，排除 TF32 卷积在不同批大小下的舍入差异。
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, spec in MODEL_SPECS.items():
        torch.manual_seed(123)
        model = build_model(name).to(device)
        images = torch.randn(2, 3, 32, 32, device=device)
        model.eval()
        with torch.no_grad():
            scores = model(images)
            features = extract_features(model, images, name)
            separate = torch.cat([model(images[i:i+1]) for i in range(2)])
        assert scores.shape == (2, 10)
        assert features.shape == (2, spec["feature_dim"])
        assert torch.isfinite(scores).all() and torch.isfinite(features).all()
        # 如果把 batch 维当成 token 维，单张推理会和批量推理不一致。
        torch.testing.assert_close(scores, separate, atol=2e-5, rtol=2e-4)
        if name.startswith("vit"):
            first, second = model.encoder.layers[:2]
            assert not torch.equal(first.self_attn.in_proj_weight, second.self_attn.in_proj_weight)
            assert not torch.equal(first.linear1.weight, second.linear1.weight)
        model.train()
        loss = torch.nn.functional.cross_entropy(model(images), torch.tensor([1, 2], device=device))
        loss.backward()
        gradients = [p.grad for p in model.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        if name.startswith("vit"):
            assert model.embedding.weight.grad.abs().sum() > 0
            for layer in model.encoder.layers:
                assert layer.self_attn.in_proj_weight.grad.abs().sum() > 0
        model.eval()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = build_model(name).to(device)
        restored.load_state_dict(torch.load(stream, map_location=device, weights_only=True))
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(images), restored(images))
        print(json.dumps({"model": name, "device": device, "feature_shape": list(features.shape),
                          "parameters": sum(p.numel() for p in model.parameters()), "checks": "passed"}), flush=True)
        del model, restored, images, scores, features, separate, loss, gradients
    print("MODEL_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
