"""验证新训练流程的学习率边界、混合标签和 BF16 梯度。"""
import argparse
import json
import torch
from torchvision.transforms import v2
from models import build_model
from train_vit import learning_rate, mixing_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vit_tiny12", "vit_small12"], default="vit_tiny12")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("batch-size must be at least 2")
    torch.set_num_threads(4)
    assert learning_rate(0, 17600, 880, 2.5e-4) == 1e-6
    assert abs(learning_rate(879, 17600, 880, 2.5e-4) - 2.5e-4) < 1e-12
    assert abs(learning_rate(880, 17600, 880, 2.5e-4) - 2.5e-4) < 1e-12
    assert abs(learning_rate(17599, 17600, 880, 2.5e-4) - 1e-5) < 1e-12
    rates = [learning_rate(i, 17600, 880, 2.5e-4) for i in range(17600)]
    assert all(a <= b for a, b in zip(rates[:879], rates[1:880]))
    assert all(a >= b for a, b in zip(rates[880:-1], rates[881:]))
    # 常量图片 0/1 对应类别 0/1；混合后像素均值应等于类别 1 的概率。
    images = torch.stack([torch.zeros(3, 32, 32), torch.ones(3, 32, 32)])
    labels = torch.tensor([0, 1])
    for mixing in [v2.MixUp(alpha=0.8, num_classes=10), v2.CutMix(alpha=1.0, num_classes=10)]:
        for seed in range(5):
            torch.manual_seed(seed)
            mixed, targets = mixing(images.clone(), labels)
            torch.testing.assert_close(mixed.mean((1, 2, 3)), targets[:, 1])
            torch.testing.assert_close(targets.sum(1), torch.ones(2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    model = build_model(args.model).to(device).train()
    assert len(model.encoder.layers) == 12
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4)
    mixed, targets = mixing_transform()(torch.randn(args.batch_size, 3, 32, 32), torch.arange(args.batch_size) % 10)
    old_embedding = model.embedding.weight.detach().clone()
    with torch.autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
        scores = model(mixed.to(device))
    loss = torch.nn.functional.cross_entropy(scores.float(), targets.to(device), label_smoothing=0.1)
    smooth_targets = targets.to(device) * 0.9 + 0.1 / 10
    expected = -(smooth_targets * scores.float().log_softmax(1)).sum(1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(layer.self_attn.in_proj_weight.grad.abs().sum() > 0 for layer in model.encoder.layers)
    optimizer.step()
    assert not torch.equal(old_embedding, model.embedding.weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    print(json.dumps({"checks": "passed", "model": args.model, "device": device,
                      "layers": 12, "batch_size": args.batch_size, "soft_target_loss": loss.item(),
                      "parameters": sum(p.numel() for p in model.parameters()), "bf16": use_bf16}))
    print("VIT_TRAINING_CHECKS_PASSED")


if __name__ == "__main__":
    main()
