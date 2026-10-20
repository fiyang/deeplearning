"""与历史实验的代码快照对比，验证旧 ViT 初始化及 state_dict 保持兼容。"""
import argparse
import importlib.util
import torch
from models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("historical_vit_models", args.baseline_source)
    old_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_models)
    for name in ["vit", "vit_tiny12"]:
        torch.manual_seed(42)
        before = old_models.build_model(name).eval()
        torch.manual_seed(42)
        after = build_model(name).eval()
        old_state, new_state = before.state_dict(), after.state_dict()
        assert old_state.keys() == new_state.keys()
        assert all(torch.equal(old_state[key], new_state[key]) for key in old_state)
        after.load_state_dict(old_state, strict=True)
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(before(images), after(images), rtol=0, atol=0)
        print(f"{name}: initialization, state_dict, inference identical", flush=True)
    print("VIT_COMPATIBILITY_CHECKS_PASSED")


if __name__ == "__main__":
    main()
