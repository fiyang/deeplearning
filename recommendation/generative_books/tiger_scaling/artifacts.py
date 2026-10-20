"""Local artifact helpers; data remains in the parent Books experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def gradient_report(model):
    import torch
    report = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is not None:
            finite = torch.isfinite(grad)
            report.append({"parameter": name, "nonfinite_gradient_count": int((~finite).sum()),
                "max_finite_gradient_abs": float(grad[finite].abs().max()) if bool(finite.any()) else None,
                "gradient_norm_float64": float(grad.double().norm()) if bool(finite.all()) else None,
                "parameter_finite": bool(torch.isfinite(param).all())})
    return report
