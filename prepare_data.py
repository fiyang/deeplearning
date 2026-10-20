"""从 AutoDL 公共盘准备 CIFAR-10，不需要从公网下载。"""
import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "/root/autodl-pub/cifar-10/cifar-10-python.tar.gz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / args.source.name
    expected = "c58f30108f718f92721af3b95e74349a"  # 数据集官网的完整性校验值。
    if not archive.exists() or md5(archive) != expected:
        partial = archive.with_suffix(archive.suffix + ".partial")
        print(f"Copying {args.source} -> {archive}", flush=True)
        shutil.copyfile(args.source, partial)
        if md5(partial) != expected:
            raise RuntimeError("Archive checksum mismatch; extraction stopped.")
        partial.replace(archive)
    print("Archive checksum OK.", flush=True)
    from torchvision.datasets import CIFAR10
    try:
        train = CIFAR10(str(args.data_dir), train=True, download=False)
        test = CIFAR10(str(args.data_dir), train=False, download=False)
    except RuntimeError:
        print("Extracting verified archive into the data disk...", flush=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(args.data_dir, filter="data")
        train = CIFAR10(str(args.data_dir), train=True, download=False)
        test = CIFAR10(str(args.data_dir), train=False, download=False)
    print(f"Ready: {len(train)} training-pool images, {len(test)} test images.", flush=True)


if __name__ == "__main__":
    main()
