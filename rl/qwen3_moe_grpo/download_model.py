"""Download public BF16 weights directly on the AutoDL host, never via a laptop."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from common import load_config, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    root = Path(args.model_path).resolve()
    if sys.platform != 'linux' or not root.is_relative_to('/root/autodl-tmp'):
        raise RuntimeError('Download this model on AutoDL under /root/autodl-tmp only')
    root.mkdir(parents=True, exist_ok=True)
    existing = sum(p.stat().st_size for p in root.rglob('*.safetensors'))
    needed = max(0, 62_000_000_000 - existing) + 12 * 2**30
    if shutil.disk_usage(root).free < needed:
        raise RuntimeError('Insufficient disk for BF16 model plus 12 GiB run headroom')
    from modelscope import snapshot_download
    config = load_config()
    snapshot_download(config['model_id'], local_dir=str(root), max_workers=8,
                      allow_file_pattern=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    model_config = json.loads((root / 'config.json').read_text())
    if model_config.get('model_type') != 'qwen3_moe' or model_config.get('quantization_config'):
        raise RuntimeError('Expected the original unquantized Qwen3 MoE checkpoint')
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    files = sorted(set(index['weight_map'].values()))
    if any(Path(name).name != name or not (root / name).is_file() for name in files):
        raise RuntimeError('Invalid or incomplete model shard index')
    manifest = {}
    for name in files:
        path = root / name
        with path.open('rb') as source:
            checksum = hashlib.file_digest(source, 'sha256').hexdigest()
        manifest[name] = {'bytes': path.stat().st_size, 'sha256': checksum}
    write_json(root / 'download_manifest.json', {'model_id': config['model_id'],
               'source': 'ModelScope public Qwen repository', 'files': manifest})
    print('MODEL_READY', root, flush=True)


if __name__ == '__main__':
    main()
