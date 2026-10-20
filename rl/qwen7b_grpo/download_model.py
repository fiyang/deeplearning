import hashlib
import json
from pathlib import Path
from modelscope import snapshot_download

root = Path('/root/autodl-tmp/qwen7b-grpo-model')
path = snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir=str(root),
                         allow_file_pattern=['*.json', '*.safetensors', '*.txt', '*.model'])
weights = sorted(root.glob('*.safetensors'))
if not weights:
    raise RuntimeError('Model download did not produce safetensors weights')
manifest = {p.name: {'size': p.stat().st_size,
                    'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()}
            for p in sorted(root.iterdir()) if p.is_file() and not p.name.startswith('.')}
(root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
print('MODEL_READY', path, flush=True)
