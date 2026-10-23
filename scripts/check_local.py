"""Offline syntax, provenance and lightweight existing tests; no GPU/downloads."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '.venvs', '__pycache__', 'data', 'runs', 'checkpoints'}


def main():
    counts = {'python': 0, 'shell': 0, 'json': 0}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or EXCLUDED.intersection(path.relative_to(ROOT).parts):
            continue
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
            counts['python'] += 1
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
            counts['shell'] += 1
        elif path.suffix == '.json':
            json.loads(path.read_text())
            counts['json'] += 1
    official = ROOT / 'recommendation/aitm_aliccp_official/official'
    manifest = json.loads((official / 'manifest.json').read_text())
    for row in manifest['files']:
        actual = hashlib.sha256((official / row['file']).read_bytes()).hexdigest()
        if actual != row['sha256']:
            raise ValueError(f'Official reference hash mismatch: {row["file"]}')
    print('SYNTAX_AND_REFERENCE_OK', json.dumps(counts), flush=True)
    groups = [
        ('rl/qwen7b_grpo', ['test_rewards.py']),
        ('rl/qwen3_moe_grpo', ['test_experiment.py', 'test_eval_full.py',
                             'test_round2.py', 'test_signal_audit.py', 'test_signal_screen.py']),
        ('recommendation/esmm_aliccp_full', ['test_download.py', 'test_ranges.py']),
        ('recommendation/esmm_aliccp_attention', ['test_queue.py']),
    ]
    for directory, tests in groups:
        subprocess.run([sys.executable, '-m', 'unittest', '-v', *tests],
                       cwd=ROOT/directory, check=True)
    print('LOCAL_CHECKS_COMPLETE: GPU/model tests are not included.', flush=True)


if __name__ == '__main__':
    main()
