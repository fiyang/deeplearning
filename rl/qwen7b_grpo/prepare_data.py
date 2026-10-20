"""Fetch the original GSM8K JSONL files on the remote training machine only."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent / 'data'
SOURCE = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data'
FILES = {
    'train': (7473, '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'),
    'test': (1319, '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'),
}


def validate(path, count, expected_hash):
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_hash:
        raise ValueError(f'GSM8K checksum mismatch: {path.name}')
    rows = [json.loads(line) for line in body.splitlines()]
    if len(rows) != count or not all(isinstance(r.get('question'), str) and
                                   isinstance(r.get('answer'), str) and
                                   '####' in r['answer'] for r in rows):
        raise ValueError(f'Unexpected GSM8K schema or count: {path.name}')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, (count, checksum) in FILES.items():
        target = ROOT / f'gsm8k_{split}.jsonl'
        url = f'{SOURCE}/{split}.jsonl'
        if not target.exists():
            temporary = target.with_suffix('.jsonl.partial')
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    body = response.read(10 * 1024 * 1024 + 1)
                if len(body) > 10 * 1024 * 1024:
                    raise ValueError('Unexpectedly large GSM8K source file')
                temporary.write_bytes(body)
                validate(temporary, count, checksum)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        validate(target, count, checksum)
        manifest.append(dict(split=split, rows=count, source=url, sha256=checksum))
        print(f'DATA_READY {split} rows={count}', flush=True)
    (ROOT / 'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
