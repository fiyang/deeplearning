"""Fetch the original GSM8K bytes through GitHub's API on the training host."""
import base64
import json
from pathlib import Path
import sys
import urllib.request

from common import ROOT, write_json
from rl.qwen7b_grpo.prepare_data import FILES, SOURCE, validate


def get_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'deeplearning-lab'})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(10 * 1024 * 1024 + 1)
    if len(body) > 10 * 1024 * 1024:
        raise ValueError('Unexpectedly large GitHub API response')
    return json.loads(body)


def main():
    if sys.platform != 'linux' or not ROOT.is_relative_to('/root/autodl-tmp'):
        raise RuntimeError('Download data on the AutoDL training host only')
    data = ROOT.parent / 'qwen7b_grpo' / 'data'
    data.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, (count, checksum) in FILES.items():
        path = data / f'gsm8k_{split}.jsonl'
        api = ('https://api.github.com/repos/openai/grade-school-math/contents/'
               f'grade_school_math/data/{split}.jsonl')
        if not path.exists():
            metadata = get_json(api)
            if metadata.get('encoding') != 'base64':
                api = metadata['git_url']
                if not api.startswith('https://api.github.com/repos/openai/grade-school-math/git/blobs/'):
                    raise ValueError('Unexpected GitHub blob URL')
                metadata = get_json(api)
            if metadata.get('encoding') != 'base64':
                raise ValueError('Expected a base64 GitHub blob')
            temporary = path.with_suffix('.jsonl.partial')
            try:
                temporary.write_bytes(base64.b64decode(metadata['content']))
                validate(temporary, count, checksum)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        validate(path, count, checksum)
        manifest.append(dict(split=split, rows=count, sha256=checksum,
                             source=f'{SOURCE}/{split}.jsonl', api_source=api))
        print(f'DATA_READY {split} rows={count}', flush=True)
    write_json(data / 'download_manifest.json', manifest)


if __name__ == '__main__':
    main()
