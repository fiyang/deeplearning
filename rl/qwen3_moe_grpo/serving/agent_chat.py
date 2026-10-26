"""Terminal client for a private candidate Agent session; synthetic data only."""
import argparse
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post(base, key, path, body):
    req = Request(base.rstrip('/') + path, data=json.dumps(body, ensure_ascii=False).encode(),
                  headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=600) as response:
            return json.load(response)
    except HTTPError as error:
        return json.load(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8001')
    parser.add_argument('--model', choices=['qwen3-base', 'qwen3-grpo'], default='qwen3-grpo')
    parser.add_argument('--industry', choices=['homecare', 'general-worker'], default='homecare')
    parser.add_argument('--message', action='append', help='Repeat for an automated multi-turn smoke')
    args = parser.parse_args()
    key = os.environ.get('QWEN3_API_KEY', '')
    if not key:
        parser.error('Set QWEN3_API_KEY')
    created = post(args.base_url, key, '/sessions', dict(model=args.model, industry=args.industry))
    if 'session_id' not in created:
        raise SystemExit(json.dumps(created, ensure_ascii=False))
    ident = created['session_id']
    print('测试会话：' + ident + '（岗位和拉群均为模拟，输入 /quit 结束）', flush=True)
    iterator = iter(args.message) if args.message else None
    while True:
        try:
            text = next(iterator) if iterator is not None else input('你：')
        except (StopIteration, EOFError):
            break
        if text == '/quit':
            break
        if iterator is not None:
            print('你：' + text, flush=True)
        result = post(args.base_url, key, f'/sessions/{ident}/messages', dict(text=text))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not result.get('completed'):
            break


if __name__ == '__main__':
    main()
