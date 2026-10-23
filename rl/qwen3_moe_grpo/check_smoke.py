"""Allow the full job only after a completed GPU smoke and a remaining-time estimate."""
import argparse
import json
from pathlib import Path
import time
from common import load_config, smoke_gate, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary', type=Path)
    parser.add_argument('--deadline', required=True, type=float)
    args = parser.parse_args()
    result = smoke_gate(json.loads(args.summary.read_text()), load_config(), args.deadline - time.time())
    write_json(args.summary.parent / 'gate.json', result)
    print('SMOKE_GATE ' + json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
