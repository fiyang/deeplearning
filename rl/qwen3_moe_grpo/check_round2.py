"""Reject an expanded paid run without measured update, memory and time headroom."""
import argparse
import json
import math
from pathlib import Path
import time

from common import expected_steps, load_config, smoke_gate, write_json


def round2_gate(smoke, previous, full_eval, config, available_seconds):
    technical = smoke_gate(smoke, config, float('inf'))
    reasons = [r for r in technical['reasons'] if 'time budget' not in r]
    if len(smoke.get('validation_history', [])) != config['epochs'] or not smoke.get('adapter_restore_verified'):
        reasons.append('smoke did not verify epoch validation and checkpoint restoration')
    old_steps = previous.get('optimizer_steps', 0)
    old_seconds = previous.get('training_seconds', 0)
    smoke_steps = smoke.get('optimizer_steps', 0)
    smoke_seconds = smoke.get('training_seconds', 0)
    values = [old_steps, old_seconds, smoke_steps, smoke_seconds]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
        reasons.append('invalid measured training duration')
        estimate = None
    else:
        train_seconds = max(old_seconds / old_steps, smoke_seconds / smoke_steps) * expected_steps(config)
        val_fraction = config['validation_samples'] / full_eval['full']['n']
        eval_seconds = (full_eval['baseline']['seconds'] * val_fraction +
                        full_eval['final']['seconds'] * (config['epochs'] * val_fraction + 1))
        estimate = (train_seconds + eval_seconds) * 1.15 + 300
        if not math.isfinite(estimate) or estimate > available_seconds:
            reasons.append('training plus validation and full test exceed remaining time budget')
    return dict(passed=not reasons, reasons=reasons, estimated_full_seconds=estimate,
                available_seconds=available_seconds, quality_used_for_gate=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('smoke', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config_round2.json'))
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    previous = json.loads((Path(config['comparison_source_run']) / 'summary.json').read_text())
    full_eval = json.loads((Path(config['comparison_full_eval_run']) / 'summary.json').read_text())
    if not previous['completed'] or not full_eval['completed'] or full_eval['full']['n'] != 1319:
        raise ValueError('Verified completed reference runs are required')
    result = round2_gate(json.loads(args.smoke.read_text()), previous, full_eval,
                         config, args.deadline - time.time())
    write_json(args.smoke.parent / 'gate.json', result)
    print('ROUND2_GATE ' + json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
