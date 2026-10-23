"""Verify the complete screening from persisted texts without GPU or training."""
import argparse
import hashlib
import json
from pathlib import Path

from probe_training_signal import group_metrics
from round2 import read_rows
from screen_training_signal import classify, screen_summary


def verify(run):
    config = json.loads((run / 'config.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    selected_path = run / 'selected_train_questions.jsonl'
    if hashlib.sha256(selected_path.read_bytes()).hexdigest() != manifest['selected_sha256']:
        raise ValueError('Selected question hash differs from manifest')
    selected, groups = read_rows(selected_path), read_rows(run / 'rollout_groups.jsonl')
    if len(selected) != config['samples'] or [r['id'] for r in selected] != [g['id'] for g in groups]:
        raise ValueError('Incomplete or misordered screening groups')
    for row, group in zip(selected, groups):
        if row['solution'] != group['solution']:
            raise ValueError('Reference answer changed')
        if len(group['truncated']) != 4 or any(type(v) is not bool for v in group['truncated']):
            raise ValueError('Invalid truncation flags')
        recomputed = dict(group_metrics(group['completions'], row['solution']),
                          **classify(group, config['known_review_ids']))
        if any(group[key] != value for key, value in recomputed.items()):
            raise ValueError('Persisted group metrics differ from the saved texts')
    summary = json.loads((run / 'summary.json').read_text())
    result = screen_summary(groups, config['samples'], summary['seconds'], config['known_review_ids'])
    if not result['completed'] or any(summary[key] != value for key, value in result.items()):
        raise ValueError('Final aggregate differs from recomputed groups')
    candidates = read_rows(run / 'numeric_mixed_candidates.jsonl')
    expected = [dict(g, prompt=r['prompt']) for r, g in zip(selected, groups) if g['valid_numeric_mixed']]
    if candidates != expected:
        raise ValueError('Saved candidate list differs from screened groups')
    return dict(verified=True, **result,
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [selected_path, run / 'rollout_groups.jsonl', run / 'numeric_mixed_candidates.jsonl', run / 'summary.json']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run / 'verified_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('SCREEN_VERIFIED ' + json.dumps({key: result[key] for key in
          ['verified', 'screened_questions', 'valid_numeric_mixed_count', 'format_only_mixed_count', 'clean_numeric_mixed_count']}))
