"""Recheck the saved sampling probe; no GPU, model update or reward change."""
import argparse
import hashlib
import json
from pathlib import Path

from probe_training_signal import group_metrics, summarize


def analyze(run):
    archive_path = run / 'result_archive.json'
    archive = json.loads(archive_path.read_text())
    summary = archive['summary.json']
    rows = archive['rollout_groups']
    assert summary['completed'] and summary['optimizer_updates'] == 0
    assert summary['groups'] == len(rows) == 256
    assert summary['completions'] == sum(len(r['completions']) for r in rows) == 1024
    assert len({(r['setting'], r['cohort'], r['id']) for r in rows}) == 256
    # Recompute every reward and duplicate measurement from the actual texts.
    for row in rows:
        recomputed = group_metrics(row['completions'], row['solution'])
        assert all(row[k] == v for k, v in recomputed.items()), row['id']
    table = []
    cases = []
    ids = {}
    for setting in ('current', 'broader'):
        for cohort in ('seen_train', 'unseen_train'):
            groups = [r for r in rows if r['setting'] == setting and r['cohort'] == cohort]
            assert len(groups) == 64
            stats = summarize(groups)
            assert stats == summary['results'][setting][cohort]
            ids[setting, cohort] = {r['id'] for r in groups}
            valid_numeric_mixed = []
            format_only_mixed = []
            for group in groups:
                valid_rewards = {reward for answer, reward in zip(group['parsed_answers'], group['rewards'])
                                 if answer is not None}
                if valid_rewards == {0, 1}:
                    valid_numeric_mixed.append(group['id'])
                elif group['category'] == 'mixed':
                    format_only_mixed.append(group['id'])
                if group['category'] == 'mixed' or group['valid_answer_count'] < 4:
                    cases.append({k: group[k] for k in ('id', 'cohort', 'setting', 'category',
                        'correct_count', 'parsed_answers', 'valid_answer_count', 'truncated')})
            table.append(dict(setting=setting, cohort=cohort, **stats,
                any_duplicate_groups=sum(r['unique_normalized_texts'] < 4 for r in groups),
                all_four_same_groups=sum(r['unique_normalized_texts'] == 1 for r in groups),
                valid_numeric_mixed_ids=valid_numeric_mixed,
                valid_numeric_mixed_rate=len(valid_numeric_mixed) / len(groups),
                format_only_mixed_ids=format_only_mixed))
    for cohort in ('seen_train', 'unseen_train'):
        assert ids['current', cohort] == ids['broader', cohort]
    assert not ids['current', 'seen_train'] & ids['current', 'unseen_train']
    paired_changes = {}
    for cohort in ('seen_train', 'unseen_train'):
        a = {r['id']: r for r in rows if r['setting'] == 'current' and r['cohort'] == cohort}
        b = {r['id']: r for r in rows if r['setting'] == 'broader' and r['cohort'] == cohort}
        paired_changes[cohort] = dict(
            newly_mixed=[i for i in a if a[i]['category'] != 'mixed' and b[i]['category'] == 'mixed'],
            no_longer_mixed=[i for i in a if a[i]['category'] == 'mixed' and b[i]['category'] != 'mixed'])
    return dict(verified=True, archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                table=table, paired_changes=paired_changes, mixed_or_invalid_cases=cases,
                definitions={
                    'duplicate': 'Same full text after NFKC and whitespace normalization; not semantic similarity.',
                    'mixed': 'At least one strict reward 1 and one strict reward 0 in four completions.',
                    'valid_numeric_mixed': 'Both a correct and a wrong numeric answer among valid parsed answers; invalid answers ignored.',
                    'format_only_mixed': 'Strict rewards are mixed, but valid parsed numeric answers do not contain both correctness classes.'},
                limitations=['64 questions per cohort and one seed; mixed-group differences are small counts.',
                             'Fixed final adapter pilot, not historical training duplication or a new accuracy benchmark.',
                             'Numerically mixed groups still need question/label quality review.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    (args.run / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for row in result['table']:
        print(json.dumps({k: row[k] for k in ('setting', 'cohort', 'n', 'category_counts',
              'any_duplicate_groups', 'all_four_same_groups', 'valid_numeric_mixed_rate',
              'format_only_mixed_ids')}, ensure_ascii=False))
