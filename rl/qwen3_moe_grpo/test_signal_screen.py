import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_signal_screen import verify
from probe_training_signal import group_metrics
from screen_training_signal import budget_deadline, classify, screen_summary, select_unseen


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]['content']
    def __call__(self, text, **kwargs):
        return {'input_ids': list(text)}


def group(ident, texts, truncated=None):
    row = dict(id=ident, completions=texts, solution='2', truncated=truncated or [False] * 4,
               **group_metrics(texts, '2'))
    return dict(row, **classify(row))


class SignalScreenTests(unittest.TestCase):
    def test_format_only_is_not_numeric_signal(self):
        row = group('train:1', ['<answer>2</answer>'] * 3 + ['<answer>2</answer>.'])
        self.assertTrue(row['format_only_mixed'])
        self.assertFalse(row['valid_numeric_mixed'])
        self.assertEqual(row['review_flags'], ['invalid_format'])

    def test_wrong_valid_number_is_signal_but_truncation_needs_review(self):
        texts = ['<answer>2</answer>'] * 3 + ['<answer>3</answer>']
        row = group('train:1', texts)
        self.assertTrue(row['clean_numeric_mixed'])
        flagged = group('train:1', texts, [False, False, False, True])
        self.assertTrue(flagged['valid_numeric_mixed'])
        self.assertFalse(flagged['clean_numeric_mixed'])
        self.assertFalse(classify(row, ['train:1'])['clean_numeric_mixed'])

    def test_summary_counts_and_partial_completion(self):
        rows = [group('train:1', ['<answer>2</answer>'] * 4),
                group('train:2', ['<answer>3</answer>'] * 4),
                group('train:3', ['<answer>2</answer>'] * 3 + ['<answer>3</answer>']),
                group('train:4', ['<answer>2</answer>'] * 3 + ['<answer>2</answer>.'])]
        summary = screen_summary(rows, 5, 3600)
        self.assertFalse(summary['completed'])
        self.assertEqual(summary['category_counts'], dict(all_correct=1, all_zero_reward=1, mixed=2))
        self.assertEqual(summary['valid_numeric_mixed_count'], 1)
        self.assertEqual(summary['format_only_mixed_count'], 1)
        self.assertEqual(summary['numeric_mixed_groups_per_hour'], 1)
        self.assertTrue(screen_summary(rows, 4, 3600)['completed'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            screen_summary(rows + rows[:1], 5, 3600)

    def test_unseen_selection_excludes_all_holdouts_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(name, rows):
                (root / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            source = [dict(question=q, answer='reason #### 2') for q in
                      ['old', 'val', 'test', 'old', 'val', 'new1', 'new2', 'new3']]
            save('gsm8k_train.jsonl', source)
            save('gsm8k_test.jsonl', [source[2]])
            save('train_selected.jsonl', [dict(id='train:0', prompt='old', solution='2')])
            save('validation_selected.jsonl', [dict(id='train:1', prompt='val', solution='2')])
            model = dict(system_prompt='', max_prompt_length=100)
            screen = dict(samples=3, expected_pool_size=3, seed=20261024)
            a, pools = select_unseen(Tokenizer(), model, root, root, screen)
            b, _ = select_unseen(Tokenizer(), model, root, root, screen)
            self.assertEqual(a, b)
            self.assertEqual({r['id'] for r in a}, {'train:5', 'train:6', 'train:7'})
            self.assertEqual(pools['unseen_eligible_pool'], 3)
            with self.assertRaisesRegex(ValueError, 'population'):
                select_unseen(Tokenizer(), model, root, root, dict(screen, expected_pool_size=4))

    def test_budget_counts_time_since_boot_and_respects_platform(self):
        config = dict(max_compute_budget_cny=40, max_price_cny_per_hour=5.59,
                      job_time_limit_seconds=23400, platform_time_limit_seconds=25200,
                      shutdown_reserve_seconds=600)
        started, now = 100000, 101000
        deadline = budget_deadline(config, 40, 5.59, now, started, started + 25200)
        self.assertLessEqual(deadline, started + 24600)
        self.assertLessEqual(deadline, started + 40 / 5.59 * 3600 - 600)
        self.assertEqual(budget_deadline(config, 40, 5.59, now, started, now + 3600), now + 3000)
        for budget, price in [(41, 5.59), (40, 6), (float('nan'), 5.59), (40, 0)]:
            with self.assertRaises(ValueError):
                budget_deadline(config, budget, price, now, started, started + 25200)
        with self.assertRaises(ValueError):
            budget_deadline(config, 40, 5.59, now, started, started + 25201)

    def test_saved_text_verification_rejects_corrupted_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = [dict(id='train:1', prompt='a question', solution='2')]
            row = group('train:1', ['<answer>2</answer>'] * 3 + ['<answer>3</answer>'])
            for name, rows in [('selected_train_questions.jsonl', selected), ('rollout_groups.jsonl', [row]),
                               ('numeric_mixed_candidates.jsonl', [dict(row, prompt='a question')])]:
                (root / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (root / 'config.json').write_text(json.dumps(dict(samples=1, known_review_ids=[])))
            (root / 'manifest.json').write_text(json.dumps(dict(selected_sha256=
                hashlib.sha256((root / 'selected_train_questions.jsonl').read_bytes()).hexdigest())))
            (root / 'summary.json').write_text(json.dumps(screen_summary([row], 1, 10)))
            self.assertTrue(verify(root)['verified'])
            row['correct_count'] = 4
            (root / 'rollout_groups.jsonl').write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'saved texts'):
                verify(root)


if __name__ == '__main__':
    unittest.main()
