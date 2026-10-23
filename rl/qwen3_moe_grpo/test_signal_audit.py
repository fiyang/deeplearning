import json
import math
from pathlib import Path
import tempfile
import unittest

from audit_reward_groups import audit, candidates
from probe_training_signal import choose_rows, group_metrics, summarize


def row_for(ks, step=1):
    return dict(step=step, reward=sum(ks) / 16,
                frac_reward_zero_std=sum(k in (0, 4) for k in ks) / 4,
                reward_std=sum(math.sqrt(k * (4 - k) / 12) for k in ks) / 4,
                **{'rewards/correctness_reward/std': sum(math.sqrt(k * (8 - k) / 56)
                    for k in [sum(ks[:2]), sum(ks[2:])]) / 2})


class SignalAuditTests(unittest.TestCase):
    def test_recover_all_correct_wrong_and_mixed(self):
        for counts in [(4, 4, 4, 4), (0, 0, 0, 0), (0, 1, 2, 4), (1, 3, 3, 4)]:
            expected = tuple(counts.count(i) for i in range(5))
            self.assertIn(expected, candidates(row_for(counts)))
        self.assertEqual(candidates(row_for((4, 4, 4, 3))), [(0, 0, 0, 1, 3)])

    def test_inconsistent_metrics_not_silently_accepted(self):
        row = row_for((4, 4, 4, 4))
        row['reward_std'] = .1
        self.assertEqual(candidates(row), [])

    def test_duplicate_log_steps_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'metrics.jsonl').write_text('\n'.join(json.dumps(row_for((4, 4, 4, 4))) for _ in range(2)))
            (root / 'config.json').write_text(json.dumps(dict(train_samples=8, epochs=1, num_generations=4,
                generation_batch_size=8, per_device_train_batch_size=1, gradient_accumulation_steps=16)))
            with self.assertRaisesRegex(ValueError, 'exactly once'):
                audit(root / 'metrics.jsonl', root / 'config.json')

    def test_reward_ties_do_not_imply_text_duplicates(self):
        texts = [f'Explanation {i}. <answer>2</answer>' for i in range(4)]
        m = group_metrics(texts, '2')
        self.assertEqual(m['category'], 'all_correct')
        self.assertEqual(m['unique_normalized_texts'], 4)
        self.assertTrue(m['all_same_valid_answer'])
        self.assertEqual(m['normalized_pair_duplicate_rate'], 0)

    def test_duplicate_and_invalid_answer_definitions(self):
        m = group_metrics(['A <answer>2</answer>', 'A  <answer>2</answer>',
                           'B <answer>3</answer>', 'invalid'], '2')
        self.assertEqual(m['category'], 'mixed')
        self.assertEqual(m['correct_count'], 2)
        self.assertEqual(m['unique_exact_texts'], 4)
        self.assertEqual(m['unique_normalized_texts'], 3)
        self.assertEqual(m['valid_answer_count'], 3)
        self.assertAlmostEqual(m['normalized_pair_duplicate_rate'], 1 / 6)
        self.assertFalse(m['all_same_valid_answer'])
        m['truncated'] = [False] * 4
        s = summarize([m])
        self.assertEqual(s['normalized_any_duplicate_rate'], 1)
        self.assertEqual(s['invalid_answer_rate'], .25)

    def test_new_training_pool_excludes_validation_test_and_question_duplicates(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[-1]['content']
            def __call__(self, text, **kwargs):
                return {'input_ids': list(text)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(name, rows):
                (root / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            source = [dict(question=q, answer='explanation #### 2') for q in
                      ['old', 'validation', 'test', 'old', 'new1', 'new2']]
            write('gsm8k_train.jsonl', source)
            write('gsm8k_test.jsonl', [source[2]])
            write('train_selected.jsonl', [dict(id='train:0', prompt='old', solution='2')])
            write('validation_selected.jsonl', [dict(id='train:1', prompt='validation', solution='2')])
            rows, pools = choose_rows(Tokenizer(), dict(system_prompt='', max_prompt_length=100), root, root, 1, 42)
            self.assertEqual(pools['unseen_eligible_pool'], 2)
            self.assertEqual(rows[0]['id'], 'train:0')
            self.assertIn(rows[1]['id'], ['train:4', 'train:5'])


if __name__ == '__main__':
    unittest.main()
