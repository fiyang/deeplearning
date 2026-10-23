"""Offline guards against silent test selection, pairing and score errors."""
import unittest

from common import load_config
from eval_full import compare, full_test_rows
from test_experiment import Tokenizer


class FullEvaluationTests(unittest.TestCase):
    def test_all_questions_preserved_and_old_subset_first(self):
        config, tokenizer = load_config(), Tokenizer()
        source = [dict(question=f'question {i}', answer='#### 42') for i in range(1319)]
        old = [dict(id='test:17', prompt='USER:question 17', solution='42')]
        rows = full_test_rows(tokenizer, config, source, old)
        self.assertEqual(len(rows), 1319)
        self.assertEqual(rows[0], old[0])
        self.assertEqual(len({r['id'] for r in rows}), 1319)
        self.assertEqual(rows, full_test_rows(tokenizer, config, source, old))

    def test_overlength_must_fail_not_drop(self):
        with self.assertRaisesRegex(ValueError, 'cannot silently exclude'):
            full_test_rows(Tokenizer(), load_config(), [dict(question='overlength', answer='#### 1')], [])

    def test_original_prompt_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'prompt/answer mismatch'):
            full_test_rows(Tokenizer(), load_config(), [dict(question='q', answer='#### 1')],
                           [dict(id='test:0', prompt='CHANGED', solution='1')])

    def test_pairing_is_by_same_ordered_ids(self):
        a = [dict(id=str(i), correct=True, solution='1') for i in range(2)]
        with self.assertRaises(ValueError):
            compare(a, list(reversed(a)))

    def test_paired_counts_and_significance(self):
        a = [dict(id=str(i), correct=i >= 4, solution='1') for i in range(128)]
        b = [dict(id=str(i), correct=i not in (4, 5), solution='1') for i in range(128)]
        result = compare(a, b)
        self.assertEqual(result['wrong_to_right'], 4)
        self.assertEqual(result['right_to_wrong'], 2)
        self.assertEqual(result['accuracy_delta_pp'], 1.5625)
        self.assertAlmostEqual(result['mcnemar_exact_two_sided_p'], .6875)


if __name__ == '__main__':
    unittest.main()
