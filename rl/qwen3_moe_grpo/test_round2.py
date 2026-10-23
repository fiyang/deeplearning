"""CPU guards for split leakage, checkpoint selection and the longer paid job."""
import json
from pathlib import Path
import tempfile
import unittest

from check_round2 import round2_gate
from common import expected_steps, load_config
from round2 import check_reference_config, read_rows, select_training_validation, validation_is_better
from test_experiment import Tokenizer


class RoundTwoTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).with_name('config_round2.json'))

    def test_literal_unicode_separators_are_not_jsonl_record_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rows.jsonl'
            rows = [dict(prompt='a\u2028b\u2029c\u0085d'), dict(prompt='next')]
            p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
            self.assertEqual(read_rows(p), rows)

    def test_1024_by_2_is_512_steps_with_unchanged_algorithm(self):
        self.assertEqual(expected_steps(self.config), 512)
        check_reference_config(self.config, load_config())
        with self.assertRaisesRegex(ValueError, 'learning_rate'):
            check_reference_config(dict(self.config, learning_rate=1e-4), load_config())

    def test_validation_is_fixed_disjoint_and_never_from_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = [dict(question=f'train {i}', answer='#### 42') for i in range(40)]
            source += [dict(question='test only', answer='#### 42'),
                       dict(question='overlength', answer='#### 42'), source[0]]
            (root / 'gsm8k_train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in source))
            (root / 'gsm8k_test.jsonl').write_text(json.dumps(dict(question='test only', answer='#### 42'))+'\n')
            config = dict(self.config, train_samples=8, validation_samples=4, comparison_source_run=None)
            for name in ['a', 'b', 'c']:
                (root/name).mkdir()
            train, val = select_training_validation(Tokenizer(), config, root, root/'a')
            train2, val2 = select_training_validation(Tokenizer(), dict(config, train_samples=16), root, root/'b')
            self.assertEqual(train, train2[:8])
            self.assertEqual(val, val2)
            self.assertTrue({r['id'] for r in train2}.isdisjoint(r['id'] for r in val))
            self.assertTrue(all(r['id'].startswith('train:') for r in train2+val))
            self.assertNotIn('USER:test only', {r['prompt'] for r in train2+val})
            self.assertEqual((train, val), select_training_validation(Tokenizer(), config, root, root/'c'))
            with self.assertRaisesRegex(ValueError, 'preserve'):
                select_training_validation(Tokenizer(), dict(config, comparison_source_run=str(root/'b')),
                                           root, root/'c')

    def test_selection_keeps_earlier_tie_and_ignores_test_metrics(self):
        best = dict(correct=230, n=256, test_accuracy=0.1)
        self.assertFalse(validation_is_better(dict(correct=230, n=256, test_accuracy=1), best))
        self.assertFalse(validation_is_better(dict(correct=229, n=256, test_accuracy=1), best))
        self.assertTrue(validation_is_better(dict(correct=231, n=256, test_accuracy=0), best))
        with self.assertRaises(ValueError):
            validation_is_better(dict(correct=231, n=255), best)

    def test_gate_accounts_for_validation_and_full_test_time(self):
        smoke = dict(completed=True, optimizer_steps=16, adapter_updated=True,
                     expert_adapter_updated=True, trainable_expert_parameters=100,
                     gpu_total_gib=79, peak_reserved_gib=64, training_seconds=1600,
                     evaluation_seconds=50, validation_history=[{}, {}], adapter_restore_verified=True)
        previous = dict(optimizer_steps=128, training_seconds=12805.87)
        full = dict(full=dict(n=1319), baseline=dict(seconds=4013.60), final=dict(seconds=6542.28))
        result = round2_gate(smoke, previous, full, self.config, 72000)
        self.assertTrue(result['passed'], result)
        self.assertGreater(result['estimated_full_seconds'], 512 * 100 * 1.15 + 6542)
        self.assertFalse(round2_gate(smoke, previous, full, self.config, 55000)['passed'])
        smoke['adapter_restore_verified'] = False
        self.assertFalse(round2_gate(smoke, previous, full, self.config, 72000)['passed'])
        smoke['adapter_restore_verified'] = True
        smoke['training_seconds'] = 2400
        self.assertFalse(round2_gate(smoke, previous, full, self.config, 72000)['passed'])
        smoke['training_seconds'] = float('nan')
        self.assertFalse(round2_gate(smoke, previous, full, self.config, 72000)['passed'])

    def test_no_actual_update_cannot_pass_with_a_large_budget(self):
        smoke = dict(completed=True, optimizer_steps=16, adapter_updated=False,
                     expert_adapter_updated=False, trainable_expert_parameters=100,
                     gpu_total_gib=79, peak_reserved_gib=64, training_seconds=1600,
                     evaluation_seconds=50)
        self.assertFalse(round2_gate(smoke, dict(optimizer_steps=128, training_seconds=12800),
            dict(full=dict(n=1319), baseline=dict(seconds=4000), final=dict(seconds=6500)),
            self.config, 100000)['passed'])


if __name__ == '__main__':
    unittest.main()
