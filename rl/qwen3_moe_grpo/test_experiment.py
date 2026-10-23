"""Offline checks for data isolation, reward reuse and the paid-run gate."""
import json
from pathlib import Path
import tempfile
import unittest

from common import (correctness_reward, expected_steps, load_config, select_rows,
                    smoke_gate, wilson)


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return 'USER:' + messages[-1]['content']

    def __call__(self, text, **kwargs):
        return {'input_ids': list(range(2000 if 'overlength' in text else 20))}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.good = dict(completed=True, optimizer_steps=16, adapter_updated=True,
                         expert_adapter_updated=True,
                         trainable_expert_parameters=100, gpu_total_gib=79,
                         peak_reserved_gib=72, training_seconds=160,
                         evaluation_seconds=20)

    def test_one_epoch_is_128_updates(self):
        self.assertEqual(expected_steps(self.config), 128)

    def test_complete_smoke_with_budget_passes(self):
        self.assertTrue(smoke_gate(self.good, self.config, 10000)['passed'])

    def test_no_update_blocks_paid_full_run(self):
        self.good['adapter_updated'] = False
        self.assertFalse(smoke_gate(self.good, self.config, 10000)['passed'])

    def test_missing_expert_adapter_blocks(self):
        self.good['trainable_expert_parameters'] = 0
        self.assertFalse(smoke_gate(self.good, self.config, 10000)['passed'])

    def test_no_expert_update_blocks(self):
        self.good['expert_adapter_updated'] = False
        self.assertFalse(smoke_gate(self.good, self.config, 10000)['passed'])

    def test_incomplete_smoke_blocks(self):
        self.good['completed'] = False
        self.assertFalse(smoke_gate(self.good, self.config, 10000)['passed'])

    def test_insufficient_memory_blocks(self):
        self.good['peak_reserved_gib'] = 78
        self.assertFalse(smoke_gate(self.good, self.config, 10000)['passed'])

    def test_invalid_memory_measurement_blocks(self):
        self.good['peak_reserved_gib'] = float('nan')
        self.assertFalse(smoke_gate(self.good, self.config, 10000)['passed'])

    def test_time_budget_blocks(self):
        self.assertFalse(smoke_gate(self.good, self.config, 100)['passed'])

    def test_invalid_time_blocks(self):
        self.good['training_seconds'] = float('nan')
        self.assertFalse(smoke_gate(self.good, self.config, 10000)['passed'])

    def test_quality_does_not_control_full_run_gate(self):
        self.good.update(baseline={'accuracy': 1}, final={'accuracy': 0})
        self.assertTrue(smoke_gate(self.good, self.config, 10000)['passed'])

    def test_rewards_for_plain_generated_strings(self):
        self.assertEqual(correctness_reward(['<answer>1/2</answer>', '<answer>1</answer>'],
                                            ['0.5', '2']), [1.0, 0.0])

    def test_wilson_interval_includes_observed_accuracy(self):
        lo, hi = wilson(115, 128)
        self.assertLess(lo, 115 / 128)
        self.assertGreater(hi, 115 / 128)

    def test_disjoint_reproducible_non_thinking_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            for split in ['train', 'test']:
                rows = [{'question': f'{split} question {i}', 'answer': 'solution #### 42'}
                        for i in range(20)]
                rows += [{'question': 'shared question', 'answer': '#### 42'},
                         {'question': 'overlength', 'answer': '#### 42'}]
                (data / f'gsm8k_{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            config = dict(self.config, train_samples=8, test_samples=8)
            tokenizer = Tokenizer()
            runs = []
            for name in ['a', 'b']:
                folder = root / name
                folder.mkdir()
                runs.append(select_rows(tokenizer, config, data, folder))
            self.assertEqual(runs[0], runs[1])
            train, test = runs[0]
            self.assertEqual(len(train), 8)
            self.assertTrue(set(r['prompt'] for r in train).isdisjoint(r['prompt'] for r in test))
            self.assertTrue(all('overlength' not in r['prompt'] for r in train + test))
            self.assertTrue(all(c['enable_thinking'] is False for c in tokenizer.calls))


if __name__ == '__main__':
    unittest.main()
