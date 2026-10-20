import unittest
from rewards import score, extract_answer, number_value


class RewardContract(unittest.TestCase):
    def test_equivalent_numeric_forms(self):
        for output, target in [('1,200', '1200'), ('-2.5', '-5/2'), ('0.50', '1/2')]:
            self.assertEqual(score(f'Work shown.\n<answer>{output}</answer>', target), 1.0)

    def test_wrong_and_reward_hacking(self):
        for text in ['The answer is 42.', '<answer>41</answer>',
                     '<answer>1</answer><answer>42</answer>',
                     '<answer>42 or 43</answer>', '<answer>42</answer> or 43',
                     '<answer>4,2</answer>', '<answer>NaN</answer>',
                     '<answer>1/0</answer>', '<answer>6*7</answer>']:
            self.assertEqual(score(text, '42'), 0.0, text)

    def test_unfinished_and_missing(self):
        self.assertIsNone(extract_answer('<answer>42'))
        self.assertIsNone(number_value('nan'))


if __name__ == '__main__':
    unittest.main()
