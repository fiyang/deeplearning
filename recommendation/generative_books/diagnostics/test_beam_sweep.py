import unittest
import numpy as np
from beam_sweep import paired_hits, prefix_hits


class MetricsTests(unittest.TestCase):
    def test_prefix_match_is_not_item_match_and_invalid_slots_never_hit(self):
        codes = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 0]])
        pred = np.array([[1, -1], [-1, -1], [2, -1]])
        result = prefix_hits(pred, np.array([0, 0, 2]), codes)
        self.assertAlmostEqual(result['prefix_3_hit@10'], 2/3)
        self.assertAlmostEqual(result['prefix_4_hit@10'], 1/3)

    def test_paired_gains_and_losses_do_not_hide_regressions(self):
        result = paired_hits(np.array([[1], [8], [3], [9]]),
                             np.array([[7], [2], [3], [4]]), np.array([1, 2, 3, 4]))
        self.assertEqual(result['gained'], 2)
        self.assertEqual(result['lost'], 1)
        self.assertEqual(result['net_hits'], 1)
        self.assertEqual(result['recall_delta_pp'], 25)

    def test_only_top_ten_count(self):
        pred = np.arange(11)[None]
        result = paired_hits(pred, pred, np.array([10]))
        self.assertEqual(result['gained'], 0)
        self.assertEqual(result['net_hits'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
