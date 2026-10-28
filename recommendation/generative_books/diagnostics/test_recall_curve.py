import unittest
from recall_curve import rank_report


class RankReportTests(unittest.TestCase):
    def test_exact_boundaries_and_nested_cutoffs(self):
        rows = [list(range(100)) for _ in range(7)]
        report = rank_report(rows, [0, 9, 10, 49, 50, 99, 100])
        self.assertEqual(report['hits'], {'10': 2, '50': 4, '100': 6})
        self.assertEqual(report['rank_buckets'],
                         {'1-10': 2, '11-50': 2, '51-100': 2, 'not_in_top100': 1})
        self.assertEqual(report['recovered_after_top10'], 4)

    def test_invalid_padding_never_hits(self):
        report = rank_report([[-1] * 100], [0])
        self.assertEqual(report['recall@100'], 0)
        self.assertEqual(report['top100_valid_fraction'], 0)

    def test_truncated_predictions_cannot_claim_recall100(self):
        with self.assertRaises(ValueError):
            rank_report([list(range(20))], [10])

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):
            rank_report([[0] * 100], [0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
