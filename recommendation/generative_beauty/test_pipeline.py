"""Protocol and beam-search tests with synthetic data, never test-set tuning."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F

from models import Tiger, RQVAE
from train import metrics, seen_mask


class PipelineTests(unittest.TestCase):
    def test_metrics_exact_item_and_rank(self):
        pred = np.array([[4, 5, 2, 7, 8], [0, 1, 3, 4, 5], [-1, -1, -1, -1, -1]])
        result = metrics(pred, np.array([2, 0, 9]))
        self.assertAlmostEqual(result["recall@5"], 2 / 3)
        self.assertAlmostEqual(result["ndcg@5"], (1 + .5) / 3)
        self.assertEqual(result["duplicate_rows"], 0)

    def test_seen_masks_do_not_include_target_or_future(self):
        seq = [{"items": [1, 3, 2, 5, 4]}]
        val = seen_mask(seq, [0], "val", 5, torch.device("cpu"))[0]
        test = seen_mask(seq, [0], "test", 5, torch.device("cpu"))[0]
        self.assertEqual(val.tolist(), [True, True, True, False, False])
        self.assertEqual(test.tolist(), [True, True, True, False, True])

    def test_beam_equals_exhaustive_and_handles_sid_collisions(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        # First two items deliberately share the first three codes.
        codes = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0]])
        model = Tiger(codes, width=32, layers=1, dropout=0, k=2, user_buckets=3).eval()
        history, user = torch.tensor([[1, 0], [3, 0]]), torch.tensor([0, 1])
        seen = torch.tensor([[True, False, False, False], [False, False, True, False]])
        predicted, score = model.recommend(history, user, seen, beam_size=8)
        with torch.no_grad():
            src = model.source(history, user)
            truth = []
            for i in range(len(codes)):
                labels = model.item_tokens[i + 1][None].expand(2, -1)
                decoder = torch.cat([torch.ones(2, 1, dtype=torch.long), labels[:, :-1]], -1)
                logits = model.net(input_ids=src, attention_mask=src.ne(0), decoder_input_ids=decoder).logits
                truth.append(logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1).sum(-1))
            exhaustive = torch.stack(truth, -1).masked_fill(seen, -torch.inf)
            values, indices = exhaustive.topk(3)
        torch.testing.assert_close(predicted[:, :3], indices)
        torch.testing.assert_close(score[:, :3], values, atol=1e-5, rtol=1e-5)
        self.assertTrue(bool((predicted[:, 3:] == -1).all()))
        self.assertFalse(bool(seen.gather(1, predicted.clamp_min(0))[predicted >= 0].any()))

    def test_padding_invariance_and_trainable_parameters(self):
        torch.manual_seed(42)
        codes = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]])
        model = Tiger(codes, width=32, layers=1, dropout=0, k=2, user_buckets=3).eval()
        a = model(torch.tensor([[1]]), torch.tensor([0]), torch.tensor([1]))
        b = model(torch.tensor([[1, 0, 0]]), torch.tensor([0]), torch.tensor([1]))
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        a.backward()
        self.assertGreater(float(model.net.shared.weight.grad.abs().sum()), 0)
        model = RQVAE(input_dim=8, k=4, levels=3, latent=4)
        loss, _, _ = model(F.normalize(torch.randn(16, 8), dim=-1))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
