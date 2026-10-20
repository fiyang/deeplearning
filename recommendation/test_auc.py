"""Independent FM algebra, architecture, shared-gradient and reload checks."""
import io
import unittest

import numpy as np
import torch

from auc_models import fm_second_order, make_auc_model, restore_auc
from features import build_history, pair_history
from score_auc import score_pairs


class AUCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.categories = np.array([[0, 0, 0], [1, 2, 3], [2, 3, 4]])
        self.metadata = np.zeros((4, 19), dtype=np.float32)
        self.metadata[1, [0, 4]] = 1
        self.metadata[2, [1, 4]] = 1
        self.metadata[3, [2, 4]] = 1
        self.metadata[1:, -1] = [-1, 0, 1]
        self.rows = np.array([[1, 1, 5, 10], [1, 2, 1, 10], [1, 3, 5, 20], [2, 1, 4, 20], [1, 2, 4, 30]])
        h, n, self.state = build_history(self.rows, self.rows[:, 0], self.rows[:, 1], self.metadata[:, :18], 2, 3, 3)
        self.inputs = (torch.tensor(self.rows[:, 0]), torch.tensor(self.rows[:, 1]), torch.tensor(h), torch.tensor(n))

    def config(self, architecture):
        return dict(architecture=architecture, user_count=2, item_count=3, embedding_dim=64, hidden_dim=64, seed=42)

    def model(self, architecture):
        return make_auc_model(self.config(architecture), self.categories, self.metadata)

    def test_fm_matches_independent_pair_sum_and_gradient(self):
        simple = torch.tensor([[[1., 0.], [0., 2.], [3., 4.]]])
        self.assertEqual(fm_second_order(simple).item(), 11.)
        values = torch.randn(3, 7, 5, dtype=torch.float64, requires_grad=True)
        expected = sum((values[:, i] * values[:, j]).sum(dim=-1) for i in range(7) for j in range(i + 1, 7))
        actual = fm_second_order(values)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        g1, = torch.autograd.grad(actual.sum(), values, retain_graph=True)
        g2, = torch.autograd.grad(expected.sum(), values)
        torch.testing.assert_close(g1, g2, rtol=1e-12, atol=1e-12)

    def test_interaction_preserves_baseline_initial_prediction_and_learns(self):
        torch.manual_seed(42)
        baseline = self.model("mlp")
        torch.manual_seed(42)
        interaction = self.model("interaction")
        torch.testing.assert_close(baseline(*self.inputs), interaction(*self.inputs), atol=1e-7, rtol=1e-5)
        self.assertEqual(interaction.input_dim, 383)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(interaction(*self.inputs), torch.tensor([1., 0., 1., 1., 1.]))
        loss.backward()
        self.assertGreater(interaction.hidden.weight.grad[:, 253:].abs().sum().item(), 0)

    def test_deepfm_shares_factors_and_handles_empty_or_padded_history(self):
        model = self.model("deepfm")
        self.assertEqual(model.input_dim, 3008)
        linear, fm, deep = model.components(*self.inputs)
        torch.testing.assert_close(model(*self.inputs), linear + fm + deep)
        for component in [fm, deep]:
            gradient, = torch.autograd.grad(component.sum(), model.user_embedding.weight, retain_graph=True)
            self.assertGreater(gradient.abs().sum().item(), 0)
        padded = self.inputs[:2] + (torch.nn.functional.pad(self.inputs[2], (0, 5)), self.inputs[3])
        torch.testing.assert_close(model(*self.inputs), model(*padded), atol=1e-6, rtol=1e-5)
        no_history = self.inputs[:2] + (torch.zeros_like(self.inputs[2]), self.inputs[3])
        self.assertTrue(torch.isfinite(model(*no_history)).all())

    def test_self_contained_checkpoint_and_scoring(self):
        for architecture in ["mlp", "interaction", "deepfm"]:
            model = self.model(architecture)
            checkpoint = dict(config=self.config(architecture), model_state=model.state_dict(), selected_by="validation_auc",
                              user_ids=torch.tensor([101, 202]), item_ids=torch.tensor([10, 20, 30]),
                              history_state={k: torch.from_numpy(v) for k, v in self.state.items()})
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            reloaded = torch.load(buffer, weights_only=True)
            restored = restore_auc(reloaded)
            torch.testing.assert_close(model(*self.inputs), restored(*self.inputs))
            h, n = pair_history(self.state, np.array([1, 1]), np.array([2, 3]))
            with torch.inference_mode():
                expected = model(torch.tensor([1, 1]), torch.tensor([2, 3]), torch.tensor(h), torch.tensor(n)).sigmoid().numpy()
            np.testing.assert_allclose(score_pairs(reloaded, 101, [20, 30]), expected, atol=1e-6)
            with self.assertRaises(ValueError):
                score_pairs(reloaded, 999, [10])


if __name__ == "__main__":
    unittest.main()
