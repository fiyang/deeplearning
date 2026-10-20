"""Leakage and checkpoint-contract checks using hand-computed examples."""
import unittest

import numpy as np
import torch

from features import build_history, pair_history
from feature_model import make_model, restore
from score_features import score_pairs


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.array([[1, 1, 5, 10], [1, 2, 1, 10], [1, 3, 5, 20], [2, 1, 4, 20], [1, 2, 4, 30]])
        self.users, self.items = self.rows[:, 0], self.rows[:, 1]
        self.genres = np.zeros((4, 18), dtype=np.float32)
        self.genres[1, 0] = self.genres[2, 1] = self.genres[3, 2] = 1
        self.history, self.numeric, self.state = self.build(self.rows)

    def build(self, rows):
        return build_history(rows, rows[:, 0], rows[:, 1], self.genres, 2, 3, 3)

    def test_hand_computed_strict_past(self):
        self.assertTrue(np.all(self.history[:2] == 0))
        np.testing.assert_allclose(self.numeric[:2, :4], [[0, .5, 0, .5], [0, .5, 0, .5]])
        self.assertEqual(self.history[2].tolist(), [1, 0, 0])
        self.assertEqual(self.history[4].tolist(), [3, 1, 0])
        np.testing.assert_allclose(self.numeric[3, 3], 2 / 3)
        np.testing.assert_allclose(self.numeric[4, [1, 3]], [.6, 1 / 3])
        np.testing.assert_allclose(self.numeric[4, 4:7], [.5, 0, .5])

    def test_current_future_and_tied_labels_cannot_leak(self):
        changed = self.rows.copy()
        changed[changed[:, 3] >= 20, 2] = 1
        histories, numeric, _ = self.build(changed)
        np.testing.assert_array_equal(self.history[:4], histories[:4])
        np.testing.assert_array_equal(self.numeric[:4], numeric[:4])
        # Tied events do not affect each other's features, regardless of row order.
        perm = np.array([4, 2, 1, 3, 0])
        histories, numeric, _ = self.build(self.rows[perm])
        np.testing.assert_array_equal(self.history[perm], histories)
        np.testing.assert_array_equal(self.numeric[perm], numeric)

    def test_held_out_state_is_frozen(self):
        before = {name: value.copy() for name, value in self.state.items()}
        h1, n1 = pair_history(self.state, np.array([1, 2]), np.array([2, 1]))
        h2, n2 = pair_history(self.state, np.array([1, 2]), np.array([2, 1]))
        np.testing.assert_array_equal(h1, h2)
        np.testing.assert_array_equal(n1, n2)
        for name in before:
            np.testing.assert_array_equal(before[name], self.state[name])

    def test_gradients_and_self_contained_inference(self):
        torch.manual_seed(42)
        categorical = np.array([[0, 0, 0], [1, 2, 3], [2, 3, 4]])
        item_metadata = np.column_stack([self.genres, np.zeros(4)])
        config = dict(user_count=2, item_count=3, embedding_dim=64, hidden_dim=64, mode="history")
        model = make_model(config, categorical, item_metadata)
        tensors = (torch.tensor(self.users), torch.tensor(self.items), torch.tensor(self.history), torch.tensor(self.numeric))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(*tensors), torch.tensor((self.rows[:, 2] >= 4).astype(np.float32)))
        loss.backward()
        for name in ["user_embedding", "item_embedding", "gender_embedding", "age_embedding", "occupation_embedding"]:
            self.assertGreater(getattr(model, name).weight.grad[1:].abs().sum(), 0)
        self.assertEqual(model.input_dim, 253)
        checkpoint = dict(model_state=model.state_dict(), config=config,
                          user_ids=torch.tensor([1, 2]), item_ids=torch.tensor([1, 2, 3]),
                          history_state={k: torch.from_numpy(v) for k, v in self.state.items()})
        reloaded = restore(checkpoint)
        with torch.inference_mode():
            torch.testing.assert_close(model(*tensors), reloaded(*tensors))
            h, n = pair_history(self.state, np.array([1, 1]), np.array([2, 3]))
            expected = model(torch.tensor([1, 1]), torch.tensor([2, 3]), torch.tensor(h), torch.tensor(n)).sigmoid().numpy()
        np.testing.assert_allclose(score_pairs(checkpoint, 1, [2, 3]), expected)
        with self.assertRaises(ValueError):
            score_pairs(checkpoint, 99, [1])


if __name__ == "__main__":
    unittest.main()
