"""Sequence leakage, tie invariance, masks, attention gradients, and checkpoint tests."""
import io
import unittest

import numpy as np
import torch

from auc_models import DeepFM
from features import build_history
from sequence_data import build_sequences, sequence_for_users
from sequence_model import make_sequence_model, restore_sequence


class SequenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.rows = np.array([[1, 1, 5, 10], [1, 2, 1, 10], [1, 3, 4, 20], [2, 1, 4, 20], [1, 2, 4, 30]])
        self.cats = np.array([[0, 0, 0], [1, 2, 3], [2, 3, 4]])
        self.meta = np.zeros((4, 19), dtype=np.float32)
        self.meta[1, [0, 4]] = 1
        self.meta[2, [1, 4]] = 1
        self.meta[3, [2, 4]] = 1
        self.meta[1:, -1] = [-1, 0, 1]
        self.sequences, self.sequence_state = self.build(self.rows)
        h, n, self.history_state = build_history(self.rows, self.rows[:, 0], self.rows[:, 1], self.meta[:, :18], 2, 3, 3)
        self.inputs = (torch.tensor(self.rows[:, 0]), torch.tensor(self.rows[:, 1]), torch.tensor(h), torch.tensor(n))
        self.inputs += tuple(torch.tensor(v) for v in self.sequences)
        self.base = DeepFM(2, 3, self.cats, self.meta)

    def build(self, rows):
        return build_sequences(rows, rows[:, 0], rows[:, 1], 2, 3)

    def config(self, encoder):
        return dict(user_count=2, item_count=3, encoder=encoder, embedding_dim=64, hidden_dim=64,
                    sequence_length=3, heads=4, layers=2, ff_dim=128, dropout=.1)

    def model(self, encoder="transformer"):
        model = make_sequence_model(self.config(encoder), self.cats, self.meta)
        model.warm_start(dict(model_state=self.base.state_dict()))
        return model

    def test_all_ratings_strict_past_and_tied_positions(self):
        ids, ratings, gaps, positions = self.sequences
        self.assertTrue(np.all(ids[:2] == 0))
        np.testing.assert_array_equal(ids[2], [2, 1, 0])
        np.testing.assert_array_equal(ratings[2], [1, 5, 0])
        np.testing.assert_array_equal(positions[2], [1, 1, 0])
        np.testing.assert_array_equal(ids[4], [3, 2, 1])
        np.testing.assert_array_equal(positions[4], [1, 2, 2])
        self.assertTrue((gaps[ids > 0] > 0).all())
        changed = self.rows.copy()
        changed[changed[:, 3] >= 20, 2] = 1
        alternative, _ = self.build(changed)
        for a, b in zip(self.sequences, alternative):
            np.testing.assert_array_equal(a[:4], b[:4])
        perm = np.array([4, 2, 1, 3, 0])
        alternative, _ = self.build(self.rows[perm])
        for a, b in zip(self.sequences, alternative):
            np.testing.assert_array_equal(a[perm], b)
        with self.assertRaises(AssertionError):
            sequence_for_users(self.sequence_state, np.array([1]), 30)

    def test_empty_padding_ties_and_position_signal(self):
        model = self.model().eval()
        seq = self.inputs[4:]
        with torch.inference_mode():
            encoded = model.extract_interest(*seq)
            self.assertTrue(torch.isfinite(encoded).all())
            torch.testing.assert_close(encoded[:2], torch.zeros_like(encoded[:2]))
            padded = tuple(torch.nn.functional.pad(v, (0, 2)) for v in seq)
            torch.testing.assert_close(encoded, model.extract_interest(*padded), atol=2e-6, rtol=1e-5)
            # Swap two events in the same timestamp group, preserving their own features.
            swapped = tuple(v.clone() for v in seq)
            for v in swapped:
                v[2, :2] = v[2, [1, 0]]
            torch.testing.assert_close(encoded[2], model.extract_interest(*swapped)[2], atol=2e-6, rtol=1e-5)
            changed_pos = seq[-1].clone()
            changed_pos[4] = torch.tensor([2, 1, 1], dtype=changed_pos.dtype)
            difference = (encoded[4] - model.extract_interest(*seq[:-1], changed_pos)[4]).abs().max()
            self.assertGreater(difference.item(), 1e-5)

    def test_warm_start_equivalence_and_attention_receives_gradients(self):
        model = self.model().eval()
        self.assertEqual(model.input_dim, 3072)
        with torch.inference_mode():
            torch.testing.assert_close(self.base(*self.inputs[:4]), model(*self.inputs), atol=2e-6, rtol=1e-5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        model.train()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(*self.inputs), torch.tensor([1., 0., 1., 1., 1.]))
            loss.backward()
            optimizer.step()
        for name in ["rating_embedding.weight", "time_embedding.weight", "position_embedding.weight",
                     "movie_feature_projection.weight", "encoder_layers.0.self_attn.in_proj_weight",
                     "encoder_layers.1.self_attn.in_proj_weight"]:
            gradient = dict(model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

    def test_checkpoint_and_cached_interest_match(self):
        for encoder in ["mean", "transformer"]:
            model = self.model(encoder).eval()
            # Make the appended interest columns active to exercise cached-feature scoring.
            with torch.no_grad():
                model.hidden.weight[:, -64:].normal_(std=.01)
            checkpoint = dict(config=self.config(encoder), model_state=model.state_dict())
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            restored = restore_sequence(torch.load(buffer, weights_only=True))
            with torch.inference_mode():
                interest = model.extract_interest(*self.inputs[4:])
                torch.testing.assert_close(model(*self.inputs), model.score_with_interest(*self.inputs[:4], interest))
                torch.testing.assert_close(model(*self.inputs), restored(*self.inputs))


if __name__ == "__main__":
    unittest.main()
