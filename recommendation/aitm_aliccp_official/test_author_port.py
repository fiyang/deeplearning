import ast
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from model import AITM, TF1Adam
from prepare import AUTHOR_FIELDS, extract, fit_vocab, encode
from raw_data import FIELDS


def csr(rows):
    tokens, offsets = [], [0]
    for fields in rows:
        for index, field in enumerate(FIELDS):
            tokens.extend(int(v) * len(FIELDS) + index for v in fields.get(field, []))
            offsets.append(len(tokens))
    return dict(tokens=np.array(tokens, dtype=np.uint32), offsets=np.array(offsets, dtype=np.uint32))


class AuthorPortTests(unittest.TestCase):
    def config(self):
        return dict(vocabulary_sizes=[4] * 18, embedding_dim=5, constraint_weight=.6, l2=1e-6)

    def test_official_snapshot_and_feature_order(self):
        from prepare import sha256
        root = Path(__file__).parent / 'official'
        manifest = json.loads((root / 'manifest.json').read_text())
        for f in manifest['files']:
            self.assertEqual(sha256(root / f['file']), f['sha256'])
        source = ast.parse((root / 'AITM.py').read_text())
        feature_order = next(ast.literal_eval(node.value) for node in source.body if isinstance(node, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'all_columns' for t in node.targets))
        self.assertEqual(feature_order, AUTHOR_FIELDS)

    def test_last_value_common_override_and_dropped_history(self):
        raw = SimpleNamespace(n=3, refs=np.array([0, 1, 0]), info={'common': {'rows': 2}},
                              parts={'common': csr([{'101': [9, 10], '109_14': [99], '508': [30]}, {}]),
                                     'sample': csr([{'101': [5], '205': [20, 21], '508': [31]},
                                                    {'101': [8], '205': [22]}, {'205': [23]}])})
        with tempfile.TemporaryDirectory() as d:
            actual = extract(raw, Path(d) / 'dense.npy')
            self.assertEqual(actual.shape, (3, 18))
            np.testing.assert_array_equal(actual[:, 0], [10, 8, 10])
            np.testing.assert_array_equal(actual[:, 9], [21, 22, 23])
            np.testing.assert_array_equal(actual[:, 13], [30, 0, 30])
            self.assertNotIn(99, actual)

    def test_frequency_scope_threshold_and_unseen(self):
        dense = np.zeros((36, 18), dtype=np.uint32)
        dense[:12, 0] = 100
        dense[12:23, 0] = 200
        dense[23:, 0] = 999  # held-out only; may never enter vocabulary
        vocabulary = fit_vocab(dense, np.arange(23))
        np.testing.assert_array_equal(vocabulary[0], [100])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'encoded.npy'
            encode(dense, vocabulary, path)
            result = np.load(path)
            np.testing.assert_array_equal(result[:12, 0], 1)
            np.testing.assert_array_equal(result[12:, 0], 0)

    def test_forward_against_numpy_author_equations_and_checkpoint(self):
        torch.manual_seed(2020)
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            model = AITM(self.config()).to(device).eval()
            ids = np.arange(54).reshape(3, 18) % 4
            def array(t):
                return t.detach().cpu().numpy().astype(np.float64)
            def dense(x, layer):
                return x @ array(layer.weight).T + array(layer.bias)
            def tower(x, layers):
                for layer in layers:
                    if isinstance(layer, torch.nn.Linear): x = dense(x, layer)
                    elif isinstance(layer, torch.nn.ReLU): x = np.maximum(x, 0)
                return x
            x = np.concatenate([array(table.weight)[ids[:, i]] for i, table in enumerate(model.embeddings)], axis=1)
            click = tower(x, model.click_tower)
            purchase = tower(x, model.purchase_tower)
            info = tower(click, model.transfer)
            pair = np.stack([purchase, info], axis=1)
            q, k, v = [pair @ array(w) for w in model.attention]
            weights = np.exp(np.sum(q * k, axis=-1) / math.sqrt(32))
            weights /= weights.sum(axis=1, keepdims=True)
            attended = np.sum(weights[..., None] * v, axis=1)
            expected = np.concatenate([dense(click, model.click_head), dense(attended, model.purchase_head)], axis=1)
            input_ids = torch.tensor(ids, device=device)
            np.testing.assert_allclose(array(model(input_ids)), expected, atol=1e-8, rtol=1e-4)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'model.pt'
                torch.save(dict(config=model.config, model_state=model.state_dict()), p)
                saved = torch.load(p, map_location=device, weights_only=True)
                restored = AITM(saved['config']).to(device).eval()
                restored.load_state_dict(saved['model_state'])
                torch.testing.assert_close(restored(input_ids), model(input_ids))

    def test_loss_regularization_oov_and_all_branch_gradients(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            model = AITM(self.config()).to(device)
            logits = torch.tensor([[-2., 2.], [1., -1.]], device=device, requires_grad=True)
            labels = torch.tensor([[0., 0.], [1., 1.]], device=device)
            parts = model.loss_parts(logits, labels)
            p = 1 / (1 + np.exp(-logits.detach().cpu().numpy().astype(np.float64)))
            y = labels.cpu().numpy()
            expected_bce = -(y*np.log(p+1e-7)+(1-y)*np.log(1-p+1e-7)).mean(0).sum()
            expected_constraint = .6*np.maximum(p[:, 1]-p[:, 0], 0).mean()
            self.assertAlmostEqual(parts['bce'].item(), expected_bce, places=5)
            self.assertAlmostEqual(parts['constraint'].item(), expected_constraint, places=6)
            parts['regularizer'].backward()
            self.assertIsNone(model.click_head.weight.grad)
            for table in model.embeddings:
                torch.testing.assert_close(table.weight.grad, table.weight * 1e-6)
            model.zero_grad(set_to_none=True)
            ids = torch.zeros((16, 18), device=device, dtype=torch.long)
            labels = torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,0.]] * 4, device=device)
            loss = model.loss(model(ids), labels)
            loss.backward()
            for name in ['embeddings','click_tower','purchase_tower','transfer','attention','click_head','purchase_head']:
                gradients = [p.grad for p in getattr(model, name).parameters()]
                self.assertTrue(any(g is not None and bool(g.abs().sum() > 0) for g in gradients), name)
                self.assertTrue(all(g is None or bool(torch.isfinite(g).all()) for g in gradients))
            self.assertTrue(bool(model.embeddings[0].weight.grad[0].abs().sum() > 0))
            TF1Adam(model.parameters()).step()
            model.eval()
            torch.testing.assert_close(model(ids), model(ids))

    def test_tf1_adam_against_independent_numpy_updates(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            p = torch.nn.Parameter(torch.tensor([.2, -.1], device=device, dtype=torch.float64))
            optimizer = TF1Adam([p])
            expected = np.array([.2, -.1]); first = np.zeros(2); second = np.zeros(2)
            for step, grad in enumerate([[1e-10, -.3], [.5, -.1], [-.3, .2]], 1):
                g = np.array(grad)
                first = .9*first + .1*g; second = .999*second + .001*g*g
                expected -= .001*np.sqrt(1-.999**step)/(1-.9**step)*first/(np.sqrt(second)+1e-8)
                p.grad = torch.tensor(grad, device=device, dtype=torch.float64)
                optimizer.step()
                np.testing.assert_allclose(p.detach().cpu().numpy(), expected, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
