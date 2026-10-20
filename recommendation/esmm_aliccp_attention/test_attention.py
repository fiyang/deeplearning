"""Independent attention math, masking, pooling, loss and checkpoint checks."""
import io
import unittest

import numpy as np
import torch

from base_model import Model as Baseline
from data import FIELDS
from model import FieldAttention, Model, present_fields
from train import validate_comparison


def part(rows, device):
    tokens, weights, offsets = [], [], [0]
    for row in rows:
        for field in range(len(FIELDS)):
            for token, weight in row.get(field, []):
                tokens.append(token)
                weights.append(weight)
            offsets.append(len(tokens))
    return (torch.tensor(tokens, dtype=torch.int32, device=device),
            torch.tensor(offsets, dtype=torch.int64, device=device),
            torch.tensor(weights, dtype=torch.float32, device=device))


def config():
    return dict(kind='esmm', fields=FIELDS, field_count=23, embedding_dim=18, id_embedding_dim=64,
                vocabulary_size=7, base_vocabulary_size=3, id_vocabulary_size=5,
                hidden=[16, 8], dropout=0., ctr_prior=.3, cvr_prior=.2,
                attention_dim=64, attention_heads=4, attention_dropout=0., attention_layers=1)


class AttentionTests(unittest.TestCase):
    def test_attention_against_numpy(self):
        torch.manual_seed(12)
        layer = FieldAttention([3, 5, 2], width=8, heads=2, dropout=0).eval()
        x = torch.randn(1, 10)
        present = torch.tensor([[True, False, True]])
        actual = layer(x, present).detach().numpy()[0]
        array = lambda t: t.detach().numpy().astype(np.float64)
        def norm(v, module):
            return ((v-v.mean(-1, keepdims=True))/np.sqrt(v.var(-1, keepdims=True)+module.eps)
                    * array(module.weight)+array(module.bias))
        fields = np.split(array(x)[0], [3, 8])
        h = np.stack([v @ array(p.weight).T for v, p in zip(fields, layer.projections)])
        h[1] = 0
        normalized = norm(h, layer.pre_norm)
        q, k, v = np.split(normalized @ array(layer.attention.in_proj_weight).T
                           + array(layer.attention.in_proj_bias), 3, axis=-1)
        heads = []
        for head in range(2):
            sl = slice(head*4, (head+1)*4)
            scores = q[:, sl] @ k[:, sl].T / 2
            scores[:, 1] = -np.inf
            weights = np.exp(scores-np.max(scores, axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            heads.append(weights @ v[:, sl])
        attended = np.concatenate(heads, axis=-1) @ array(layer.attention.out_proj.weight).T
        attended += array(layer.attention.out_proj.bias)
        expected = norm(h+attended, layer.output_norm)
        expected[1] = 0
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)

    def test_masking_cross_field_dependency_and_batch_isolation(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(8)
            layer = FieldAttention([4, 4, 4], width=8, heads=2, dropout=0).to(device).eval()
            x = torch.randn(2, 12, device=device, requires_grad=True)
            mask = torch.tensor([[True, True, False], [False, False, False]], device=device)
            result = layer(x, mask)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result[1], torch.zeros_like(result[1]))
            changed = x.detach().clone()
            changed[0, 8:] += 100
            changed[1] -= 100
            torch.testing.assert_close(result, layer(changed, mask))
            torch.testing.assert_close(result[:1], layer(x[:1], mask[:1]), atol=2e-6, rtol=2e-5)
            result[0, 0, 0].backward()
            self.assertGreater(x.grad[0, 4:8].abs().sum().item(), 0)
            self.assertEqual(x.grad[0, 8:].abs().sum().item(), 0)
            self.assertEqual(x.grad[1].abs().sum().item(), 0)

    def test_pooling_loss_gradients_and_checkpoint(self):
        vocabulary = np.array([23, 24, 59, 71, 92, 128], dtype=np.uint32)
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(7)
            model = Model(config(), vocabulary).to(device)
            baseline = Baseline(config(), vocabulary).to(device)
            baseline.embedding.load_state_dict(model.embedding.state_dict())
            common = part([{0:[(1,.5),(5,2.)],1:[(2,3.)],2:[(4,.25)]},
                           {0:[(0,1.)]}, {1:[(2,0.)]}], device)
            sample = part([{13:[(3,2.),(6,.5)],1:[(2,1.)]}, {}, {}], device)
            torch.testing.assert_close(model.embedding(common,sample), baseline.embedding(common,sample))
            mask = present_fields(common, 23) | present_fields(sample, 23)
            self.assertEqual(mask[0].nonzero().flatten().tolist(), [0,1,2,13])
            self.assertFalse(mask[1:].any())
            self.assertEqual(model.encode(common, sample).shape, (3,23,64))
            self.assertEqual(model.input_dim, 1472)
            labels = torch.tensor([[1.,1.],[0.,0.],[1.,0.]], device=device)
            logits = model(common, sample)
            loss = model.loss(logits, labels)
            torch.testing.assert_close(loss, baseline.loss(logits, labels))
            loss.backward()
            for module in [model.embedding.ids, model.embedding.base, model.feature_encoder.projections,
                           model.feature_encoder.attention, model.ctr, model.cvr]:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()))
            for table in [model.embedding.ids, model.embedding.base]:
                torch.testing.assert_close(table.weight.grad[0], torch.zeros_like(table.weight.grad[0]))
            torch.optim.Adam(model.parameters(), lr=.0001).step()
            model.eval()
            before = model.probabilities(common, sample)
            self.assertTrue(torch.isfinite(before).all())
            torch.testing.assert_close(before[:,2], before[:,0]*before[:,1])
            stream = io.BytesIO()
            torch.save(dict(config=config(), state=model.state_dict()), stream)
            stream.seek(0)
            checkpoint = torch.load(stream, map_location=device, weights_only=True)
            restored = Model(checkpoint['config']).to(device).eval()
            restored.load_state_dict(checkpoint['state'], strict=True)
            torch.testing.assert_close(before, restored.probabilities(common, sample))

    def test_comparison_rejects_training_recipe_drift(self):
        reference = dict(epochs=10, patience=3, batch_size=4096, lr=.0001, embedding_dim=18,
                         id_embedding_dim=64, hidden=[360,200,80], dropout=.1, seed=42,
                         min_delta=.0001, validation_fraction=.1)
        self.assertEqual(validate_comparison(reference, reference)['variable'], 'feature_interaction')
        for field, value in [('lr',.0003), ('batch_size',2048), ('seed',43), ('id_embedding_dim',18)]:
            with self.assertRaisesRegex(ValueError, field):
                validate_comparison(dict(reference, **{field:value}), reference)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
