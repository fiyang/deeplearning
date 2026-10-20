import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import canonical, split_standard, FeatureBuilder, PROFILE, ITEM
from metrics import binary_metrics, funnel_metrics
from model import FunnelModel, product_bce_logits
from verify import roc_area


def config(kind="esmm"):
    return dict(kind=kind, vocabulary_size=20, embedding_dim=3, category_count=2,
                hidden=[8, 4], dropout=0., ctr_prior=.4, cvr_prior=.1)


class Tests(unittest.TestCase):
    def test_product_loss_and_gradient(self):
        a = torch.tensor([-.4, 1., 2.], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([.3, -2., .7], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0., 1., 0.], dtype=torch.float64)
        loss = product_bce_logits(a, b, y)
        reference = F.binary_cross_entropy(a.sigmoid()*b.sigmoid(), y)
        torch.testing.assert_close(loss, reference)
        for x, z in zip(torch.autograd.grad(loss, (a, b), retain_graph=True), torch.autograd.grad(reference, (a, b))):
            torch.testing.assert_close(x, z)

    def test_extreme_logits(self):
        a = torch.tensor([-1000., 1000., 1000., -1000.], requires_grad=True)
        b = torch.tensor([1000., -1000., 1000., -1000.], requires_grad=True)
        loss = product_bce_logits(a, b, torch.tensor([1., 1., 0., 0.])); loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())

    def test_joint_labels_do_not_drop_unclicked_likes(self):
        y = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
        s = np.tile([.5, .2, .1], (4, 1))
        m = funnel_metrics(y, s)
        self.assertEqual(m["ctcvr_click_and_like"]["count"], 4)
        self.assertEqual(m["ctcvr_click_and_like"]["positives"], 1)
        self.assertEqual(m["cvr_like_given_click"]["count"], 2)

    def test_ties_auc_and_ap(self):
        y = np.array([0, 1, 1, 0, 1]); p = np.array([.4, .4, .9, .1, .9])
        brute = np.mean([float(x > z) + .5*float(x == z) for x in p[y == 1] for z in p[y == 0]])
        m = binary_metrics(y, p)
        self.assertAlmostEqual(m["auc"], brute)
        self.assertAlmostEqual(m["average_precision"], 2/3 + 1/3 * .75)
        self.assertEqual(binary_metrics(y, np.ones(5)*.5)["auc"], .5)
        self.assertIsNone(binary_metrics(np.zeros(3), np.zeros(3))["auc"])

    def test_independent_roc_uses_double_precision(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=10001).astype(np.float32)
        p = np.round(rng.random(10001), 2).astype(np.float32)
        self.assertAlmostEqual(roc_area(y, p), binary_metrics(y, p)["auc"], places=12)

    def test_shared_gradient_and_reload(self):
        torch.manual_seed(3); model = FunnelModel(config())
        x = torch.tensor([[1, 2], [3, 4], [1, 4]])
        tags = torch.tensor([[5, 0], [6, 7], [5, 0]])
        y = torch.tensor([[1., 1.], [0., 0.], [1., 0.]])
        model.loss(model(x, tags), y).backward()
        for name in ["encoder", "ctr", "cvr"]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()))
        self.assertEqual(model.encoder.embedding.weight.grad[0].abs().sum(), 0)
        probability = model.probabilities(x, tags)
        torch.testing.assert_close(probability[:, 2], probability[:, 0]*probability[:, 1])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"best.pt"; torch.save(model.state_dict(), p)
            restored = FunnelModel(config()); restored.load_state_dict(torch.load(p, weights_only=True))
            torch.testing.assert_close(restored.probabilities(x, tags), probability)

    def test_independent_branches(self):
        model = FunnelModel(config("independent"))
        x = torch.tensor([[1, 2], [3, 4]]); tags = torch.tensor([[5], [6]])
        ctr, cvr = model(x, tags)
        F.binary_cross_entropy_with_logits(ctr, torch.tensor([1., 0.])).backward()
        self.assertIsNone(model.cvr_encoder.embedding.weight.grad)
        self.assertIsNone(model.cvr[-1].weight.grad)

    def test_temporal_split(self):
        rows = np.zeros((5, 8), dtype=np.int64)
        # Deliberately inconsistent date fields: only milliseconds select splits.
        rows[:, 2] = [20220418, 20220419, 20220408, 20220421, 20220422]
        rows[:, 4] = [1650297600000, 1650297599999, 1650200000000, 1650556799999, 1650556800000]
        train, val, test = split_standard(rows)
        self.assertEqual(set(train), {1, 2}); self.assertEqual(set(val), {0, 3}); self.assertEqual(set(test), {4})

    def test_train_only_vocabulary_multitag(self):
        b = FeatureBuilder.__new__(FeatureBuilder)
        b.users = {0: {k: "1" for k in PROFILE}, 1: {k: "999" for k in PROFILE}}
        b.items = {0: {**{k: "1" for k in ITEM}, "upload_dt": "2022-04-01", "tag": "3,4"},
                   1: {**{k: "999" for k in ITEM}, "upload_dt": "2022-04-01", "tag": "8"}}
        b.maps = {}; b.vocabulary_size = 1
        train = np.array([[0, 0, 20220408, 1030, 1, 1, 1, 1]])
        test = np.array([[1, 1, 20220422, 1030, 2, 0, 1, 2]])
        b.fit(train); x, tags = b.transform(train); unknown, unseen_tags = b.transform(test)
        self.assertEqual(np.count_nonzero(tags), 2)
        self.assertEqual(unknown[0, 0], 0); self.assertEqual(unknown[0, 1], 0)
        self.assertEqual(unknown[0, 5], 0); self.assertEqual(np.count_nonzero(unseen_tags), 0)
        self.assertEqual(canonical("-124"), "<MISSING>")


if __name__ == "__main__":
    unittest.main()
