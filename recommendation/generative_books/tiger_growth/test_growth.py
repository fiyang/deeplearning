import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from growth import expand_tiger, check_equivalence, check_initial_validation

spec = importlib.util.spec_from_file_location('original_books_models', Path(__file__).resolve().parents[1] / 'models.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


class GrowthTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        codes = np.array([[0,0,0,0], [0,0,0,1], [0,1,2,0], [1,0,0,0],
                          [1,1,1,0], [2,0,1,0], [2,1,0,0], [3,2,1,0]])
        self.base = source.create_model('tiger', codes).eval()
        self.grown = expand_tiger(self.base).eval()
        self.h = torch.tensor([[1,2,0], [3,4,5]])
        self.u, self.y = torch.tensor([0,17]), torch.tensor([3,6])

    def test_logits_loss_padding_and_tied_output_preserved(self):
        check_equivalence(self.base, self.grown, self.h, self.u, self.y)
        padded = torch.cat([self.h, torch.zeros(2,2,dtype=torch.long)], -1)
        torch.testing.assert_close(self.grown(self.h,self.u,self.y),
                                   self.grown(padded,self.u,self.y), rtol=2e-5, atol=2e-5)
        self.assertIs(self.grown.net.shared.weight, self.grown.net.lm_head.weight)

    def test_beam_predictions_and_scores_preserved(self):
        seen = torch.zeros(2,8,dtype=torch.bool)
        seen[0,0] = True
        a, sa = self.base.recommend(self.h,self.u,seen,20)
        b, sb = self.grown.recommend(self.h,self.u,seen,20)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(sa,sb,rtol=2e-5,atol=2e-5)

    def test_size_gradients_symmetry_breaking_and_reload(self):
        self.assertEqual(sum(p.numel() for p in self.base.parameters()),5043712)
        self.assertEqual(sum(p.numel() for p in self.grown.parameters()),14281344)
        self.grown.train()
        optimizer=torch.optim.AdamW(self.grown.parameters(),lr=3e-5)
        optimizer.zero_grad()
        self.grown(self.h,self.u,self.y).backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.grown.parameters()))
        torch.nn.utils.clip_grad_norm_(self.grown.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        e=self.grown.net.shared.weight
        self.assertGreater(float((e[:,:128]-e[:,128:]).abs().max()),0.)
        self.grown.eval()
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'grown.pt'
            torch.save(self.grown.state_dict(),file)
            restored=expand_tiger(self.base).eval()
            restored.load_state_dict(torch.load(file,weights_only=True))
            torch.testing.assert_close(restored(self.h,self.u,self.y),self.grown(self.h,self.u,self.y),rtol=0,atol=0)

    def test_large_initial_drift_rejected(self):
        with self.assertRaises(RuntimeError):
            check_initial_validation({'recall@10':.04,'ndcg@10':.026},
                                     {'recall@10':.01,'ndcg@10':.01})


if __name__=='__main__':
    unittest.main(verbosity=2)
