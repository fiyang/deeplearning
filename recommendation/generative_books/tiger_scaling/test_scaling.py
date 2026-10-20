"""Check scaled TIGER generation, padding, gradients, and checkpoint dimensions."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from models import Tiger


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.codes = np.array([[0,0,0,0], [0,0,0,5], [0,1,2,0], [1,0,0,0],
                               [1,1,1,0], [2,0,1,0], [2,1,0,0], [3,2,1,0]])

    def model(self):
        return Tiger(self.codes, k=4, collision_size=8).to(self.device)

    def test_constrained_generation_matches_full_scores(self):
        model = self.model().eval()
        h = torch.tensor([[1,2,0]], device=self.device)
        u = torch.tensor([0], device=self.device)
        seen = torch.zeros(1, len(self.codes), dtype=torch.bool, device=self.device)
        seen[0,0] = True
        with torch.no_grad():
            predictions, scores = model.recommend(h,u,seen,beam_size=16)
            src = model.source(h.repeat(8,1),u.repeat(8))
            targets = model.item_tokens[torch.arange(8,device=self.device)+1]
            logits = model.net(input_ids=src,attention_mask=src.ne(0),labels=targets).logits.float()
            joint = logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1).sum(-1)
            joint[0] = -torch.inf
            expected_scores, expected_items = joint.topk(7)
            torch.testing.assert_close(predictions[0,:7],expected_items)
            torch.testing.assert_close(scores[0,:7],expected_scores,rtol=3e-5,atol=8e-5)
            self.assertTrue(bool(predictions[0,7:].eq(-1).all()))

    def test_padding_gradients_and_checkpoint(self):
        model = self.model().eval()
        h = torch.tensor([[1,2]],device=self.device)
        padded = torch.tensor([[1,2,0,0]],device=self.device)
        u, target = torch.tensor([0],device=self.device),torch.tensor([3],device=self.device)
        loss = model(h,u,target)
        torch.testing.assert_close(loss,model(padded,u,target),rtol=3e-5,atol=3e-5)
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        self.assertGreater(float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)),0.)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'weights.pt'
            torch.save(model.state_dict(),path)
            restored = self.model().eval()
            restored.load_state_dict(torch.load(path,map_location=self.device,weights_only=True))
            torch.testing.assert_close(model(h,u,target),restored(h,u,target),rtol=0,atol=0)

    def test_only_width_and_ffn_configuration_change(self):
        old = Tiger(self.codes,k=4,collision_size=8,d_model=128,d_ff=1024)
        new = Tiger(self.codes,k=4,collision_size=8)
        a,b=old.net.config.to_dict(),new.net.config.to_dict()
        changed={key for key in a if a[key] != b[key]}
        self.assertEqual(changed,{'d_model','d_ff'})
        self.assertEqual(b['d_model'],256)
        self.assertEqual(b['d_ff'],2048)
        self.assertEqual(b['num_layers'],4)
        self.assertEqual(b['num_decoder_layers'],4)


if __name__=='__main__':
    unittest.main(verbosity=2)
