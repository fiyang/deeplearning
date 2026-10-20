"""Verify split isolation and cached constrained generation against exhaustive scores."""
import tempfile
import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from models import DecoderOnly, RQVAE, Tiger
from prepare import prefix_examples, unique_semantic_ids
import prepare
from train import Dataset, metrics


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.codes = np.array([[0,0,0,0], [0,0,0,5], [0,1,2,0], [1,0,0,0],
                               [1,1,1,0], [2,0,1,0], [2,1,0,0], [3,2,1,0]])

    def test_metric_hand_calculation(self):
        pred = np.full((3,20), -1)
        pred[0,0], pred[1,2] = 7, 8
        m = metrics(pred, np.array([7,8,9]))
        self.assertAlmostEqual(m["recall@10"], 2/3)
        self.assertAlmostEqual(m["ndcg@10"], .5)

    def test_collision_group_over_original_512_slots(self):
        codes = np.zeros((569,3),dtype=np.int64)
        ids,count = unique_semantic_ids(codes,1024)
        self.assertEqual(ids[-1,3],568)
        self.assertEqual(len(np.unique(ids,axis=0)),569)
        self.assertEqual(max(count.values()),569)
        with self.assertRaises(RuntimeError):
            unique_semantic_ids(codes,512)

    def test_split_history_and_seen_isolation(self):
        seq = [[1,2,3,4,5], [6,7,8,9,10,11]]
        off, pos, owner, val, test = prefix_examples(seq)
        np.testing.assert_array_equal(pos, [1,2,6,7,8])
        np.testing.assert_array_equal(val, [3,9])
        np.testing.assert_array_equal(test, [4,10])
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            for k,v in dict(sequence_items=np.concatenate(seq), sequence_offsets=off,
                            train_position=pos, train_user=owner, val_position=val,
                            test_position=test, user_hash=[0,1], popularity=np.ones(11)).items():
                np.save(p / f"{k}.npy", v)
            data = Dataset(p, self.device, history=2)
            h,_,target = data.batch("val", np.array([0,1]))
            np.testing.assert_array_equal(h.cpu(), [[2,3],[8,9]])
            np.testing.assert_array_equal(target.cpu(), [3,9])
            seen = data.seen("val", [0,1]).cpu().numpy()
            self.assertTrue(seen[0,0])  # Outside model's truncated history, still excluded.
            self.assertFalse(seen[0,3])  # Validation target not leaked.
            self.assertFalse(seen[0,4])  # Test item not leaked.

    def test_raw_data_pipeline_sorts_deduplicates_and_holds_out(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/"raw").mkdir()
            rows = [{"reviewerID":"user-a","asin":f"book-{i}","unixReviewTime":t}
                    for i,t in enumerate([30,10,50,20,40])]
            rows.append({"reviewerID":"user-a","asin":"book-0","unixReviewTime":90})
            rows.extend({"reviewerID":"short-user","asin":f"short-{i}","unixReviewTime":i} for i in range(4))
            with gzip.open(p/"raw/reviews_Books_5.json.gz","wt") as f:
                for row in rows:
                    f.write(json.dumps(row)+"\n")
            with gzip.open(p/"raw/meta_Books.json.gz","wt") as f:
                for asin in sorted({r["asin"] for r in rows}):
                    f.write(repr({"asin":asin,"title":"Title "+asin,"categories":[["Books"]]})+"\n")
            with patch.object(prepare,"DATA",p):
                prepare.prepare_data()
            info = json.loads((p/"dataset_info.json").read_text())
            self.assertEqual(info["dropped_duplicate_user_item"],1)
            self.assertEqual(info["dropped_users_under_5"],1)
            self.assertEqual(info["train_examples"],2)
            self.assertEqual(info["train_interactions"],3)
            names = [i["asin"] for i in json.loads((p/"items.json").read_text())]
            flat = np.load(p/"sequence_items.npy")
            self.assertEqual([names[i-1] for i in flat], ["book-1","book-3","book-0","book-4","book-2"])
            self.assertEqual(names[flat[np.load(p/"val_position.npy")[0]]-1],"book-4")
            self.assertEqual(names[flat[np.load(p/"test_position.npy")[0]]-1],"book-2")
            self.assertEqual(info["missing_metadata"],0)

    def exact_scores(self, model, history, user):
        n = len(self.codes)
        h, u = history.repeat(n,1), user.repeat(n)
        labels = model.item_tokens[torch.arange(n, device=self.device)+1]
        if isinstance(model, Tiger):
            src = model.source(h,u)
            logits = model.net(input_ids=src, attention_mask=src.ne(0), labels=labels).logits.float()
        else:
            inputs = torch.cat([model.prefix(h,u), labels[:,:-1]], -1)
            mask = inputs.ne(0)
            position = (mask.long().cumsum(-1)-1).clamp_min(0)
            hidden = model.net(input_ids=inputs, attention_mask=mask, position_ids=position,
                               use_cache=False).last_hidden_state[:,-4:]
            logits = F.linear(hidden, model.net.wte.weight).float()
        return logits.log_softmax(-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1).sum(-1)

    def test_sparse_beam_and_kv_cache_match_exhaustive(self):
        for cls in (Tiger, DecoderOnly):
            with self.subTest(model=cls.__name__):
                model = cls(self.codes, k=4, collision_size=8).to(self.device).eval()
                h = torch.tensor([[1,2,0], [3,0,0]], device=self.device)
                u = torch.tensor([0,1], device=self.device)
                seen = torch.zeros(2,len(self.codes),dtype=torch.bool,device=self.device)
                seen[0,0], seen[1,2] = True, True
                with torch.no_grad():
                    predicted, scores = model.recommend(h,u,seen,beam_size=16)
                    for i in range(2):
                        exact = self.exact_scores(model,h[i:i+1],u[i:i+1]).masked_fill(seen[i],-torch.inf)
                        expected_scores, expected_items = exact.topk(7)
                        torch.testing.assert_close(predicted[i,:7], expected_items)
                        torch.testing.assert_close(scores[i,:7], expected_scores, rtol=2e-5, atol=5e-5)
                        self.assertTrue(bool(predicted[i,7:].eq(-1).all()))
                del model

    def test_padding_invariance_and_gradients(self):
        for cls in (Tiger, DecoderOnly):
            with self.subTest(model=cls.__name__):
                model = cls(self.codes,k=4,collision_size=8).to(self.device).eval()
                h = torch.tensor([[1,2]],device=self.device)
                hp = torch.tensor([[1,2,0,0]],device=self.device)
                u, y = torch.tensor([0],device=self.device), torch.tensor([3],device=self.device)
                loss, padded = model(h,u,y), model(hp,u,y)
                torch.testing.assert_close(loss,padded,rtol=2e-5,atol=2e-5)
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
                del model
        rq = RQVAE(input_dim=16,k=4,latent=4).to(self.device)
        loss,rec,_ = rq(torch.randn(12,16,device=self.device))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(rq.encoder[0].weight.grad)
        self.assertTrue(all(c.grad is not None for c in rq.codebooks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
