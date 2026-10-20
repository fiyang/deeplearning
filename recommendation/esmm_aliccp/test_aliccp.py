import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import parse_line,parse_file,RawTable,fit_vocabulary,remap_tokens,DeviceTable,FIELDS,FIELD_INDEX,stratified_split
from metrics import binary_metrics,auc_interval
from model import Model,product_bce_logits


def cfg(kind='esmm'):
    return dict(kind=kind,vocabulary_size=20,embedding_dim=3,field_count=len(FIELDS),hidden=[8,4],dropout=0.,ctr_prior=.1,cvr_prior=.01)


class Tests(unittest.TestCase):
    def test_first_feature_and_all_multivalues_retained(self):
        sid,label,bags=parse_line('1,1,1,210:1,210:2,101:12,205:6\n')
        self.assertEqual(label,(1,1));self.assertEqual(len(bags[FIELD_INDEX['210']]),2)
        self.assertEqual(bags[FIELD_INDEX['210']][0]//len(FIELDS),1)
        with self.assertRaises(ValueError):parse_line('1,0,1,101:12')

    def test_large_bags_are_not_truncated(self):
        _,_,bags=parse_line('1,0,0,'+','.join(f'109_14:{i}' for i in range(1000)))
        self.assertEqual(len(bags[FIELD_INDEX['109_14']]),1000)

    def test_vocabulary_and_ragged_batch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=p/'raw.csv';raw.write_text('1,0,0,101:1,205:2,210:3,210:4\n2,1,1,101:8,205:9,210:3\n')
            parse_file(raw,p/'cache');table=RawTable(p/'cache');vocab=fit_vocabulary(table,np.array([0]))
            mapping=remap_tokens(table.tokens,vocab)
            self.assertEqual(np.count_nonzero(mapping==0),2)
            device_table=DeviceTable(table,vocab,'cpu');ids,offsets=device_table.batch([1,0])
            emb=torch.nn.EmbeddingBag(len(vocab)+1,2,mode='sum',include_last_offset=True,padding_idx=0)
            with torch.no_grad():emb.weight.fill_(1);emb.weight[0].zero_()
            out=emb(ids,offsets).reshape(2,len(FIELDS),2)
            self.assertEqual(out[1,FIELD_INDEX['210'],0],2)
            self.assertEqual(out[0,FIELD_INDEX['101'],0],0)
            self.assertEqual(out[0,FIELD_INDEX['210'],0],1)

    def test_disjoint_stratified_splits(self):
        y=np.array([[0,0]]*80+[[1,0]]*15+[[1,1]]*5)
        a,b=stratified_split(y)
        self.assertEqual(len(a),80);self.assertEqual(len(b),20)
        self.assertFalse(set(a)&set(b));self.assertEqual(int(y[b,1].sum()),1)

    def test_product_loss(self):
        a=torch.tensor([-.2,1.,2.],dtype=torch.float64,requires_grad=True)
        b=torch.tensor([.7,-2.,1.],dtype=torch.float64,requires_grad=True);y=torch.tensor([0.,1.,0.],dtype=torch.float64)
        loss=product_bce_logits(a,b,y);ref=F.binary_cross_entropy(a.sigmoid()*b.sigmoid(),y)
        torch.testing.assert_close(loss,ref)
        for x,z in zip(torch.autograd.grad(loss,(a,b),retain_graph=True),torch.autograd.grad(ref,(a,b))):torch.testing.assert_close(x,z)
        a=torch.tensor([1000.,-1000.],requires_grad=True);b=-a.detach().clone().requires_grad_(True)
        loss=product_bce_logits(a,b,torch.tensor([1.,0.]));loss.backward();self.assertTrue(torch.isfinite(a.grad).all())

    def test_gradients_and_probabilities(self):
        m=Model(cfg());ids=torch.tensor([1,2,3]);offsets=torch.tensor([0,2]+[2]*(len(FIELDS)-1)+[3]+[3]*(len(FIELDS)-1))
        self.assertEqual(len(offsets),2*len(FIELDS)+1)
        pred=m.probabilities(ids,offsets)
        torch.testing.assert_close(pred[:,2],pred[:,0]*pred[:,1])
        m.loss(m(ids,offsets),torch.tensor([[1.,1.],[0.,0.]])).backward()
        for module in [m.embedding,m.ctr,m.cvr]:self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))

    def test_metric_ties(self):
        y=np.array([0,1,1,0]);s=np.array([.1,.1,.8,.9])
        brute=np.mean([float(a>b)+.5*float(a==b) for a in s[y==1] for b in s[y==0]])
        self.assertAlmostEqual(binary_metrics(y,s)['auc'],brute)
        ci=auc_interval(y,s);self.assertLessEqual(ci['lower'],brute);self.assertGreaterEqual(ci['upper'],brute)


if __name__=='__main__':unittest.main()
