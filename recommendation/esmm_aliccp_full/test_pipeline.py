import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from data import RawTable,DeviceTable,FIELDS,fit_vocabulary,remap,split_rows
from model import Model,product_bce_logits

HERE=Path(__file__).parent

def features(items):return '\1'.join(f'{field}\2{value}\3{weight}' for field,value,weight in items)

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build=tempfile.TemporaryDirectory();cls.binary=Path(cls.build.name)/'prepare'
        subprocess.run(['g++','-std=c++17','-O2',str(HERE/'prepare.cpp'),'-o',str(cls.binary)],check=True)
    @classmethod
    def tearDownClass(cls):cls.build.cleanup()
    def fixture(self,root,bad_count=False):
        common=root/'common.csv';sample=root/'sample.csv';cache=root/'cache'
        common.write_text('aaa,3,'+features([('101',1,1),('109_14',2,2),('109_14',3,.5)])+'\nbbb,1,'+features([('101',999,1)])+'\n')
        sample.write_text('x,0,0,aaa,'+('2' if bad_count else '1')+','+features([('205',4,1)])+'\ny,1,1,bbb,1,'+features([('205',888,3)])+'\n')
        result=subprocess.run([str(self.binary),str(common),str(sample),str(cache)],capture_output=True,text=True)
        return cache,result
    def test_parser_vocab_pool_and_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cache,result=self.fixture(root);self.assertEqual(result.returncode,0,result.stderr)
            raw=RawTable(cache);self.assertEqual(raw.info['label_pairs_00_01_10_11'],[1,0,0,1])
            vocab,lookup=fit_vocabulary(raw,np.array([0]),root)
            self.assertEqual(set(vocab.tolist()),{23,47,70,4*23+13})
            mapped=root/'mapped';stats=remap(raw,lookup,mapped)
            self.assertEqual(stats['common']['unknown_tokens'],1);self.assertEqual(stats['sample']['unknown_tokens'],1)
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                table=DeviceTable(raw,mapped,device)
                config=dict(kind='esmm',field_count=23,embedding_dim=2,vocabulary_size=len(vocab)+1,hidden=[8],dropout=0,ctr_prior=.5,cvr_prior=.5)
                model=Model(config).to(device)
                with torch.no_grad():model.embedding.weight.copy_(torch.arange(10,device=device).reshape(5,2)/10);model.embedding.weight[0].zero_()
                common,sample=table.batch([0,1])
                actual=(model.embedding(common[0],common[1],per_sample_weights=common[2])+model.embedding(sample[0],sample[1],per_sample_weights=sample[2])).reshape(2,23,2)
                expected=torch.zeros_like(actual);expected[0,0]=model.embedding.weight[lookup[23]];expected[0,1]=2*model.embedding.weight[lookup[47]]+.5*model.embedding.weight[lookup[70]];expected[0,13]=model.embedding.weight[lookup[105]]
                torch.testing.assert_close(actual,expected)
                logits=model(common,sample);loss=model.loss(logits,table.labels);loss.backward()
                for name in ['embedding','ctr','cvr']:self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters()))
                probs=model.probabilities(common,sample);torch.testing.assert_close(probs[:,2],probs[:,0]*probs[:,1]);self.assertTrue(torch.isfinite(loss))
    def test_parser_rejects_wrong_feature_count(self):
        with tempfile.TemporaryDirectory() as folder:
            _,r=self.fixture(Path(folder),True);self.assertNotEqual(r.returncode,0);self.assertIn('feature count',r.stderr)
    def test_product_loss_matches_reference_and_extremes(self):
        a=torch.tensor([-2.,.4,3.],requires_grad=True);b=torch.tensor([1.,-.5,2.],requires_grad=True);y=torch.tensor([0.,1.,1.])
        expected=torch.nn.functional.binary_cross_entropy(a.sigmoid()*b.sigmoid(),y)
        torch.testing.assert_close(product_bce_logits(a,b,y),expected)
        loss=product_bce_logits(torch.tensor([1000.,-1000.]),torch.tensor([1000.,1000.]),torch.tensor([0.,1.]));self.assertTrue(torch.isfinite(loss))
    def test_split_preserves_strata_and_disjoint_rows(self):
        y=np.array([[0,0]]*100+[[1,0]]*20+[[1,1]]*10,dtype=np.uint8)
        tr,va=split_rows(y);self.assertEqual(len(va),13);self.assertEqual(y[va].sum(0).tolist(),[3,1]);self.assertEqual(len(np.intersect1d(tr,va)),0)
        np.testing.assert_array_equal(split_rows(y)[1],va)
        tr2,va2=split_rows(np.concatenate([y,[[0,1]]]).astype(np.uint8))
        self.assertEqual(len(tr2)+len(va2),len(y));self.assertNotIn(len(y),tr2);self.assertNotIn(len(y),va2)

if __name__=='__main__':unittest.main()
