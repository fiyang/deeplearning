"""Reference checks for mixed-width pooling, OOVs, gradients and checkpoint reload."""
import io
import unittest
import numpy as np
import torch
from data import FIELDS
from model import Model


def part(rows,device):
    tokens=[];weights=[];offsets=[0]
    for row in rows:
        for f in range(23):
            for token,weight in row.get(f,[]):tokens.append(token);weights.append(weight)
            offsets.append(len(tokens))
    return (torch.tensor(tokens,dtype=torch.int32,device=device),torch.tensor(offsets,dtype=torch.int64,device=device),torch.tensor(weights,dtype=torch.float32,device=device))


class MixedEmbeddingTests(unittest.TestCase):
    def config(self):
        return dict(kind='esmm',fields=FIELDS,field_count=23,embedding_dim=18,id_embedding_dim=64,vocabulary_size=7,
                    base_vocabulary_size=3,id_vocabulary_size=5,hidden=[16,8],dropout=0,ctr_prior=.3,cvr_prior=.2)

    def test_weighted_reference_gradients_and_checkpoint_roundtrip(self):
        vocabulary=np.array([23,24,59,71,92,128],dtype=np.uint32)
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(7);model=Model(self.config(),vocabulary).to(device)
            self.assertEqual(model.input_dim,506)
            self.assertEqual(model.embedding.base.weight.shape,(3,18));self.assertEqual(model.embedding.ids.weight.shape,(5,64))
            self.assertEqual(model.embedding.base_lookup.tolist(),[0,0,1,0,2,0,0])
            self.assertEqual(model.embedding.id_lookup.tolist(),[0,1,0,2,0,3,4])
            with torch.no_grad():
                model.embedding.base.weight.copy_(torch.arange(54,device=device).reshape(3,18)/100)
                model.embedding.ids.weight.copy_(torch.arange(320,device=device).reshape(5,64)/100)
                model.embedding.base.weight[0].zero_();model.embedding.ids.weight[0].zero_()
            common=part([{0:[(1,.5),(5,2.)],1:[(2,3.)],2:[(4,.25)]},{0:[(0,1.)]}],device)
            sample=part([{13:[(3,2.),(6,.5)],1:[(2,1.)]},{}],device)
            actual=model.embedding(common,sample)
            dims=[64 if f in [0,13] else 18 for f in range(23)]
            offsets=np.r_[0,np.cumsum(dims)];expected=torch.zeros((2,506),device=device)
            expected[0,offsets[0]:offsets[1]]=.5*model.embedding.ids.weight[1]+2*model.embedding.ids.weight[3]
            expected[0,offsets[1]:offsets[2]]=4*model.embedding.base.weight[1]
            expected[0,offsets[2]:offsets[3]]=.25*model.embedding.base.weight[2]
            expected[0,offsets[13]:offsets[14]]=2*model.embedding.ids.weight[2]+.5*model.embedding.ids.weight[4]
            torch.testing.assert_close(actual,expected)
            # Including all-empty bags must retain batch shape and zero representation.
            empty=part([{},{}],device);torch.testing.assert_close(model.embedding(empty,empty),torch.zeros((2,506),device=device))
            loss=model.loss(model(common,sample),torch.tensor([[1.,1.],[0.,0.]],device=device));loss.backward()
            for name in ['base','ids']:
                grad=getattr(model.embedding,name).weight.grad
                self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
                torch.testing.assert_close(grad[0],torch.zeros_like(grad[0]))
            for tower in [model.ctr,model.cvr]:self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in tower.parameters()))
            torch.optim.Adam(model.parameters(),lr=.001).step()
            before=model.probabilities(common,sample).detach()
            stream=io.BytesIO();torch.save(dict(config=self.config(),state=model.state_dict()),stream);stream.seek(0)
            checkpoint=torch.load(stream,map_location=device,weights_only=True)
            restored=Model(checkpoint['config']).to(device);restored.load_state_dict(checkpoint['state'])
            torch.testing.assert_close(before,restored.probabilities(common,sample))
            torch.testing.assert_close(before[:,2],before[:,0]*before[:,1])

    def test_old_uniform_checkpoint_remains_loadable(self):
        config={k:v for k,v in self.config().items() if k not in ['id_embedding_dim','base_vocabulary_size','id_vocabulary_size']}
        old=Model(config);restored=Model(config);restored.load_state_dict(old.state_dict(),strict=True)
        inputs=part([{0:[(1,1.)],13:[(3,1.)]}],'cpu')
        torch.testing.assert_close(old.probabilities(inputs,inputs),restored.probabilities(inputs,inputs))
        self.assertEqual(restored.input_dim,414)

if __name__=='__main__':unittest.main()
