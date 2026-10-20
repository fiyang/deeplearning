"""Variable-length field pooling + ESMM. No feature truncation."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def product_bce_logits(a,b,y):
    log_yes=F.logsigmoid(a)+F.logsigmoid(b)
    log_no=torch.logaddexp(F.logsigmoid(-a),F.logsigmoid(a)+F.logsigmoid(-b))
    return -(y*log_yes+(1-y)*log_no).mean()


def make_tower(width,hidden,dropout):
    layers=[]
    for h in hidden:
        layers.extend([nn.Linear(width,h),nn.ReLU()])
        if dropout:layers.append(nn.Dropout(dropout))
        width=h
    layers.append(nn.Linear(width,2))
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;self.kind=config['kind']
        self.fields=config['field_count'];self.input_dim=self.fields*config['embedding_dim']
        self.embedding=nn.EmbeddingBag(config['vocabulary_size'],config['embedding_dim'],mode='sum',include_last_offset=True,padding_idx=0)
        nn.init.normal_(self.embedding.weight,std=.01)
        with torch.no_grad():self.embedding.weight[0].zero_()
        if self.kind=='independent':
            self.cvr_embedding=nn.EmbeddingBag(config['vocabulary_size'],config['embedding_dim'],mode='sum',include_last_offset=True,padding_idx=0)
            self.cvr_embedding.load_state_dict(self.embedding.state_dict())
        self.ctr=make_tower(self.input_dim,config['hidden'],config['dropout'])
        self.cvr=make_tower(self.input_dim,config['hidden'],config['dropout'])
        for head,rate in [(self.ctr,config['ctr_prior']),(self.cvr,config['cvr_prior'])]:
            rate=min(max(rate,1e-7),1-1e-7)
            with torch.no_grad():head[-1].bias.copy_(torch.tensor([0.,math.log(rate/(1-rate))]))

    def forward(self,ids,offsets):
        x=self.embedding(ids,offsets).reshape(-1,self.input_dim)
        v=self.cvr_embedding(ids,offsets).reshape(-1,self.input_dim) if self.kind=='independent' else x
        a=self.ctr(x);b=self.cvr(v)
        return a[:,1]-a[:,0],b[:,1]-b[:,0]

    def loss(self,logits,labels):
        a,b=logits;click,conversion=labels.unbind(1)
        first=F.binary_cross_entropy_with_logits(a,click)
        if self.kind=='esmm':return first+product_bce_logits(a,b,conversion)
        mask=click==1
        return first+(F.binary_cross_entropy_with_logits(b[mask],conversion[mask]) if mask.any() else b.sum()*0)

    def probabilities(self,ids,offsets):
        a,b=self(ids,offsets);p=a.sigmoid();q=b.sigmoid()
        return torch.stack([p,q,p*q],dim=1)
