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


class MixedWidthEmbedding(nn.Module):
    """Separate compact tables for ID fields and other fields; shared by both tasks.

    Input token IDs are still the collision-free, train-only vocabulary indices.
    Persisted lookups make checkpoint inference independent of the raw vocabulary.
    """
    def __init__(self,config,vocabulary=None):
        super().__init__()
        self.fields=config['field_count'];self.base_dim=config['embedding_dim'];self.id_dim=config['id_embedding_dim']
        self.id_fields=[config['fields'].index(f) for f in ['101','205']]
        self.register_buffer('base_lookup',torch.zeros(config['vocabulary_size'],dtype=torch.int32))
        self.register_buffer('id_lookup',torch.zeros(config['vocabulary_size'],dtype=torch.int32))
        if vocabulary is not None:
            values=torch.as_tensor(vocabulary.astype('int64'))
            field=values.remainder(self.fields);is_id=(field==self.id_fields[0])|(field==self.id_fields[1])
            assert len(values)+1==config['vocabulary_size']
            assert int(is_id.sum())+1==config['id_vocabulary_size']
            assert int((~is_id).sum())+1==config['base_vocabulary_size']
            self.base_lookup[1:]=torch.where(is_id,0,(~is_id).cumsum(0)).to(torch.int32)
            self.id_lookup[1:]=torch.where(is_id,is_id.cumsum(0),0).to(torch.int32)
        self.base=nn.EmbeddingBag(config['base_vocabulary_size'],self.base_dim,mode='sum',include_last_offset=True,padding_idx=0)
        self.ids=nn.EmbeddingBag(config['id_vocabulary_size'],self.id_dim,mode='sum',include_last_offset=True,padding_idx=0)
        for table in [self.base,self.ids]:
            nn.init.normal_(table.weight,std=.01)
            with torch.no_grad():table.weight[0].zero_()

    def pool(self,part):
        tokens,offsets,weights=part
        base=self.base(self.base_lookup[tokens.long()],offsets,per_sample_weights=weights).reshape(-1,self.fields,self.base_dim)
        # Select just the user/item bags before the wide lookup. All values and
        # weights are retained, including an empty or multi-valued ID bag.
        n=base.shape[0];fields=torch.as_tensor(self.id_fields,device=tokens.device)
        bags=(torch.arange(n,device=tokens.device)[:,None]*self.fields+fields).flatten()
        starts=offsets[bags];lengths=offsets[bags+1]-starts
        compact_offsets=torch.cat([offsets.new_zeros(1),lengths.cumsum(0)])
        positions=torch.arange(int(compact_offsets[-1]),device=tokens.device)+torch.repeat_interleave(starts-compact_offsets[:-1],lengths)
        wide=self.ids(self.id_lookup[tokens[positions].long()],compact_offsets,per_sample_weights=weights[positions]).reshape(n,len(self.id_fields),self.id_dim)
        return base,wide

    def forward(self,common,sample):
        base_a,wide_a=self.pool(common);base_b,wide_b=self.pool(sample)
        base=base_a+base_b;wide=wide_a+wide_b
        return torch.cat([wide[:,self.id_fields.index(f)] if f in self.id_fields else base[:,f] for f in range(self.fields)],dim=1)


class Model(nn.Module):
    def __init__(self,config,vocabulary=None):
        super().__init__();self.config=config;self.kind=config['kind']
        self.fields=config['field_count'];self.mixed=config.get('id_embedding_dim',config['embedding_dim'])!=config['embedding_dim']
        self.input_dim=(self.fields-2)*config['embedding_dim']+2*config['id_embedding_dim'] if self.mixed else self.fields*config['embedding_dim']
        if self.mixed:
            assert self.kind=='esmm'
            self.embedding=MixedWidthEmbedding(config,vocabulary)
        else:
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

    def forward(self,common,sample):
        assert self.kind=='esmm'
        def pooled(part):
            ids,offsets,weights=part
            return self.embedding(ids,offsets,per_sample_weights=weights).reshape(-1,self.input_dim)
        x=self.embedding(common,sample) if self.mixed else pooled(common)+pooled(sample)
        a=self.ctr(x);b=self.cvr(x)
        return a[:,1]-a[:,0],b[:,1]-b[:,0]

    def loss(self,logits,labels):
        a,b=logits;click,conversion=labels.unbind(1)
        first=F.binary_cross_entropy_with_logits(a,click)
        if self.kind=='esmm':return first+product_bce_logits(a,b,conversion)
        mask=click==1
        return first+(F.binary_cross_entropy_with_logits(b[mask],conversion[mask]) if mask.any() else b.sum()*0)

    def probabilities(self,common,sample):
        a,b=self(common,sample);p=a.sigmoid();q=b.sigmoid()
        return torch.stack([p,q,p*q],dim=1)
