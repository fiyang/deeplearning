"""Check the real 4096 x 23 x 64 attention shape with a small synthetic vocabulary."""
import json
import time
import numpy as np
import torch
from data import FIELDS
from model import Model


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device('cuda')
    # Two known token IDs per field; repeated weighted history entries also pass
    # through the real EmbeddingBag and presence-mask code.
    vocabulary = np.array([23*(i+1)+i%23 for i in range(46)], dtype=np.uint32)
    config = dict(kind='esmm', fields=FIELDS, field_count=23, embedding_dim=18,
                  id_embedding_dim=64, vocabulary_size=47, base_vocabulary_size=43,
                  id_vocabulary_size=5, hidden=[360,200,80], dropout=.1,
                  ctr_prior=.0388714, cvr_prior=.0053533, attention_dim=64,
                  attention_heads=4, attention_dropout=.1, attention_layers=1)
    model = Model(config, vocabulary).to(device)
    batch = 4096
    lengths = torch.ones((batch,23), dtype=torch.int64, device=device)
    lengths[:,1:5] = 32
    offsets = torch.cat([torch.zeros(1,dtype=torch.int64,device=device),lengths.flatten().cumsum(0)])
    tokens = torch.repeat_interleave(torch.arange(batch*23,device=device)%23+1,lengths.flatten()).int()
    common = (tokens, offsets, torch.ones(len(tokens), device=device))
    sample = (torch.empty(0,dtype=torch.int32,device=device),torch.zeros(batch*23+1,dtype=torch.int64,device=device),torch.empty(0,device=device))
    labels = torch.zeros((batch,2),device=device)
    labels[::25,0] = 1
    labels[::1000,1] = 1
    optimizer = torch.optim.Adam(model.parameters(),lr=.0001,foreach=True)
    started = time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(model(common,sample),labels)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite smoke loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True,foreach=True)
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        scores = model.probabilities(common,sample)
    assert scores.shape == (batch,3) and torch.isfinite(scores).all()
    torch.testing.assert_close(scores[:,2],scores[:,0]*scores[:,1])
    torch.cuda.synchronize()
    report = dict(status='passed',batch=batch,attention_shape=[batch,23,64],loss=loss.item(),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  seconds=time.monotonic()-started,
                  limitation='Small synthetic vocabulary; full-data embedding/optimizer memory is additional.')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
