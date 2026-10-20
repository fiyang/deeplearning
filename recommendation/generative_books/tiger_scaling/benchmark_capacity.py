"""Actual full-catalog batch capacity check; no quality metrics or saved weights."""
import time
import numpy as np
import torch
from artifacts import ROOT, save_json
from models import create_model, Tiger
from train import Dataset, amp


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda')
    data=Dataset(ROOT.parent/'data',device)
    codes=np.load(ROOT.parent/'data/semantic_ids.npy')
    old=Tiger(codes,collision_size=1024,d_model=128,d_ff=1024)
    old_count=sum(p.numel() for p in old.parameters())
    del old
    model=create_model('tiger',codes).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    elapsed=[]
    for step in range(24):
        ids=torch.randint(len(data.train_position),(512,),device=device)
        h,u,y=data.batch('train',ids)
        torch.cuda.synchronize()
        start=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with amp(device):
            loss=model(h,u,y)
        assert bool(torch.isfinite(loss))
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        if step>=4:
            elapsed.append(time.monotonic()-start)
    model.eval()
    users=np.arange(64)
    h,u,_=data.batch('val',users)
    seen=data.seen('val',users)
    with torch.no_grad(),amp(device):
        predictions,_=model.recommend(h,u,seen,beam_size=20)
    torch.cuda.synchronize()
    assert predictions.shape==(64,20)
    assert not bool((seen.gather(1,predictions.clamp_min(0)) & predictions.ge(0)).any())
    count=sum(p.numel() for p in model.parameters())
    stats={'baseline_parameters':old_count,'scaled_parameters':count,'parameter_ratio':count/old_count,
        'training_batch_size':512,'eval_batch_size':64,'beam_size':20,'timed_steps':len(elapsed),
        'examples_per_second':512*len(elapsed)/sum(elapsed),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'note':'Capacity and short-run throughput only, not recommendation quality or service latency.'}
    save_json(ROOT/'capacity_check.json',stats)
    print('CAPACITY_OK',stats,flush=True)


if __name__=='__main__':
    main()
