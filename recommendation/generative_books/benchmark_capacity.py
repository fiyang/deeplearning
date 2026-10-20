"""Synthetic 400k-item capacity check; not a recommendation-quality experiment."""
import json
import time
import numpy as np
import torch
from download import ROOT, save_json
from models import create_model

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
rng = np.random.default_rng(123)
keys = np.unique(rng.integers(0,512**3,size=800000,dtype=np.int64))[:400000]
codes = np.column_stack([keys//512**2,keys//512%512,keys%512,np.zeros(len(keys),dtype=np.int64)])
history = torch.tensor(rng.integers(1,len(codes)+1,size=(512,20)),device="cuda")
users = torch.tensor(rng.integers(0,2000,size=512),device="cuda")
targets = torch.tensor(rng.integers(0,len(codes),size=512),device="cuda")
report = {"purpose":"synthetic capacity only, no quality metric", "catalog_items":len(codes), "models":{}}
for name in ("tiger","decoder_only"):
    model = create_model(name,codes).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for step in range(12):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss = model(history,users,targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        assert bool(torch.isfinite(loss))
        if step == 1:
            torch.cuda.synchronize()
            start = time.monotonic()
    torch.cuda.synchronize()
    train_seconds = time.monotonic()-start
    model.eval()
    seen = torch.zeros(64,len(codes),dtype=torch.bool,device="cuda")
    seen.scatter_(1,history[:64]-1,True)
    start = time.monotonic()
    with torch.autocast("cuda",dtype=torch.bfloat16),torch.no_grad():
        predicted,_ = model.recommend(history[:64],users[:64],seen,20)
    torch.cuda.synchronize()
    assert bool((predicted >= 0).all())
    assert not bool(seen.gather(1,predicted).any())
    result = {"parameters":sum(p.numel() for p in model.parameters()),
              "train_examples_per_second_10_steps":5120/train_seconds,
              "beam20_64_requests_seconds":time.monotonic()-start,
              "peak_gpu_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
    report["models"][name] = result
    print("CAPACITY",name,json.dumps(result),flush=True)
    del model,optimizer,seen
    torch.cuda.empty_cache()
save_json(ROOT/"runs/capacity_check.json",report)
print("CAPACITY_CHECK_OK",flush=True)
