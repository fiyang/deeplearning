"""Reuse a small real Beauty fixture to validate the runner while Books downloads."""
import json
import shutil
import numpy as np
from download import ROOT, save_json
from prepare import prefix_examples

source = ROOT.parent / "generative_beauty/data"
target = ROOT / "smoke_fixture"
target.mkdir(exist_ok=True)
records = json.loads((source / "sequences.json").read_text())[:2048]
seq = [r["items"] for r in records]
offsets,pos,owner,val,test = prefix_examples(seq)
flat = np.concatenate(seq).astype(np.int32)
codes = np.load(source / "semantic_ids.npy")
pop = np.bincount(flat-1,minlength=len(codes))
np.add.at(pop,flat[val]-1,-1)
np.add.at(pop,flat[test]-1,-1)
for key,value in dict(sequence_items=flat,sequence_offsets=offsets,user_hash=[r["user_hash"] for r in records],
                      train_position=pos,train_user=owner,val_position=val,test_position=test,popularity=pop).items():
    np.save(target / f"{key}.npy",value)
for name in ("semantic_ids.npy","quantizer_info.json","download_manifest.json"):
    shutil.copy2(source / name,target / name)
save_json(target / "dataset_info.json",{"dataset":"REAL BEAUTY FIXTURE ONLY - NOT BOOKS RESULTS",
                                         "users":len(records),"items":len(codes),"train_examples":len(pos)})
print("FIXTURE_READY",target)
