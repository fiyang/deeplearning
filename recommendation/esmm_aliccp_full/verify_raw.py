"""Independent raw CSV versus CSR verification, including rare positive labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import RawTable,FIELDS,write_json


def check_bags(raw,part,row,payload):
    expected=[[] for _ in FIELDS]
    for token in payload.rstrip(b'\r\n').split(b'\1'):
        f,value=token.split(b'\2');v,w=value.split(b'\3');i=FIELDS.index(f.decode())
        expected[i].append((int(v)*len(FIELDS)+i,np.float32(w)))
    for f,items in enumerate(expected):
        lo,hi=map(int,raw.parts[part]['offsets'][row*len(FIELDS)+f:row*len(FIELDS)+f+2])
        np.testing.assert_array_equal(raw.parts[part]['tokens'][lo:hi],[x[0] for x in items])
        np.testing.assert_allclose(raw.parts[part]['weights'][lo:hi],[x[1] for x in items],rtol=1e-6,atol=1e-6)


def verify(data,split):
    raw=RawTable(data/'cache'/split);rng=np.random.default_rng(20261007)
    positives=np.flatnonzero(raw.labels[:,1]);selected=set(range(min(32,raw.n)))|set(rng.integers(0,raw.n,64).tolist())|{raw.n-1}
    if len(positives):selected.update(rng.choice(positives,min(32,len(positives)),replace=False).tolist())
    needed={};checked=0
    with (data/'raw'/split/f'sample_skeleton_{split}.csv').open('rb') as f:
        for row,line in enumerate(f):
            if row not in selected:continue
            sid,c,y,cid,n,payload=line.split(b',',5)
            np.testing.assert_array_equal(raw.labels[row],[int(c),int(y)])
            assert len(payload.rstrip(b'\r\n').split(b'\1'))==int(n)
            check_bags(raw,'sample',row,payload);ref=int(raw.refs[row])
            if ref in needed:assert needed[ref]==cid
            needed[ref]=cid;checked+=1
    assert row+1==raw.n and checked==len(selected)
    common_checked=0
    with (data/'raw'/split/f'common_features_{split}.csv').open('rb') as f:
        for row,line in enumerate(f):
            if row not in needed:continue
            cid,n,payload=line.split(b',',2);assert cid==needed[row]
            assert len(payload.rstrip(b'\r\n').split(b'\1'))==int(n)
            check_bags(raw,'common',row,payload);common_checked+=1
    assert common_checked==len(needed) and row+1==raw.info['common']['rows']
    report=dict(split=split,sample_rows_checked=checked,common_rows_checked=common_checked,positive_samples_checked=sum(int(raw.labels[i,1]) for i in selected),
                checks=['Raw line counts','Common-feature foreign keys','Every sampled field, token and original weight','Raw labels against cached labels'],status='passed')
    write_json(data/f'verification-{split}.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path('../data/aliccp-full'));p.add_argument('--split',choices=['train','test'],required=True);a=p.parse_args();verify(a.data_dir,a.split)
