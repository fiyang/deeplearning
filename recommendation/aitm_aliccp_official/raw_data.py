"""Memory-mapped CSR tables; fit vocabulary exclusively on the training partition."""
import json
from pathlib import Path
import numpy as np
import torch

FIELDS=['101','109_14','110_14','127_14','150_14','121','122','124','125','126','127','128','129','205','206','207','210','216','508','509','702','853','301']

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

class RawTable:
    def __init__(self,path):
        self.path=Path(path);self.info=json.loads((self.path/'manifest.json').read_text())
        assert self.info['fields']==FIELDS
        self.n=self.info['sample']['rows']
        self.labels=np.memmap(self.path/'labels.bin',dtype='u1',mode='r',shape=(self.n,2))
        self.refs=np.memmap(self.path/'common_refs.bin',dtype='<u4',mode='r')
        self.parts={}
        assert len(self.refs)==self.n and self.refs.max()<self.info['common']['rows']
        for name in ['common','sample']:
            p={k:np.memmap(self.path/f'{name}.{k}.bin',dtype=d,mode='r') for k,d in [('tokens','<u4'),('offsets','<u4'),('weights','<f4')]}
            assert len(p['offsets'])==self.info[name]['rows']*len(FIELDS)+1
            assert p['offsets'][0]==0 and p['offsets'][-1]==len(p['tokens'])==len(p['weights'])
            self.parts[name]=p

def split_rows(labels,seed=42,fraction=.1):
    rng=np.random.default_rng(seed);mask=np.zeros(len(labels),dtype=bool)
    strata=labels[:,0]*2+labels[:,1]
    valid=strata!=1
    for value in np.unique(strata):
        if value==1:continue
        rows=np.flatnonzero(strata==value);rng.shuffle(rows)
        count=max(1,round(len(rows)*fraction))
        if count>=len(rows):raise ValueError('Too few positives for holdout')
        mask[rows[:count]]=True
    return np.flatnonzero(valid&~mask).astype(np.int32),np.flatnonzero(mask).astype(np.int32)

def fit_vocabulary(raw,train_rows,out):
    maximum=max(raw.info[p]['max_token'] for p in raw.parts)
    if maximum>500_000_000:raise ValueError('Dense lookup exceeds memory budget')
    seen=np.zeros(maximum+1,dtype=bool);selected=np.zeros(raw.n,dtype=bool);selected[train_rows]=True
    common_selected=np.zeros(raw.info['common']['rows'],dtype=bool);common_selected[raw.refs[train_rows]]=True
    for name,keep in [('sample',selected),('common',common_selected)]:
        part=raw.parts[name];offsets=part['offsets'][::len(FIELDS)]
        for start in range(0,len(keep),10000):
            end=min(start+10000,len(keep));left,right=int(offsets[start]),int(offsets[end])
            token_keep=np.repeat(keep[start:end],np.diff(offsets[start:end+1]).astype(np.int64))
            seen[part['tokens'][left:right][token_keep]]=True
        print('VOCAB_SCANNED '+name,flush=True)
    vocabulary=np.flatnonzero(seen).astype(np.uint32);np.save(Path(out)/'vocabulary.npy',vocabulary)
    lookup=np.zeros(maximum+1,dtype=np.int32);lookup[vocabulary]=np.arange(1,len(vocabulary)+1,dtype=np.int32)
    return vocabulary,lookup

def lookup_from_vocabulary(vocabulary):
    lookup=np.zeros(int(vocabulary[-1])+1,dtype=np.int32)
    lookup[vocabulary]=np.arange(1,len(vocabulary)+1,dtype=np.int32)
    return lookup

def remap(raw,lookup,folder):
    folder=Path(folder);folder.mkdir(exist_ok=True);stats={}
    for name,part in raw.parts.items():
        target=folder/f'{name}.tokens.bin';mapped=np.memmap(target,dtype='<i4',mode='w+',shape=(len(part['tokens']),));unknown=0
        for start in range(0,len(mapped),2_000_000):
            tokens=part['tokens'][start:start+2_000_000];known=tokens<len(lookup)
            chunk=np.zeros(len(tokens),dtype=np.int32);chunk[known]=lookup[tokens[known]]
            mapped[start:start+len(chunk)]=chunk;unknown+=int((chunk==0).sum())
        mapped.flush();stats[name]={'tokens':len(mapped),'unknown_tokens':unknown};del mapped
    write_json(folder/'mapping_stats.json',stats);return stats

class DeviceTable:
    def __init__(self,raw,mapped,device):
        self.device=device;self.parts={};self.fields=torch.arange(len(FIELDS),device=device)
        self.zero=torch.zeros(1,dtype=torch.int64,device=device)
        for name,part in raw.parts.items():
            tokens=np.memmap(Path(mapped)/f'{name}.tokens.bin',dtype='<i4',mode='r')
            # Preserve uint32 offsets with int64 when signed int32 would overflow.
            odtype=torch.int64 if int(part['offsets'][-1])>2**31-1 else torch.int32
            self.parts[name]={
                'tokens':torch.tensor(tokens,device=device,dtype=torch.int32),
                'offsets':torch.tensor(part['offsets'].astype(np.int64 if odtype==torch.int64 else np.int32),device=device,dtype=odtype),
                'weights':torch.tensor(part['weights'],device=device,dtype=torch.float32)}
        self.refs=torch.tensor(raw.refs.astype(np.int32),device=device)
        self.labels=torch.tensor(np.array(raw.labels),device=device,dtype=torch.float32)

    def gather(self,name,rows):
        part=self.parts[name];bags=(rows[:,None].long()*len(FIELDS)+self.fields).flatten()
        start=part['offsets'][bags].long();lengths=(part['offsets'][bags+1].long()-start)
        offsets=torch.cat([self.zero,lengths.cumsum(0)])
        positions=torch.arange(int(offsets[-1]),device=self.device)+torch.repeat_interleave(start-offsets[:-1],lengths)
        return part['tokens'][positions],offsets,part['weights'][positions]

    def batch(self,rows):
        rows=torch.as_tensor(rows,dtype=torch.long,device=self.device)
        return self.gather('common',self.refs[rows]),self.gather('sample',rows)

def counts(labels,rows=None):
    y=labels if rows is None else labels[rows]
    return dict(rows=len(y),clicks=int(y[:,0].sum()),conversions=int(y[:,1].sum()))
