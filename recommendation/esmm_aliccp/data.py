"""PaddleRec Ali-CCP parser: preserve every field value; fit vocabulary on train only."""
from array import array
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

FIELDS = ['101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124', '125',
          '126', '127', '128', '129', '205', '206', '207', '210', '216', '508', '509', '702', '853', '301']
FIELD_INDEX = {f: i for i, f in enumerate(FIELDS)}
N_FIELDS = len(FIELDS)
TRAIN_NAME = 'traindata_10w.csv'
TEST_NAME = 'testdata_10w.csv'


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def parse_line(line):
    # The fourth CSV element is ALREADY a feature, not a feature-count column.
    prefix = line.rstrip('\r\n').split(',', 3)
    if len(prefix) != 4: raise ValueError('Missing fields')
    sample_id, click, conversion = map(int, prefix[:3])
    if click not in (0,1) or conversion not in (0,1) or conversion > click:
        raise ValueError('Invalid sequential click/conversion labels')
    bags=[[] for _ in FIELDS]
    for token in prefix[3].split(','):
        field, value = token.split(':')
        i=FIELD_INDEX[field]; value=int(value)
        if value < 0: raise ValueError('Negative feature ID')
        # Field-qualified IDs prevent accidental collisions between namespaces.
        bags[i].append(value*N_FIELDS+i)
    return sample_id, (click, conversion), bags


def parse_file(path, cache):
    path=Path(path); cache=Path(cache)
    cache.mkdir(parents=True,exist_ok=False)
    token_buffer=array('q'); offset_buffer=array('q',[0]); token_count=0
    labels=[]; sample_ids=[]; user_ids=[]; item_ids=[]; histogram=Counter()
    maximum=np.zeros(N_FIELDS,dtype=np.int64); totals=np.zeros(N_FIELDS,dtype=np.int64)
    lengths=[]; digest=hashlib.sha256()
    with path.open('rb') as source, (cache/'tokens.bin').open('wb') as token_file, (cache/'offsets.bin').open('wb') as offset_file:
        for row, raw in enumerate(source):
            digest.update(raw)
            try: sample_id, label, bags = parse_line(raw.decode('ascii'))
            except Exception as exc: raise ValueError(f'{path.name} row {row+1}: {exc}') from exc
            sample_ids.append(sample_id); labels.append(label); histogram[str(label)]+=1
            user_ids.append(bags[0][0]//N_FIELDS if bags[0] else -1)
            item_bag=bags[FIELD_INDEX['205']]
            item_ids.append(item_bag[0]//N_FIELDS if item_bag else -1)
            sizes=[len(b) for b in bags]; lengths.append(sum(sizes))
            maximum=np.maximum(maximum,sizes); totals+=sizes
            for bag in bags:
                token_buffer.extend(bag); token_count+=len(bag); offset_buffer.append(token_count)
            if (row+1)%1000==0:
                token_buffer.tofile(token_file); offset_buffer.tofile(offset_file)
                token_buffer=array('q'); offset_buffer=array('q')
            if (row+1)%25000==0: print(f'PARSED {path.name} rows={row+1} tokens={token_count}',flush=True)
        token_buffer.tofile(token_file); offset_buffer.tofile(offset_file)
    y=np.asarray(labels,dtype=np.int8)
    if not len(y): raise ValueError('Empty file')
    for name, values in [('labels',y),('sample_ids',sample_ids),('user_ids',user_ids),('item_ids',item_ids)]:
        np.save(cache/(name+'.npy'),np.asarray(values))
    info=dict(file=path.name,source=f'https://paddlerec.bj.bcebos.com/esmm/{path.name}',sha256=digest.hexdigest(),
              bytes=path.stat().st_size,rows=len(y),clicks=int(y[:,0].sum()),conversions=int(y[:,1].sum()),
              invalid_conversion_without_click=0,label_pairs=dict(histogram),fields=FIELDS,total_feature_tokens=token_count,
              feature_tokens_quantiles={str(q):float(np.quantile(lengths,q)) for q in [0,.5,.9,.99,1]},
              field_max_lengths=dict(zip(FIELDS,maximum.tolist())),field_total_tokens=dict(zip(FIELDS,totals.tolist())),
              unique_users=len(set(user_ids)-{-1}),unique_items=len(set(item_ids)-{-1}),
              missing_user_rows=user_ids.count(-1),missing_item_rows=item_ids.count(-1),sample_ids_unique=len(set(sample_ids))==len(y),
              timestamp_available=False,original_feature_weights_available=False,feature_values_truncated=0)
    write_json(cache/'manifest.json',info)
    print('FILE_AUDIT '+json.dumps(info),flush=True)
    return info


class RawTable:
    def __init__(self,cache):
        self.cache=Path(cache); self.manifest=json.loads((self.cache/'manifest.json').read_text())
        self.tokens=np.memmap(self.cache/'tokens.bin',dtype=np.int64,mode='r')
        self.offsets=np.memmap(self.cache/'offsets.bin',dtype=np.int64,mode='r')
        self.labels=np.load(self.cache/'labels.npy')
        self.sample_ids=np.load(self.cache/'sample_ids.npy')
        self.user_ids=np.load(self.cache/'user_ids.npy');self.item_ids=np.load(self.cache/'item_ids.npy')
        assert len(self.offsets)==len(self.labels)*N_FIELDS+1
        assert self.offsets[0]==0 and self.offsets[-1]==len(self.tokens)


def stratified_split(labels,seed=42,fraction=.2):
    rng=np.random.default_rng(seed); train=[]; validation=[]
    strata=labels[:,0]*2+labels[:,1]
    for value in np.unique(strata):
        ids=np.flatnonzero(strata==value); rng.shuffle(ids)
        if len(ids)<2: raise ValueError('Too few examples of a label stratum for holdout')
        n=min(len(ids)-1,max(1,int(round(len(ids)*fraction))))
        validation.extend(ids[:n]); train.extend(ids[n:])
    return np.sort(train),np.sort(validation)


def fit_vocabulary(table,train_rows):
    pieces=[table.tokens[table.offsets[i*N_FIELDS]:table.offsets[(i+1)*N_FIELDS]] for i in train_rows]
    return np.unique(np.concatenate(pieces))


def remap_tokens(tokens,vocabulary):
    if max(int(tokens.max()),int(vocabulary[-1])) < 100_000_000:
        lookup=np.zeros(max(int(tokens.max()),int(vocabulary[-1]))+1,dtype=np.int32)
        lookup[vocabulary]=np.arange(1,len(vocabulary)+1,dtype=np.int32)
        return lookup[tokens]
    result=np.zeros(len(tokens),dtype=np.int32)
    for start in range(0,len(tokens),1_000_000):
        values=tokens[start:start+1_000_000]; positions=np.searchsorted(vocabulary,values)
        safe=np.minimum(positions,len(vocabulary)-1); known=(positions<len(vocabulary))&(vocabulary[safe]==values)
        result[start:start+len(values)]=np.where(known,positions+1,0)
    return result


class DeviceTable:
    def __init__(self,table,vocabulary,device):
        self.tokens=torch.as_tensor(remap_tokens(table.tokens,vocabulary),dtype=torch.long,device=device)
        self.offsets=torch.as_tensor(np.array(table.offsets),dtype=torch.long,device=device)
        self.fields=torch.arange(N_FIELDS,device=device)
        self.zero=torch.zeros(1,dtype=torch.long,device=device)

    def batch(self,rows):
        rows=torch.as_tensor(rows,dtype=torch.long,device=self.tokens.device)
        bag_ids=(rows[:,None]*N_FIELDS+self.fields).flatten()
        starts=self.offsets[bag_ids]; lengths=self.offsets[bag_ids+1]-starts
        offsets=torch.cat((self.zero,lengths.cumsum(0)))
        token_indices=torch.arange(int(offsets[-1]),device=self.tokens.device)+torch.repeat_interleave(starts-offsets[:-1],lengths)
        return self.tokens[token_indices],offsets


def split_counts(table,indices):
    y=table.labels[indices]
    return dict(rows=len(indices),clicks=int(y[:,0].sum()),conversions=int(y[:,1].sum()),
                users=len(set(table.user_ids[indices]) - {-1}),items=len(set(table.item_ids[indices]) - {-1}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path('../data/aliccp-paddlerec'));args=p.parse_args()
    for name in [TRAIN_NAME,TEST_NAME]:
        output=args.data_dir/(name+'.cache')
        if output.exists():
            old=json.loads((output/'manifest.json').read_text())
            assert sha256(args.data_dir/name)==old['sha256'],'Raw data changed'
            print('CACHE_VERIFIED '+name,flush=True)
        else:parse_file(args.data_dir/name,output)
