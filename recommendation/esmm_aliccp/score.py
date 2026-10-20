"""Load a saved checkpoint and score a zero-based row from the published CSV."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import FIELDS, parse_line, sha256
from model import Model


def pack_bags(rows, vocabulary):
    values=[]; offsets=[0]
    for bags in rows:
        for bag in bags:
            values.extend(bag); offsets.append(len(values))
    values=np.asarray(values,dtype=np.int64)
    positions=np.searchsorted(vocabulary,values)
    safe=np.minimum(positions,len(vocabulary)-1)
    known=(positions<len(vocabulary)) & (vocabulary[safe]==values)
    ids=np.where(known,positions+1,0)
    return torch.from_numpy(ids),torch.tensor(offsets,dtype=torch.long)


def load_model(checkpoint_path):
    checkpoint_path=Path(checkpoint_path); root=checkpoint_path.parent.parent
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    schema=json.loads((root/'feature_schema.json').read_text())
    assert schema['fields']==FIELDS
    assert sha256(root/'feature_schema.json')==checkpoint['config']['feature_schema_sha256']
    assert sha256(root/'vocabulary.npy')==schema['vocabulary_sha256']
    vocabulary=np.load(root/'vocabulary.npy')
    model=Model(checkpoint['config']); model.load_state_dict(checkpoint['model_state']); model.eval()
    return model,vocabulary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--row-index',type=int,default=0)
    args=p.parse_args()
    if args.row_index<0:p.error('row-index must be nonnegative')
    torch.set_num_threads(4)
    model,vocabulary=load_model(args.checkpoint)
    with args.csv.open() as f:
        for index,line in enumerate(f):
            if index==args.row_index:break
        else:raise IndexError('row-index is outside the CSV')
    sample_id,labels,bags=parse_line(line)
    with torch.inference_mode():scores=model.probabilities(*pack_bags([bags],vocabulary))[0].tolist()
    # Labels and the sample ID are returned for inspection, never model inputs.
    print(json.dumps(dict(sample_id=sample_id,row_index=args.row_index,pCTR=scores[0],pCVR=scores[1],
                          pCTCVR=scores[2],observed_click=labels[0],observed_conversion=labels[1]),indent=2))


if __name__=='__main__':main()
