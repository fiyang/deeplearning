"""Verify splits, train-only vocabulary, raw feature decoding and saved predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import FIELDS,TRAIN_NAME,TEST_NAME,RawTable,fit_vocabulary,stratified_split,sha256,write_json
from metrics import funnel_metrics
from score import load_model,pack_bags


def roc_area(y,p):
    # Independent descending ROC integration, with all tied thresholds merged.
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    order=np.argsort(-p,kind='stable'); labels=y[order]
    ends=np.r_[np.flatnonzero(np.diff(p[order])),len(p)-1]
    tp=np.r_[0.,np.cumsum(labels)[ends]]
    fp=np.r_[0.,np.cumsum(1-labels)[ends]]
    return float(np.trapezoid(tp/tp[-1],fp/fp[-1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('../data/aliccp-paddlerec'));args=p.parse_args()
    torch.set_num_threads(4)
    root=args.run;summary=json.loads((root/'summary.json').read_text())
    config=json.loads((root/'experiment_config.json').read_text())
    raw_train=RawTable(args.data_dir/(TRAIN_NAME+'.cache')); raw_test=RawTable(args.data_dir/(TEST_NAME+'.cache'))
    for raw in [raw_train,raw_test]:assert sha256(args.data_dir/raw.manifest['file'])==raw.manifest['sha256']
    indices=np.load(root/'split_indices.npz'); tr,va=stratified_split(raw_train.labels,config['seed'])
    np.testing.assert_array_equal(tr,indices['train']);np.testing.assert_array_equal(va,indices['validation'])
    assert not np.intersect1d(tr,va).size
    vocabulary=np.load(root/'vocabulary.npy');np.testing.assert_array_equal(vocabulary,fit_vocabulary(raw_train,tr))
    chosen=np.unique(np.r_[np.linspace(0,len(raw_test.labels)-1,128,dtype=np.int64),np.flatnonzero(raw_test.labels[:,1])])
    selected=set(chosen.tolist());rows=[]
    # Decode independently of data.parse_line and compare every value in selected rows.
    with (args.data_dir/TEST_NAME).open() as f:
        for index,line in enumerate(f):
            if index not in selected:continue
            elements=line.strip().split(',');sample,click,conversion=map(int,elements[:3])
            assert sample==raw_test.sample_ids[index]
            np.testing.assert_array_equal([click,conversion],raw_test.labels[index])
            bags=[[] for _ in FIELDS]
            for element in elements[3:]:
                field,value=element.split(':');fi=FIELDS.index(field)
                bags[fi].append(int(value)*len(FIELDS)+fi)
            for fi,bag in enumerate(bags):
                k=index*len(FIELDS)+fi
                np.testing.assert_array_equal(bag,raw_test.tokens[raw_test.offsets[k]:raw_test.offsets[k+1]])
            rows.append(bags)
    packed=pack_bags(rows,vocabulary);fixed=json.loads((root/'checkpoints_fixed.json').read_text());results={}
    for kind in ['independent','esmm']:
        path=root/kind; assert sha256(path/'best.pt')==fixed[kind]
        predictions=np.load(path/'test_predictions.npz'); scores=predictions['scores']
        np.testing.assert_array_equal(predictions['labels'],raw_test.labels)
        np.testing.assert_array_equal(predictions['sample_ids'],raw_test.sample_ids)
        np.testing.assert_allclose(scores[:,2],scores[:,0]*scores[:,1],rtol=1e-6,atol=1e-8)
        assert np.all(scores[:,2]<=scores[:,0]+1e-7)
        recalculated=funnel_metrics(raw_test.labels,scores)
        for task,metrics in recalculated.items():
            for key in ['auc','average_precision','log_loss','accuracy_at_0_5','positives','count']:
                assert abs(metrics[key]-summary['results'][kind]['test'][task][key])<1e-12
        click,conversion=raw_test.labels.T
        for task,y,probs in [('ctr',click,scores[:,0]),('cvr',conversion[click==1],scores[click==1,1]),('ctcvr',conversion,scores[:,2])]:
            assert abs(roc_area(y,probs)-recalculated[task]['auc'])<1e-12
        model,_=load_model(path/'best.pt')
        with torch.inference_mode():cpu=model.probabilities(*packed).numpy()
        max_error=float(np.max(np.abs(cpu-scores[chosen])))
        np.testing.assert_allclose(cpu,scores[chosen],rtol=1e-4,atol=2e-6)
        results[kind]=dict(cpu_reload_rows=len(chosen),max_cpu_gpu_absolute_error=max_error,
                          independent_roc_matches=True,checkpoint_hash_matches=True)
    report=dict(passed=True,train_validation_disjoint=True,train_only_vocabulary_rebuilt=True,
                original_file_hashes_match=True,raw_rows_redecoded=len(chosen),all_test_conversion_rows_checked=True,
                source_train_test_files_separate=True,models=results)
    write_json(root/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
