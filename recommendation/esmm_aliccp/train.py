"""Train ESMM on real PaddleRec Ali-CCP; preserve source train/test separation."""
import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from data import FIELDS,TRAIN_NAME,TEST_NAME,RawTable,DeviceTable,stratified_split,fit_vocabulary,split_counts,sha256,write_json
from model import Model
from metrics import funnel_metrics,auc_interval


@torch.inference_mode()
def predict(model,table,indices,batch_size=1024):
    model.eval();out=[]
    for start in range(0,len(indices),batch_size):
        out.append(model.probabilities(*table.batch(indices[start:start+batch_size])).cpu().numpy())
    return np.concatenate(out)


def train_one(args,config,initial,table,raw,train_rows,val_rows):
    path=args.output/config['kind'];path.mkdir()
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    device=table.tokens.device;model=Model(config).to(device)
    incompatible=model.load_state_dict(initial,strict=(config['kind']=='esmm'))
    if config['kind']=='independent':
        assert incompatible.missing_keys==['cvr_embedding.weight'] and not incompatible.unexpected_keys
        model.cvr_embedding.load_state_dict(model.embedding.state_dict())
    config=dict(config,input_dim=model.input_dim,parameter_count=sum(p.numel() for p in model.parameters()))
    write_json(path/'config.json',config)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.999),eps=1e-8,weight_decay=args.weight_decay)
    labels=torch.as_tensor(raw.labels,dtype=torch.float32,device=device)
    rng=np.random.default_rng(args.seed+1000);best=-1.;patience_best=-1.;stale=0;history=[];gradient_checks={}
    started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        tick=time.perf_counter();model.train();total=torch.zeros((),device=device)
        order=rng.permutation(train_rows)
        for start in range(0,len(order),args.batch_size):
            idx=order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss=model.loss(model(*table.batch(idx)),labels[idx])
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward()
            if not gradient_checks:
                for name in ['embedding','ctr','cvr']+(['cvr_embedding'] if config['kind']=='independent' else []):
                    gradient_checks[name]=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in getattr(model,name).parameters())
                assert all(gradient_checks.values())
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();total+=loss.detach()*len(idx)
        val_scores=predict(model,table,val_rows,args.batch_size)
        val=funnel_metrics(raw.labels[val_rows],val_scores);metric=val['ctcvr']['auc']
        row=dict(epoch=epoch,train_loss=total.item()/len(train_rows),val_ctr_auc=val['ctr']['auc'],
                 val_cvr_auc=val['cvr']['auc'],val_ctcvr_auc=metric,val_ctcvr_ap=val['ctcvr']['average_precision'],
                 val_ctcvr_log_loss=val['ctcvr']['log_loss'],seconds=time.perf_counter()-tick)
        history.append(row)
        if metric>best:
            best=metric
            torch.save(dict(config=config,epoch=epoch,validation=val,
                            model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),path/'best.pt')
            np.save(path/'validation_scores.npy',val_scores)
        if metric>patience_best+args.min_delta:patience_best=metric;stale=0
        else:stale+=1
        with (path/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        print(f"[{config['kind']}] EPOCH {epoch:02d}/{args.epochs} | val CTR={val['ctr']['auc']:.5f} CVR={val['cvr']['auc']:.5f} CTCVR={metric:.5f} best={best:.5f} | train_loss={row['train_loss']:.5f} | {row['seconds']:.1f}s",flush=True)
        if stale>=args.patience:
            print(f"[{config['kind']}] EARLY_STOPPING patience={args.patience}",flush=True);break
    checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
    result=dict(kind=config['kind'],best_epoch=checkpoint['epoch'],completed_epochs=len(history),validation=checkpoint['validation'],
                parameter_count=config['parameter_count'],input_dim=model.input_dim,training_seconds=time.perf_counter()-started,
                gradient_checks=gradient_checks,checkpoint_sha256=sha256(path/'best.pt'))
    write_json(path/'training_summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('../data/aliccp-paddlerec'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=6)
    p.add_argument('--min-delta',type=float,default=1e-4);p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=1e-6)
    p.add_argument('--embedding-dim',type=int,default=18);p.add_argument('--hidden',nargs='+',type=int,default=[360,200,80])
    p.add_argument('--dropout',type=float,default=.1);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if min(args.epochs,args.patience,args.batch_size,args.embedding_dim,*args.hidden)<1 or args.lr<=0 or not 0<=args.dropout<1:p.error('Invalid hyperparameters')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    source=args.output/'source';source.mkdir();source_hashes={}
    for file in Path(__file__).parent.iterdir():
        if file.suffix in ['.py','.sh','.md']:
            shutil.copy2(file,source/file.name);source_hashes[file.name]=sha256(file)
    write_json(args.output/'source_sha256.json',source_hashes)
    started=time.perf_counter();torch.set_num_threads(4)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);np.random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    raw_train=RawTable(args.data_dir/(TRAIN_NAME+'.cache'));raw_test=RawTable(args.data_dir/(TEST_NAME+'.cache'))
    for raw in [raw_train,raw_test]:
        assert sha256(args.data_dir/raw.manifest['file'])==raw.manifest['sha256']
    train_rows,val_rows=stratified_split(raw_train.labels,args.seed)
    assert not np.intersect1d(train_rows,val_rows).size
    assert len(train_rows)+len(val_rows)==len(raw_train.labels)
    np.savez(args.output/'split_indices.npz',train=train_rows,validation=val_rows)
    print('FITTING train-only vocabulary; all multi-value features retained',flush=True)
    vocabulary=fit_vocabulary(raw_train,train_rows);np.save(args.output/'vocabulary.npy',vocabulary)
    schema=dict(fields=FIELDS,field_count=len(FIELDS),vocabulary_size=len(vocabulary)+1,embedding_dim=args.embedding_dim,
                input_dim=len(FIELDS)*args.embedding_dim,pooling='sum',unknown_id=0,
                source_token_encoding='feature_id * field_count + field_index',fit_on='training partition only',max_length=None,
                vocabulary_sha256=sha256(args.output/'vocabulary.npy'))
    write_json(args.output/'feature_schema.json',schema)
    counts={'train':split_counts(raw_train,train_rows),'validation':split_counts(raw_train,val_rows),
            'test':split_counts(raw_test,np.arange(len(raw_test.labels)))}
    write_json(args.output/'data_manifest.json',dict(raw_train=raw_train.manifest,raw_test=raw_test.manifest,counts=counts,
               split_sha256=sha256(args.output/'split_indices.npz'),train_test_separation='Source files remain separate',
               validation='Fixed stratified 80/20 split of training file by (click, conversion); no timestamps present',rows_dropped=0))
    print('DATA_COUNTS '+json.dumps(counts),flush=True)
    table=DeviceTable(raw_train,vocabulary,device)
    y=raw_train.labels[train_rows];ctr_prior=float(y[:,0].mean());cvr_prior=float(y[y[:,0]==1,1].mean())
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(kind='esmm',vocabulary_size=len(vocabulary)+1,field_count=len(FIELDS),ctr_prior=ctr_prior,cvr_prior=cvr_prior,
                  source_sha256=source_hashes,feature_schema_sha256=sha256(args.output/'feature_schema.json'),
                  raw_sha256={r.manifest['file']:r.manifest['sha256'] for r in [raw_train,raw_test]},
                  torch=str(torch.__version__),device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                  started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    protocol=dict(selection='Highest validation CTCVR AUC; both checkpoints fixed before test',seed=args.seed,
                  loss='ESMM: mean BCE(click,pCTR) + mean BCE(conversion,pCTR*pCVR) over all training impressions; no direct CVR loss',
                  baseline='Independent CTR/CVR embeddings and towers, CVR trained only on clicked rows',
                  sampling='All provided rows retained; no negative downsampling or positive oversampling',
                  initialization='Identical initial tower and embedding weights; baseline clones separate CVR embedding; same row-shuffle seed',
                  limitations=['Only 41/42 conversions in supplied train/test files; validation has only 8. AUC and early stopping are noisy.',
                               'This is a preprocessed sample, not the 84-million-impression original data.',
                               'Preprocessing/sampling provenance and original feature weights cannot be reconstructed from these files.',
                               'No event timestamps; validation is stratified random, not chronological.',
                               'One seed; independent model has more embedding parameters. No statistical superiority claim.'])
    write_json(args.output/'protocol.json',protocol);write_json(args.output/'experiment_config.json',config)
    initial=copy.deepcopy(Model(config).state_dict());results={}
    for kind in ['independent','esmm']:
        results[kind]=train_one(args,dict(config,kind=kind),initial,table,raw_train,train_rows,val_rows)
    write_json(args.output/'checkpoints_fixed.json',{k:r['checkpoint_sha256'] for k,r in results.items()})
    print('CHECKPOINTS_FIXED; evaluate source test file once per model',flush=True)
    test_table=DeviceTable(raw_test,vocabulary,device);test_rows=np.arange(len(raw_test.labels))
    for kind,result in results.items():
        path=args.output/kind;checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
        model=Model(checkpoint['config']).to(device);model.load_state_dict(checkpoint['model_state'])
        scores=predict(model,test_table,test_rows,args.batch_size);metrics=funnel_metrics(raw_test.labels,scores)
        click,conversion=raw_test.labels.T
        for task,label,prob in [('ctr',click,scores[:,0]),('cvr',conversion[click==1],scores[click==1,1]),('ctcvr',conversion,scores[:,2])]:
            metrics[task]['auc_interval_approx_95']=auc_interval(label,prob)
        np.savez_compressed(path/'test_predictions.npz',scores=scores,labels=raw_test.labels,sample_ids=raw_test.sample_ids)
        result['test']=metrics;write_json(path/'summary.json',result)
        print('RESULT '+kind+' '+json.dumps(metrics),flush=True)
    baseline_scores=np.tile([ctr_prior,cvr_prior,ctr_prior*cvr_prior],(len(raw_test.labels),1))
    summary=dict(results=results,constant_train_prior_baseline=funnel_metrics(raw_test.labels,baseline_scores),counts=counts,
                 elapsed_seconds=time.perf_counter()-started,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
                 limitations=protocol['limitations'])
    write_json(args.output/'summary.json',summary)
    print(f"TRAINING_COMPLETE {args.output} elapsed={summary['elapsed_seconds']:.1f}s",flush=True)


if __name__=='__main__':main()
