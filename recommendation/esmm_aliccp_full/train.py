"""Full original Ali-CCP ESMM: all exposures, weighted sparse features, held-out test."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from data import RawTable,DeviceTable,FIELDS,split_rows,fit_vocabulary,lookup_from_vocabulary,remap,write_json,counts
from model import Model
from metrics import funnel_metrics


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def validate_comparison(current,reference,variable):
    """Allow only the named experimental variable to differ from a baseline."""
    if variable not in ['id_embedding_dim','lr']:raise ValueError('Unsupported comparison variable')
    keys=['epochs','patience','batch_size','lr','embedding_dim','id_embedding_dim','hidden','dropout','seed','min_delta','validation_fraction']
    previous=dict(reference)
    previous.setdefault('id_embedding_dim',previous['embedding_dim'])
    for key in keys:
        if key!=variable and current[key]!=previous[key]:raise ValueError(f'Controlled comparison changed {key}')
    return dict(variable=variable,baseline_value=previous[variable],experiment_value=current[variable],
                fixed_hyperparameters={key:current[key] for key in keys if key!=variable})

@torch.inference_mode()
def predict(model,table,rows,batch_size,output=None):
    model.eval();scores=np.lib.format.open_memmap(output,mode='w+',dtype='float32',shape=(len(rows),3)) if output else np.empty((len(rows),3),dtype=np.float32)
    tick=time.monotonic()
    for start in range(0,len(rows),batch_size):
        end=min(start+batch_size,len(rows));scores[start:end]=model.probabilities(*table.batch(rows[start:end])).cpu().numpy()
        if time.monotonic()-tick>60:print(f'EVALUATING {end}/{len(rows)}',flush=True);tick=time.monotonic()
    if output:scores.flush()
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('../data/aliccp-full'))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--patience',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=4096);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--embedding-dim',type=int,default=18);p.add_argument('--hidden',type=int,nargs='+',default=[360,200,80])
    p.add_argument('--id-embedding-dim',type=int,help='Width of fields 101 and 205; other fields keep --embedding-dim')
    p.add_argument('--reference-run',type=Path,help='Require matched hyperparameters, split and vocabulary against this baseline')
    p.add_argument('--comparison-change',choices=['id_embedding_dim','lr'],default='id_embedding_dim',help='Only this hyperparameter may differ from --reference-run')
    p.add_argument('--dropout',type=float,default=.1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-delta',type=float,default=.0001);p.add_argument('--validation-fraction',type=float,default=.1)
    p.add_argument('--benchmark-only',action='store_true')
    args=p.parse_args()
    if args.id_embedding_dim is None:args.id_embedding_dim=args.embedding_dim
    if not 0<args.validation_fraction<1 or not np.isfinite(args.lr) or args.lr<=0 or min(args.epochs,args.patience,args.batch_size,args.embedding_dim,args.id_embedding_dim,*args.hidden)<1:p.error('invalid configuration')
    if args.comparison_change=='lr' and not args.reference_run:p.error('Learning-rate comparison requires --reference-run')
    reference=None;comparison=None
    if args.reference_run:
        args.reference_run=args.reference_run.resolve();reference=json.loads((args.reference_run/'config.json').read_text())
        try:comparison=validate_comparison(vars(args),reference,args.comparison_change)
        except ValueError as error:p.error(str(error))
        comparison['reference_run']=str(args.reference_run)
        if args.comparison_change=='lr':
            comparison['unchanged_source_sha256']={}
            for name in ['model.py','data.py','metrics.py']:
                current_hash=sha256(Path(__file__).parent/name)
                if current_hash!=sha256(args.reference_run/'source'/name):p.error(f'Learning-rate comparison changed source: {name}')
                comparison['unchanged_source_sha256'][name]=current_hash
    args.output.mkdir(parents=True,exist_ok=False);source=args.output/'source';source.mkdir()
    for f in Path(__file__).parent.iterdir():
        if f.suffix in ['.py','.cpp','.json','.sh','.md']:shutil.copy2(f,source/f.name)
    write_json(args.output/'source_sha256.json',{f.name:sha256(f) for f in source.iterdir()})
    started=time.monotonic();torch.set_num_threads(8);torch.manual_seed(args.seed);np.random.seed(args.seed)
    if not torch.cuda.is_available():raise RuntimeError('Full experiment requires CUDA')
    device=torch.device('cuda');torch.cuda.manual_seed_all(args.seed);torch.backends.cuda.matmul.allow_tf32=False
    raw=RawTable(args.data_dir/'cache/train');train_rows,val_rows=split_rows(raw.labels,args.seed,args.validation_fraction)
    write_json(args.output/'data_manifest.json',dict(train_cache=raw.info,
        train_source=json.loads((args.data_dir/'raw/train/extraction.json').read_text()),
        row_order='Indices refer to zero-based lines of the original sample_skeleton CSV'))
    np.save(args.output/'train_rows.npy',train_rows);np.save(args.output/'validation_rows.npy',val_rows)
    split_counts={'train':counts(raw.labels,train_rows),'validation':counts(raw.labels,val_rows)}
    excluded=np.flatnonzero((raw.labels[:,0]==0)&(raw.labels[:,1]==1)).astype(np.int32)
    np.save(args.output/'excluded_train_rows.npy',excluded)
    split_counts['source_train_rows']=raw.n;split_counts['excluded_train_inconsistent_labels']=len(excluded)
    write_json(args.output/'counts.json',split_counts);print('DATA_COUNTS '+json.dumps(split_counts),flush=True)
    print('FITTING_TRAIN_ONLY_VOCABULARY',flush=True);vocabulary,lookup=fit_vocabulary(raw,train_rows,args.output)
    if reference:
        comparison['matched_artifact_sha256']={}
        for name in ['train_rows.npy','validation_rows.npy','vocabulary.npy']:
            if sha256(args.output/name)!=sha256(args.reference_run/name):raise ValueError(f'Reference mismatch: {name}')
            comparison['matched_artifact_sha256'][name]=sha256(args.output/name)
        write_json(args.output/'reference_comparison.json',comparison)
        print('REFERENCE_SPLIT_AND_VOCABULARY_MATCH',flush=True)
        print('CONTROLLED_CHANGE '+json.dumps({key:comparison[key] for key in ['variable','baseline_value','experiment_value']}),flush=True)
    mapping=remap(raw,lookup,args.output/'mapped_train');del lookup;gc.collect()
    y=raw.labels[train_rows];ctr_prior=float(y[:,0].mean());cvr_prior=float(y[y[:,0]==1,1].mean());del y
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    id_field_indices=[FIELDS.index(f) for f in ['101','205']]
    id_count=int(np.isin(vocabulary%len(FIELDS),id_field_indices).sum())
    field_dims=[args.id_embedding_dim if i in id_field_indices else args.embedding_dim for i in range(len(FIELDS))]
    config.update(kind='esmm',field_count=len(FIELDS),fields=FIELDS,vocabulary_size=len(vocabulary)+1,input_dim=sum(field_dims),
                  field_embedding_dims=dict(zip(FIELDS,field_dims)),id_vocabulary_size=id_count+1,base_vocabulary_size=len(vocabulary)-id_count+1,
                  id_mapping='Collision-free train-only vocabulary; no hash buckets; missing/unseen IDs map to zero padding',
                  ctr_prior=ctr_prior,cvr_prior=cvr_prior,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                  pooling='weighted sum, all original tokens retained',optimizer='Adam, no weight decay',
                  split='Stratified random validation from original training file only; original test remains separate',
                  selection='Highest validation CTCVR AUC; evaluate original test once after checkpoint selection',
                  train_rows_sha256=sha256(args.output/'train_rows.npy'),validation_rows_sha256=sha256(args.output/'validation_rows.npy'))
    write_json(args.output/'config.json',config)
    write_json(args.output/'protocol.json',dict(loss='BCE(click,pCTR)+BCE(conversion,pCTR*pCVR) over all training exposures',
        labels='Exclude click=0/conversion=1 from both training and evaluation; preserve raw files and record excluded line indices',sampling='All funnel-consistent exposures retained; no class weighting',
        label_filter_reference='https://github.com/xidongbo/AITM/blob/main/process_public_dataset.py',
        features='Common rows referenced by training examples and sample features of training rows only build the vocabulary; other tokens map to padding/unknown 0',
        limitations=['One seed. Validation is random because no event timestamps are present in these files.','Row split may include the same user in both partitions; it measures known-user generalization, not a cold-start experiment.','Sparse conversion rate makes raw accuracy misleading; report AUC, average precision and log loss.','No superiority claim over other models without matched full-data baselines.']))
    print('LOADING_GPU_TABLE',flush=True);table=DeviceTable(raw,args.output/'mapped_train',device)
    print(f'GPU_TABLE_GIB {torch.cuda.memory_allocated()/1024**3:.2f}',flush=True)
    model=Model(config,vocabulary=vocabulary).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,foreach=True)
    print(f'TRAINING_CONFIG lr={optimizer.param_groups[0]["lr"]:.7f} batch_size={args.batch_size} scheduler=none weight_decay=0',flush=True)
    assert model.input_dim==config['input_dim']
    config['parameter_count']=sum(x.numel() for x in model.parameters());write_json(args.output/'config.json',config)
    print(f'EMBEDDING_WIDTHS user={args.id_embedding_dim} item={args.id_embedding_dim} other={args.embedding_dim} input={model.input_dim} parameters={config["parameter_count"]}',flush=True)
    best=-1.;patience_best=-1.;stale=0;history=[];rng=np.random.default_rng(args.seed+1000);best_epoch=0
    for epoch in range(1,args.epochs+1):
        tick=time.monotonic();update=tick;model.train();order=rng.permutation(train_rows)
        total=torch.zeros((),device=device);seen=0
        for step,start in enumerate(range(0,len(order),args.batch_size),1):
            idx=order[start:start+args.batch_size];optimizer.zero_grad(set_to_none=True)
            loss=model.loss(model(*table.batch(idx)),table.labels[idx])
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward()
            if epoch==1 and step==1:
                check={name:any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in getattr(model,name).parameters()) for name in ['embedding','ctr','cvr']}
                if not all(check.values()):raise RuntimeError('missing gradient branch')
                write_json(args.output/'gradient_check.json',check)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True,foreach=True)
            optimizer.step();total+=loss.detach()*len(idx);seen+=len(idx)
            if step==1 or time.monotonic()-update>=30:
                elapsed=time.monotonic()-tick
                print(f'EPOCH {epoch}/{args.epochs} STEP {step} rows={seen}/{len(order)} loss={total.item()/seen:.6f} examples_per_s={seen/elapsed:.0f} gpu_gib={torch.cuda.max_memory_allocated()/1024**3:.2f}',flush=True);update=time.monotonic()
            if args.benchmark_only and step==30:
                torch.cuda.synchronize();write_json(args.output/'benchmark.json',dict(rows=seen,seconds=time.monotonic()-tick,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3));print('BENCHMARK_COMPLETE',flush=True);return
        scores=predict(model,table,val_rows,args.batch_size);val=funnel_metrics(raw.labels[val_rows],scores);metric=val['ctcvr']['auc']
        row=dict(epoch=epoch,train_loss=total.item()/seen,lr=optimizer.param_groups[0]['lr'],val_ctr_auc=val['ctr']['auc'],val_cvr_auc=val['cvr']['auc'],val_ctcvr_auc=metric,val_ctcvr_ap=val['ctcvr']['average_precision'],val_ctcvr_log_loss=val['ctcvr']['log_loss'],seconds=time.monotonic()-tick)
        history.append(row)
        if metric>best:
            best=metric;best_epoch=epoch
            temp=args.output/'best.pt.tmp';torch.save(dict(config=config,epoch=epoch,validation=val,model_state={k:v.detach().cpu() for k,v in model.state_dict().items()}),temp);temp.replace(args.output/'best.pt')
            np.save(args.output/'validation_scores.npy',scores);write_json(args.output/'best_validation.json',val)
        if metric>patience_best+args.min_delta:patience_best=metric;stale=0
        else:stale+=1
        with (args.output/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerows(history)
        print(f'EPOCH_COMPLETE {epoch}/{args.epochs} CTR_AUC={val["ctr"]["auc"]:.6f} CVR_AUC={val["cvr"]["auc"]:.6f} CTCVR_AUC={metric:.6f} BEST={best:.6f}@{best_epoch} seconds={row["seconds"]:.1f}',flush=True)
        if stale>=args.patience:print('EARLY_STOPPING',flush=True);break
    write_json(args.output/'checkpoint_fixed.json',dict(sha256=sha256(args.output/'best.pt'),best_epoch=best_epoch))
    print('CHECKPOINT_FIXED; original test evaluated once',flush=True)
    del optimizer,model,table,raw,scores;gc.collect();torch.cuda.empty_cache()
    raw=RawTable(args.data_dir/'cache/test');lookup=lookup_from_vocabulary(vocabulary);remap(raw,lookup,args.output/'mapped_test');del lookup;gc.collect()
    test_rows=np.flatnonzero(~((raw.labels[:,0]==0)&(raw.labels[:,1]==1))).astype(np.int32)
    np.save(args.output/'test_rows.npy',test_rows)
    excluded_test=np.flatnonzero((raw.labels[:,0]==0)&(raw.labels[:,1]==1)).astype(np.int32)
    np.save(args.output/'excluded_test_rows.npy',excluded_test)
    provenance=json.loads((args.output/'data_manifest.json').read_text());provenance.update(test_cache=raw.info,test_source=json.loads((args.data_dir/'raw/test/extraction.json').read_text()));write_json(args.output/'data_manifest.json',provenance)
    table=DeviceTable(raw,args.output/'mapped_test',device)
    checkpoint=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
    model=Model(checkpoint['config']).to(device);model.load_state_dict(checkpoint['model_state']);del checkpoint
    scores=predict(model,table,test_rows,args.batch_size,args.output/'test_scores.npy')
    test_labels=raw.labels[test_rows]
    metrics=funnel_metrics(test_labels,scores);split_counts['test']=counts(test_labels)
    split_counts['source_test_rows']=raw.n;split_counts['excluded_test_inconsistent_labels']=len(excluded_test)
    constant=np.broadcast_to(np.array([ctr_prior,cvr_prior,ctr_prior*cvr_prior],dtype=np.float32),(len(test_rows),3))
    summary=dict(best_epoch=best_epoch,completed_epochs=len(history),validation=json.loads((args.output/'best_validation.json').read_text()),test=metrics,
                 counts=split_counts,constant_train_prior_baseline=funnel_metrics(test_labels,constant),
                 parameter_count=config['parameter_count'],peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,elapsed_seconds=time.monotonic()-started)
    write_json(args.output/'summary.json',summary);print('RESULT '+json.dumps(summary),flush=True);print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
