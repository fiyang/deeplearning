"""Train the author-parameter AITM port; select on validation, test once."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from model import AITM, TF1Adam
from metrics import binary_metrics
from prepare import sha256, AUTHOR_FIELDS
from raw_data import RawTable, write_json


@torch.inference_mode()
def predict(model, features, rows, output=None):
    model.eval()
    scores = np.lib.format.open_memmap(output, mode='w+', dtype='float32', shape=(len(rows), 2)) if output else np.empty((len(rows), 2), dtype=np.float32)
    tick = time.monotonic()
    for start in range(0, len(rows), 16000):
        ids = torch.as_tensor(rows[start:start + 16000], dtype=torch.long, device=features.device)
        scores[start:start + len(ids)] = model.probabilities(features[ids]).cpu().numpy()
        if time.monotonic() - tick >= 30:
            print(f'EVALUATING {start + len(ids)}/{len(rows)}', flush=True)
            tick = time.monotonic()
    if output:
        scores.flush()
    return scores


def metrics(labels, scores):
    return {'ctr': binary_metrics(labels[:, 0], scores[:, 0]),
            'ctcvr': binary_metrics(labels[:, 1], scores[:, 1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('../data/aliccp-full'))
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    assert manifest['status'] == 'passed' and manifest['fields'] == AUTHOR_FIELDS
    reference = Path(manifest['reference_run'])
    for part in ['train', 'validation']:
        assert sha256(reference / f'{part}_rows.npy') == manifest[f'{part}_rows_sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / 'source'
    shutil.copytree(Path(__file__).parent, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write_json(args.output / 'source_sha256.json', {str(f.relative_to(source)): sha256(f) for f in source.rglob('*') if f.is_file()})
    write_json(args.output / 'data_manifest.json', manifest)
    config = dict(model='AITM_author_parameters_pytorch_port', fields=AUTHOR_FIELDS,
                  vocabulary_sizes=manifest['vocabulary_sizes'], embedding_dim=5, input_dim=90,
                  hidden=[128, 64, 32], dropout=[.1, .3, .3], transfer_width=32,
                  constraint_weight=.6, l2=1e-6, l2_scope='Embedding tables and attention W1/W2/W3 only',
                  optimizer='TF1 Adam equations, dense PyTorch foreach implementation', lr=.001,
                  betas=[.9, .999], epsilon=1e-8, batch_size=2000, max_epochs=100, patience=1,
                  selection='Highest validation CTR AUC + validation CTCVR AUC', seed=args.seed,
                  official=json.loads((source / 'official/manifest.json').read_text()),
                  prepared=str(args.prepared.resolve()), reference_run=str(reference),
                  output=str(args.output.resolve()))
    protocol = dict(
        outputs=['pCTR', 'pCTCVR'], cvr='No native conditional CVR output; do not label purchase AUC as CVR AUC',
        loss='Mean click log loss + mean purchase log loss + .6*mean(relu(pPurchase-pClick)) + .5e-6*(sum embedding squared + sum attention matrices squared)',
        matches_author=['18 fields and last-value/common-overrides-sample preprocessing',
                        'Actual source-code frequency threshold >=12; trainable per-field OOV row 0',
                        '5-d embeddings; 128/64/32 towers; dropout .1/.3/.3; attention equations',
                        'Adam .001, beta1=.9, beta2=.999, epsilon=1e-8; batch 2000',
                        'Embedding/attention L2 1e-6; constraint .6; early stopping 1 on sum of AUCs'],
        differences=['PyTorch port; not the original TensorFlow 1.10 runtime and not a bitwise reproduction',
                     'Keep previous seed-42 stratified row split; author uses seed-2020 random split',
                     'Vocabulary and feature frequencies fit on actual training partition only; author fits before validation split',
                     'Vocabulary sizes derived from data with one extra OOV row, instead of hard-coded counts',
                     'Sort vocabulary IDs deterministically; shuffle all training rows each epoch instead of 100k-row buffers',
                     'Explicit dropout train/eval mode; retain final partial batch',
                     'Test is the complete original holdout, filtered for click=0/conversion=1 exactly as previous runs'],
        comparison='Multiple changes including feature subset, regularization, OOV handling, architecture and selection; not an architecture-only ablation',
        source_note='EasyRec benchmark requires four undocumented derived scores and is not used for this experiment')
    write_json(args.output / 'protocol.json', protocol)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Full-data experiment needs CUDA')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    raw = RawTable(args.data_dir / 'cache/train')
    train_rows, val_rows = [np.load(reference / f'{p}_rows.npy') for p in ['train', 'validation']]
    print('REFERENCE_SPLIT_MATCH ' + json.dumps({p: manifest[f'{p}_rows_sha256'] for p in ['train', 'validation']}), flush=True)
    features = torch.tensor(np.load(args.prepared / 'train.ids.npy', mmap_mode='r'), device='cuda', dtype=torch.int32)
    labels = torch.tensor(np.array(raw.labels), device='cuda', dtype=torch.float32)
    model = AITM(config).cuda()
    optimizer = TF1Adam(model.parameters(), lr=config['lr'])
    config.update(parameter_count=sum(p.numel() for p in model.parameters()), torch=str(torch.__version__), gpu=torch.cuda.get_device_name())
    write_json(args.output / 'config.json', config)
    print('CONFIG ' + json.dumps({k: config[k] for k in ['model','input_dim','embedding_dim','hidden','batch_size','lr','parameter_count']}), flush=True)
    rng = np.random.default_rng(args.seed + 1000)
    best, best_epoch, stale, history = -1., 0, 0, []
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        order = rng.permutation(train_rows)
        tick = update = time.monotonic()
        total, seen = torch.zeros((), device='cuda'), 0
        for step, start in enumerate(range(0, len(order), config['batch_size']), 1):
            rows = torch.as_tensor(order[start:start + config['batch_size']], device='cuda', dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            parts = model.loss_parts(model(features[rows]), labels[rows])
            loss = sum(parts.values())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if epoch == 1 and step == 1:
                checked = {name: any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in getattr(model, name).parameters())
                           for name in ['embeddings','click_tower','purchase_tower','transfer','attention','click_head','purchase_head']}
                assert all(checked.values()), checked
                write_json(args.output / 'gradient_check.json', checked)
            optimizer.step()
            total += loss.detach() * len(rows)
            seen += len(rows)
            if step == 1 or time.monotonic() - update >= 30:
                print(f'EPOCH {epoch}/{config["max_epochs"]} STEP {step} rows={seen}/{len(order)} loss={total.item()/seen:.6f} examples_per_s={seen/(time.monotonic()-tick):.0f} gpu_gib={torch.cuda.max_memory_allocated()/1024**3:.2f}', flush=True)
                update = time.monotonic()
            if args.benchmark_only and step >= 30:
                print('BENCHMARK_COMPLETE', flush=True)
                return
        scores = predict(model, features, val_rows)
        val = metrics(raw.labels[val_rows], scores)
        selection = val['ctr']['auc'] + val['ctcvr']['auc']
        row = dict(epoch=epoch, train_loss=total.item()/seen, val_ctr_auc=val['ctr']['auc'],
                   val_ctcvr_auc=val['ctcvr']['auc'], val_ctcvr_ap=val['ctcvr']['average_precision'],
                   val_ctcvr_log_loss=val['ctcvr']['log_loss'], selection=selection, seconds=time.monotonic()-tick)
        history.append(row)
        if selection > best:
            best, best_epoch, stale = selection, epoch, 0
            temporary = args.output / 'best.pt.tmp'
            torch.save(dict(config=config, epoch=epoch, model_state={k: v.detach().cpu() for k,v in model.state_dict().items()}), temporary)
            temporary.replace(args.output / 'best.pt')
            write_json(args.output / 'best_validation.json', val)
            np.save(args.output / 'validation_scores.npy', scores)
        else:
            stale += 1
        with (args.output / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row)
            writer.writeheader(); writer.writerows(history)
        print(f'EPOCH_COMPLETE {epoch} CTR_AUC={val["ctr"]["auc"]:.6f} CTCVR_AUC={val["ctcvr"]["auc"]:.6f} BEST_SUM={best:.6f}@{best_epoch} seconds={row["seconds"]:.1f}', flush=True)
        if stale >= config['patience']:
            print('EARLY_STOPPING', flush=True)
            break
    write_json(args.output / 'checkpoint_fixed.json', dict(best_epoch=best_epoch, sha256=sha256(args.output / 'best.pt')))
    print('CHECKPOINT_FIXED; evaluating original test once', flush=True)
    del optimizer, model, features, labels, raw, scores
    gc.collect(); torch.cuda.empty_cache()
    raw = RawTable(args.data_dir / 'cache/test')
    test_rows = np.load(args.prepared / 'test_rows.npy')
    assert sha256(args.prepared / 'test_rows.npy') == manifest['test_rows_sha256']
    features = torch.tensor(np.load(args.prepared / 'test.ids.npy', mmap_mode='r'), device='cuda', dtype=torch.int32)
    checkpoint = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    model = AITM(checkpoint['config']).cuda(); model.load_state_dict(checkpoint['model_state']); del checkpoint
    scores = predict(model, features, test_rows, args.output / 'test_scores.npy')
    test_metrics = metrics(raw.labels[test_rows], scores)
    write_json(args.output / 'test_scores_schema.json', dict(columns=['pCTR', 'pCTCVR'], row_indices=str(args.prepared.resolve() / 'test_rows.npy')))
    summary = dict(model=config['model'], best_epoch=best_epoch, completed_epochs=len(history),
                   validation=json.loads((args.output / 'best_validation.json').read_text()), test=test_metrics,
                   counts={k: manifest[k] for k in ['train','validation','test']},
                   parameter_count=config['parameter_count'], peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                   elapsed_seconds=time.monotonic()-started, cvr='Not directly modeled; purchase output is CTCVR')
    write_json(args.output / 'summary.json', summary)
    print('RESULT ' + json.dumps(summary), flush=True)
    print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
