"""Read-only Top-K diagnostic on a frozen Books checkpoint and fixed validation users."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from zoneinfo import ZoneInfo


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def rank_report(predicted, targets):
    """Count exact-item hits in nested prefixes of ONE shared Top-100 list."""
    if not len(targets) or len(predicted) != len(targets):
        raise ValueError('Nonempty, aligned predictions and targets required')
    buckets = {'1-10': 0, '11-50': 0, '51-100': 0, 'not_in_top100': 0}
    valid_count = 0
    ranks = []
    for row, target in zip(predicted, targets):
        if len(row) < 100 or target < 0:
            raise ValueError('Need 100 stored positions and a nonnegative target')
        row = [int(item) for item in row[:100]]
        valid = [item for item in row if item >= 0]
        if len(valid) != len(set(valid)):
            raise ValueError('Duplicate items in recommendation list')
        valid_count += len(valid)
        rank = row.index(int(target)) + 1 if int(target) in row else 0
        ranks.append(rank)
        key = 'not_in_top100' if not rank else ('1-10' if rank <= 10 else '11-50' if rank <= 50 else '51-100')
        buckets[key] += 1
    n = len(targets)
    hits = {str(k): sum(0 < rank <= k for rank in ranks) for k in (10, 50, 100)}
    return {'requests': n, 'hits': hits,
            **{f'recall@{k}': hits[str(k)] / n for k in (10, 50, 100)},
            'rank_buckets': buckets,
            'top100_valid_fraction': valid_count / (n * 100),
            'recovered_after_top10': buckets['11-50'] + buckets['51-100']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    model_run, run = args.model_run.resolve(), args.run_dir.resolve()
    # This diagnostic may coexist with training: only its own immutable copy is read.
    # It does not acquire/change the training lock, optimizer, status, or best pointer.
    lock = (model_run.parent / 'recall_curve.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((model_run / 'config.json').read_text())
    refinement = 'inherited_source_sha256' in config
    source = model_run / ('inherited_source' if refinement else 'source')
    source_hashes = config['inherited_source_sha256' if refinement else 'source_sha256']
    checkpoint_path = model_run / ('best.pt' if refinement else 'tiger/best.pt')
    directory = Path(config['data_dir'])
    for name, expected in source_hashes.items():
        if digest(source / name) != expected:
            raise RuntimeError(f'Source mismatch: {name}')
    for name, expected in config['data_signature'].items():
        if digest(directory / name) != expected:
            raise RuntimeError(f'Data mismatch: {name}')
    info = json.loads((directory / 'dataset_info.json').read_text())
    for name, expected in info['data_sha256'].items():
        if digest(directory / (name + '.npy')) != expected:
            raise RuntimeError(f'Data array mismatch: {name}')
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / 'recall_curve.py')
    # The trainer atomically replaces best.pt; opening/copying it pins one generation.
    shutil.copyfile(checkpoint_path, run / 'checkpoint_snapshot.pt')
    shutil.copy2(model_run / 'validation_users.npy', run / 'validation_users.npy')
    shutil.copytree(source, run / 'model_source')
    sys.path.insert(0, str(run / 'model_source'))
    started = time.monotonic()
    save(run / 'status.json', {'stage': 'loading', 'test_evaluated': False})
    try:
        import numpy as np
        import torch
        import train
        from models import create_model
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        free, _ = torch.cuda.mem_get_info()
        if free < 5 * 1024**3:
            raise RuntimeError('Less than 5 GiB GPU memory free; leave active training untouched')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device('cuda')
        codes = np.load(directory / 'semantic_ids.npy')
        users = np.load(run / 'validation_users.npy')
        if len(users) != 20000 or len(np.unique(users)) != len(users):
            raise RuntimeError('Expected original 20,000 unique validation users')
        if refinement and digest(run / 'validation_users.npy') != config['validation_users_sha256']:
            raise RuntimeError('Validation users changed')
        checkpoint = torch.load(run / 'checkpoint_snapshot.pt', map_location=device, weights_only=False)
        model = create_model('tiger', codes).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        if not all(bool(torch.isfinite(t).all()) for t in model.parameters()):
            raise RuntimeError('Nonfinite checkpoint')
        data = train.Dataset(directory, device)
        batch_size = config.get('eval_batch_size', 64)
        train.evaluate(model, data, 'val', users[:64], batch_size, 20)
        save(run / 'status.json', {'stage': 'reproduce_baseline', 'epoch': checkpoint['epoch']})
        baseline, before, target = train.evaluate(model, data, 'val', users, batch_size, 20)
        for key in ('recall@10', 'ndcg@10', 'valid_result_fraction'):
            if abs(baseline[key] - checkpoint['validation'][key]) > 1e-8:
                raise RuntimeError(f'Baseline reload mismatch: {key}')
        print('BASELINE_RELOAD_OK', json.dumps(baseline), flush=True)
        # Change only this process's reporting cutoffs, retaining all 100 predictions.
        train.KS = (5, 10, 20, 50, 100)
        torch.cuda.reset_peak_memory_stats()
        save(run / 'status.json', {'stage': 'top100_validation', 'epoch': checkpoint['epoch']})
        metrics, predicted, after_target = train.evaluate(model, data, 'val', users, batch_size, 100)
        if not np.array_equal(target, after_target) or predicted.shape != (len(users), 100):
            raise RuntimeError('Target alignment or candidate width mismatch')
        curve = rank_report(predicted, target)
        for k in (10, 50, 100):
            if abs(curve[f'recall@{k}'] - metrics[f'recall@{k}']) > 1e-12:
                raise RuntimeError('Independent rank count disagrees with evaluator')
        before_hit = (before[:, :10] == target[:, None]).any(1)
        after_hit = (predicted[:, :10] == target[:, None]).any(1)
        np.savez_compressed(run / 'top100_validation.npz', users=users, predicted=predicted,
                            target=target, baseline_beam20_top20=before)
        report = {
            'completed': True, 'kind': 'validation_recall_curve', 'model_run': str(model_run),
            'checkpoint_epoch': checkpoint['epoch'],
            'checkpoint_sha256': digest(run / 'checkpoint_snapshot.pt'),
            'validation_users_file_sha256': digest(run / 'validation_users.npy'),
            'source_sha256': source_hashes, 'data_signature': config['data_signature'],
            'diagnostic_sha256': digest(__file__), 'optimizer_updates': 0, 'test_evaluated': False,
            'items': len(codes), 'baseline_beam20': baseline, 'beam100_metrics': metrics,
            'curve': curve,
            'search_change_at10': {'gained': int((after_hit & ~before_hit).sum()),
                                   'lost': int((before_hit & ~after_hit).sum()),
                                   'delta_pp': float((after_hit.mean() - before_hit.mean()) * 100)},
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
            'elapsed_seconds': time.monotonic() - started,
            'completed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'note': 'Same frozen checkpoint and validation users. All curve cutoffs are prefixes of one beam=100 list. Full catalog, all previously seen items filtered. No test-set tuning. Training may run concurrently; timings are not isolated inference benchmarks. A miss is outside the generated Top-100, not proof of exact model rank >100.'}
        save(run / 'summary.json', report)
        save(run / 'status.json', {'stage': 'complete', 'test_evaluated': False})
        print('RECALL_CURVE_COMPLETE', json.dumps(report), flush=True)
    except Exception as error:
        save(run / 'status.json', {'stage': 'failed', 'error': repr(error), 'test_evaluated': False})
        raise


if __name__ == '__main__':
    main()
