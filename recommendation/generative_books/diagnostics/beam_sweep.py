"""Validation-only beam diagnostic of an immutable, selected Books checkpoint."""
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

import numpy as np
import torch


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prefix_hits(predicted, target, codes):
    valid = predicted >= 0
    match = codes[predicted.clip(min=0)] == codes[target, None]
    return {f'prefix_{depth}_hit@10': float((match[:, :10, :depth].all(-1)
            & valid[:, :10]).any(-1).mean()) for depth in range(1, 5)}


def paired_hits(baseline, predicted, target):
    before = (baseline[:, :10] == target[:, None]).any(-1)
    after = (predicted[:, :10] == target[:, None]).any(-1)
    return dict(gained=int((after & ~before).sum()), lost=int((before & ~after).sum()),
                net_hits=int(after.sum())-int(before.sum()),
                recall_delta_pp=float((after.mean()-before.mean())*100))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--beams', type=int, nargs='+', default=[20, 50, 100])
    parser.add_argument('--eval-batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.beams[0] != 20 or len(set(args.beams)) != len(args.beams) or min(args.beams) < 20:
        parser.error('Start at beam 20; unique beams >=20 required')
    if max(args.beams) > 100 or args.eval_batch_size < 1:
        parser.error('This bounded diagnostic supports beams <=100 and a positive batch size')
    parent, run = args.parent_run.resolve(), args.run_dir.resolve()
    lock = (parent.parent / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((parent / 'config.json').read_text())
    directory, source = Path(config['data_dir']), parent / 'source'
    # Use the parent's recorded source, never silently import a newer trainer.
    for name, expected in config['source_sha256'].items():
        if digest(source/name) != expected:
            raise RuntimeError(f'Parent source mismatch: {name}')
    for name, expected in config['data_signature'].items():
        if digest(directory/name) != expected:
            raise RuntimeError(f'Parent data mismatch: {name}')
    info = json.loads((directory/'dataset_info.json').read_text())
    for name, expected in info['data_sha256'].items():
        if digest(directory/(name+'.npy')) != expected:
            raise RuntimeError(f'Data array mismatch: {name}')
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run/'beam_sweep.py')
    sys.path.insert(0, str(source))
    from models import create_model
    from train import Dataset, evaluate

    started = time.monotonic()
    save(run/'status.json', dict(stage='loading', test_evaluated=False))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this bounded diagnostic')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device('cuda')
        codes = np.load(directory/'semantic_ids.npy')
        users = np.load(parent/'validation_users.npy')
        data = Dataset(directory, device)
        model = create_model('tiger', codes).to(device).eval()
        checkpoint_path = parent/'tiger/best.pt'
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        if not all(bool(torch.isfinite(t).all()) for t in model.parameters()):
            raise RuntimeError('Nonfinite selected model')
        # Warmup is not included in reported validation times.
        evaluate(model, data, 'val', users[:64], args.eval_batch_size, 20)
        report = dict(kind='validation_beam_sweep', parent_run=str(parent),
                      checkpoint_sha256=digest(checkpoint_path), parent_epoch=checkpoint['epoch'],
                      validation_users_file_sha256=digest(parent/'validation_users.npy'),
                      data_signature=config['data_signature'],
                      source_sha256=config['source_sha256'],
                      diagnostic_sha256=digest(__file__),
                      started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      users=len(users), items=len(codes), model='tiger',
                      parameter_count=sum(t.numel() for t in model.parameters()),
                      optimizer_updates=0, test_evaluated=False, results=[])
        baseline = None
        for beam in args.beams:
            save(run/'status.json', dict(stage='validation', beam_size=beam, test_evaluated=False))
            torch.cuda.reset_peak_memory_stats()
            result, predicted, target = evaluate(model, data, 'val', users, args.eval_batch_size, beam)
            if baseline is None:
                for key in ('recall@10', 'ndcg@10', 'valid_result_fraction'):
                    if abs(result[key]-checkpoint['validation'][key]) > 1e-8:
                        raise RuntimeError(f'Baseline reload mismatch: {key}')
                baseline = predicted.copy()
                print('BASELINE_RELOAD_OK', flush=True)
            result.update(prefix_hits(predicted, target, codes))
            result.update(paired_hits(baseline, predicted, target))
            result['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/1024**2
            result['batch_size'] = args.eval_batch_size
            report['results'].append(result)
            np.savez_compressed(run/f'beam_{beam}_validation.npz', users=users,
                                predicted=predicted, target=target)
            save(run/'summary.json', report)
            print('BEAM_RESULT', json.dumps(result), flush=True)
        # No cherry-picking: first five fixed validation users, with anonymous indices.
        items = json.loads((directory/'items.json').read_text())
        def item(index):
            return dict(item_index=int(index), asin=items[index]['asin'],
                        title=items[index].get('title', ''), semantic_id=codes[index].tolist())
        examples = []
        selected = np.load(run/f'beam_{args.beams[-1]}_validation.npz')
        for row, user in enumerate(users[:5]):
            begin = int(data.cpu_offsets[user])
            stop = int(data.val_position[user])
            observed = np.asarray(data.cpu_items[begin:stop])-1
            examples.append(dict(user_index=int(user), history_length=len(observed),
                                 history_last_5=[item(i) for i in observed[-5:]],
                                 validation_target=item(int(selected['target'][row])),
                                 recommendations=[item(i) for i in selected['predicted'][row, :10] if i>=0]))
        save(run/'examples.json', dict(beam_size=args.beams[-1], examples=examples,
                                     note='Fixed first 5 validation users; illustrative, not a quality sample.'))
        report.update(completed=True, elapsed_seconds=time.monotonic()-started,
                      note='One checkpoint, fixed 20000 validation users, full catalog; no training or test-set tuning. Timings are batch evaluation, not online latency. Prefix matches are diagnostic and not item recall.')
        save(run/'summary.json', report)
        save(run/'status.json', dict(stage='complete', test_evaluated=False))
        print('DIAGNOSTIC_COMPLETE', flush=True)
    except Exception as error:
        save(run/'status.json', dict(stage='failed', error=repr(error), test_evaluated=False))
        raise


if __name__ == '__main__':
    main()
