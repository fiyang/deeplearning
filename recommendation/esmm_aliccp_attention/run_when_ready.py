"""Run this experiment after its exact baseline finishes and releases the GPU."""
import argparse
import fcntl
import json

from pathlib import Path
import subprocess
import sys
import time


def matching_train_processes(reference):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if args and 'python' in Path(args[0]).name and '--output' in args:
                if args[args.index('--output')+1] == str(reference):
                    found.append(int(entry.name))
        except (OSError, UnicodeError, IndexError):
            continue
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--poll-seconds', type=float, default=30)
    p.add_argument('--wait-hours', type=float, default=16)
    args = p.parse_args()
    reference = args.reference_run.resolve()
    output = args.output.resolve()
    status_file = output.with_suffix('.queue.json')
    def status(state, **details):
        payload = dict(status=state, output=str(output), reference_run=str(reference),
                       updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **details)
        temp = status_file.with_suffix('.json.tmp')
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n')
        temp.replace(status_file)
        print('QUEUE '+json.dumps(payload, ensure_ascii=False), flush=True)
    lock = (output.parent/'.esmm-field-attention.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        status('failed', reason='Another field-attention experiment is queued or running')
        return 1
    try:
        if output.exists():
            raise RuntimeError('Experiment output already exists')
        deadline = time.monotonic()+args.wait_hours*3600
        last_message = 0
        while matching_train_processes(reference):
            if time.monotonic()>deadline:
                raise RuntimeError('Timed out waiting for baseline')
            if time.monotonic()-last_message>=300:
                status('waiting_for_baseline', baseline_pids=matching_train_processes(reference))
                last_message = time.monotonic()
            time.sleep(args.poll_seconds)
        baseline_log = reference.with_suffix('.log')
        with baseline_log.open('rb') as f:
            f.seek(max(0,baseline_log.stat().st_size-1024))
            complete = b'TRAINING_COMPLETE' in f.read()
        summary = json.loads((reference/'summary.json').read_text())
        if not complete or summary.get('best_epoch',0)<1 or not (reference/'checkpoint_fixed.json').is_file():
            raise RuntimeError('Baseline did not finish successfully; refusing to start')
        reference_config = json.loads((reference/'config.json').read_text())
        if reference_config['lr'] != .0001:
            raise RuntimeError('This attention experiment requires the 0.0001 baseline')
        last_message = 0
        while True:
            result = subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                                    check=True,capture_output=True,text=True)
            memory_mib = [int(line) for line in result.stdout.splitlines()]
            if len(memory_mib)!=1:
                raise RuntimeError('This launcher expects the single-GPU AutoDL environment')
            if memory_mib[0]<2048:
                break
            if time.monotonic()>deadline:
                raise RuntimeError('Timed out waiting for free GPU memory')
            if time.monotonic()-last_message>=300:
                status('waiting_for_gpu',memory_used_mib=memory_mib[0])
                last_message = time.monotonic()
            time.sleep(args.poll_seconds)
        command = [sys.executable,'-u',str(Path(__file__).parent/'train.py'),
                   '--data-dir',str(args.data_dir.resolve()),'--output',str(output),
                   '--embedding-dim','18','--id-embedding-dim','64','--lr','0.0001',
                   '--epochs','10','--patience','3','--batch-size','4096',
                   '--attention-dim','64','--attention-heads','4','--attention-dropout','0.1',
                   '--reference-run',str(reference)]
        status('starting',command=command,baseline_best_epoch=summary['best_epoch'])
        with subprocess.Popen(command) as child:
            status('training',pid=child.pid)
            returncode = child.wait()
        if returncode or not (output/'summary.json').is_file():
            raise RuntimeError(f'Training did not complete; exit code {returncode}')
        status('complete')
        return 0
    except Exception as error:
        status('failed',reason=str(error))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    raise SystemExit(main())
