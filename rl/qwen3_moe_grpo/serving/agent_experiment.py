"""Bounded remote smoke and A/B runner. Parent shell owns shutdown on all exits."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from candidate_cases import CASES
from candidate_demo import preflight, run_case
from candidate_prompts import PromptBundle
from recruiting_demo import chat
from agent_chat import post


def emit(**row):
    print(json.dumps(row, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prompt-dir', type=Path, default=Path('../runs/candidate-prompt-snapshot-20261024'))
    parser.add_argument('--model-path', default='/root/autodl-tmp/qwen3-30b-a3b-bf16')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(exist_ok=True)
    summary = dict(completed=False, optimizer_updates=0, ab_results=[], dialogues=[], started=time.time())
    children, streams = [], []
    def checkpoint():
        (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    def terminated(*_):
        raise RuntimeError('Experiment deadline reached')
    signal.signal(signal.SIGTERM, terminated)
    key = secrets.token_urlsafe(32)
    descriptor = os.open(run / 'api_key', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as output:
        output.write(key)
    env = dict(os.environ, QWEN3_API_KEY=key)
    try:
        bundle = PromptBundle(args.prompt_dir)
        emit(stage='token_preflight', prompt_sha256=bundle.sha256)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
        report = preflight(bundle, tokenizer)
        (run / 'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        peak = max(r['with_both_skills_tokens'] + 4096 for r in report['cases'])
        emit(stage='token_preflight_complete', fits=report['fits'], required_with_reserve=peak)
        if not report['fits']:
            raise RuntimeError('Verbatim business prompts exceed context limit')
        max_context = 16384 if peak <= 16384 else 32768
        summary.update(context=max_context, preflight_required_tokens=peak, prompt_sha256=bundle.sha256)
        del tokenizer
        def launch(name, command):
            log = (run / (name + '.log')).open('w')
            streams.append(log)
            child = subprocess.Popen([sys.executable, *command], env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            return child
        server = launch('model', ['serve.py', '--model-path', args.model_path, '--max-context', str(max_context)])
        ready_by = min(time.time() + 600, float(env['QWEN3_INSTANCE_STARTED_AT']) + 3300)
        while time.time() < ready_by:
            if server.poll() is not None:
                raise RuntimeError('Model server exited: see model.log')
            try:
                with urlopen(Request('http://127.0.0.1:8000/health', headers={'Authorization':'Bearer ' + key}), timeout=3) as response:
                    health = json.load(response)
                break
            except OSError:
                time.sleep(3)
        else:
            raise RuntimeError('Model did not become ready before deadline')
        summary['model_health'] = health
        emit(stage='model_ready', max_context=max_context)
        agent = launch('agent', ['agent_server.py', '--prompt-dir', str(args.prompt_dir.resolve()),
                                '--trace-dir', str(run / 'agent-traces')])
        for _ in range(20):
            if agent.poll() is not None:
                raise RuntimeError('Agent server exited: see agent.log')
            try:
                with urlopen(Request('http://127.0.0.1:8001/health', headers={'Authorization':'Bearer ' + key}), timeout=2) as response:
                    summary['agent_health'] = json.load(response)
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError('Agent did not become ready')
        emit(stage='agent_ready', run_dir=str(run))
        for model in ['qwen3-base', 'qwen3-grpo']:
            created = post('http://127.0.0.1:8001', key, '/sessions', dict(model=model))
            if 'session_id' not in created:
                raise RuntimeError('Cannot create Agent session')
            for text in ['怎么称呼你？', '我35岁，在济南，想找保洁。', '这个工资是到手的吗？']:
                result = post('http://127.0.0.1:8001', key, '/sessions/' + created['session_id'] + '/messages', dict(text=text))
                summary['dialogues'].append(dict(model=model, user=text, **result))
                checkpoint()
                emit(stage='dialogue', model=model, user=text, **result)
                if not result.get('completed'):
                    break
        # Run representative cases first; no new request in the final 3 minutes.
        order = ['current_salary', 'consent_success', 'consent_failure', 'industry_switch']
        cases = sorted(CASES, key=lambda c: order.index(c['id']) if c['id'] in order else len(order))
        with (run / 'ab.jsonl').open('x') as output:
            for index, case in enumerate(cases):
                for model in (['qwen3-base', 'qwen3-grpo'] if index % 2 == 0 else ['qwen3-grpo', 'qwen3-base']):
                    if time.time() >= float(env['QWEN3_INSTANCE_STARTED_AT']) + 3420:
                        raise RuntimeError('Stopped new work before service deadline; partial results saved')
                    result = run_case(case, model, lambda body: chat('http://127.0.0.1:8000/v1', key, body), bundle)
                    output.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output.flush()
                    compact = {k:v for k,v in result.items() if k not in ('trace', 'history', 'final_context', 'prompt')}
                    summary['ab_results'].append(compact)
                    checkpoint()
                    emit(stage='ab_case', **compact)
                    if 'HTTP Error 500' in result.get('error', ''):
                        raise RuntimeError('Model service failed; stop and inspect model.log')
        summary['completed'] = True
    except Exception as error:
        summary['error'] = str(error)
        emit(stage='failure', error=str(error))
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for log in streams:
            log.close()
        summary['finished'] = time.time()
        summary['elapsed_seconds'] = summary['finished'] - summary['started']
        (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        emit(stage='experiment_finished', completed=summary['completed'], ab_cases=len(summary['ab_results']),
             dialogue_turns=len(summary['dialogues']), elapsed_seconds=summary['elapsed_seconds'], summary_file=str(run/'summary.json'))


if __name__ == '__main__':
    main()
