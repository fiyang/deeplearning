"""CPU-only experiment bookkeeping. Importing this module starts no jobs."""
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from rl.qwen7b_grpo.rewards import correctness_reward, extract_answer, number_value, score


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def event(run, stage, **values):
    record = dict(stage=stage, timestamp=time.time(), **values)
    write_json(Path(run) / 'status.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def load_config(path=ROOT / 'config.json'):
    config = json.loads(Path(path).read_text())
    if config['precision'] != 'bf16' or config['enable_thinking'] is not False:
        raise ValueError('This experiment requires BF16 and non-thinking mode')
    if config['num_generations'] < 2:
        raise ValueError('GRPO requires at least two completions per question')
    for key in ['generation_batch_size']:
        if config[key] % config['num_generations']:
            raise ValueError(f'{key} must be divisible by num_generations')
    batch = config['per_device_train_batch_size'] * config['gradient_accumulation_steps']
    if batch % config['generation_batch_size']:
        raise ValueError('Generation batches must divide each optimizer batch')
    if config['train_samples'] * config['num_generations'] % batch:
        raise ValueError('Use complete optimizer batches for a one-epoch comparison')
    return config


def expected_steps(config):
    return (config['train_samples'] * config['num_generations'] * config['epochs'] //
            (config['per_device_train_batch_size'] * config['gradient_accumulation_steps']))


def select_rows(tokenizer, config, data_dir, run):
    selected, questions = {}, set()
    for split, count in [('test', config['test_samples']), ('train', config['train_samples'])]:
        path = Path(data_dir) / f'gsm8k_{split}.jsonl'
        source = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        indices = list(range(len(source)))
        random.Random(config['seed'] + (split == 'test')).shuffle(indices)
        rows = []
        for index in indices:
            item = source[index]
            if item['question'] in questions:
                continue
            answer = item['answer'].rsplit('####', 1)[-1].strip()
            if number_value(answer) is None:
                raise ValueError(f'Invalid answer {split}:{index}')
            messages = [{'role': 'system', 'content': config['system_prompt']},
                        {'role': 'user', 'content': item['question']}]
            # Render once so GRPO and both evaluations cannot silently enable thinking.
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                         add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
            if len(ids) > config['max_prompt_length']:
                continue
            rows.append(dict(id=f'{split}:{index}', prompt=prompt, solution=answer))
            questions.add(item['question'])
            if len(rows) == count:
                break
        if len(rows) != count:
            raise ValueError(f'Insufficient eligible {split} examples')
        selected[split] = rows
        (Path(run) / f'{split}_selected.jsonl').write_text(''.join(
            json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    write_json(Path(run) / 'data_manifest.json', {
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(data_dir).glob('gsm8k_*.jsonl'))},
        'selected_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(Path(run).glob('*_selected.jsonl'))},
        'train_count': len(selected['train']), 'test_count': len(selected['test']),
        'test_policy': 'Paired before/after; no test rewards or checkpoint selection',
        'enable_thinking': False,
    })
    return selected['train'], selected['test']


def wilson(correct, n):
    p, z = correct / n, 1.96
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [center - radius, center + radius]


def smoke_gate(summary, config, available_seconds):
    """Technical and budget checks only; never gate on held-out accuracy."""
    reasons = []
    if not summary.get('completed') or summary.get('optimizer_steps') != config['smoke_steps']:
        reasons.append('smoke did not finish the requested optimizer steps')
    if not summary.get('adapter_updated'):
        reasons.append('no verified adapter update')
    if not summary.get('expert_adapter_updated'):
        reasons.append('no verified expert adapter update')
    if summary.get('trainable_expert_parameters', 0) <= 0:
        reasons.append('expert LoRA was not attached')
    gpu_total = summary.get('gpu_total_gib', 0)
    gpu_peak = summary.get('peak_reserved_gib', float('inf'))
    if (not math.isfinite(gpu_total) or not math.isfinite(gpu_peak) or gpu_peak <= 0 or
            gpu_total - gpu_peak < config['minimum_free_gpu_gib']):
        reasons.append('insufficient measured GPU memory headroom')
    steps = summary.get('optimizer_steps', 0)
    seconds = summary.get('training_seconds', float('inf'))
    if steps <= 0 or not math.isfinite(seconds) or seconds <= 0:
        estimate = float('inf')
    else:
        # Include a conservative allowance for length variation and full paired evaluation.
        estimate = (seconds / steps * expected_steps(config) * 1.3 +
                    summary.get('evaluation_seconds', float('inf')) *
                    config['test_samples'] / config['smoke_eval_samples'] * 1.3 + 300)
    if not math.isfinite(estimate) or estimate > available_seconds:
        reasons.append('projected full run exceeds remaining job time budget')
    return {'passed': not reasons, 'reasons': reasons,
            'estimated_full_seconds': estimate if math.isfinite(estimate) else None,
            'available_seconds': available_seconds}
