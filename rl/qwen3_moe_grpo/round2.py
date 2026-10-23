"""Data isolation, checkpoint restoration and comparisons for the expanded run."""
import hashlib
import json
from pathlib import Path
import random

from common import number_value, write_json


def read_rows(path):
    # JSON strings may contain literal U+2028/U+2029: splitlines() would split
    # inside a valid JSON record. JSONL boundaries are physical newline bytes.
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def select_training_validation(tokenizer, config, data_dir, run):
    """Reserve the shuffled tail for validation; never use official test rewards."""
    data_dir, run = Path(data_dir), Path(run)
    source = read_rows(data_dir / 'gsm8k_train.jsonl')
    test_questions = {r['question'] for r in read_rows(data_dir / 'gsm8k_test.jsonl')}
    indices = list(range(len(source)))
    random.Random(config['seed']).shuffle(indices)
    eligible, used, excluded = [], set(), []
    for index in indices:
        item = source[index]
        if item['question'] in test_questions or item['question'] in used:
            excluded.append(dict(id=f'train:{index}', reason='duplicate_or_test_overlap'))
            continue
        answer = item['answer'].rsplit('####', 1)[-1].strip()
        if number_value(answer) is None:
            raise ValueError(f'Invalid train answer {index}')
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': config['system_prompt']},
             {'role': 'user', 'content': item['question']}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        if len(tokenizer(prompt, add_special_tokens=False)['input_ids']) > config['max_prompt_length']:
            excluded.append(dict(id=f'train:{index}', reason='overlength'))
            continue
        eligible.append(dict(id=f'train:{index}', prompt=prompt, solution=answer))
        used.add(item['question'])
    ntrain, nval = config['train_samples'], config['validation_samples']
    if nval <= 0 or len(eligible) < ntrain + nval:
        raise ValueError('Insufficient disjoint training and validation examples')
    train, validation = eligible[:ntrain], eligible[-nval:]
    source_run = config.get('comparison_source_run')
    if source_run:
        previous = read_rows(Path(source_run) / 'train_selected.jsonl')
        if train[:len(previous)] != previous:
            raise ValueError('Expanded training must preserve the original 512-question prefix')
    for name, rows in [('train', train), ('validation', validation)]:
        (run / f'{name}_selected.jsonl').write_text(''.join(
            json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_json(run / 'data_manifest.json', dict(
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(data_dir.glob('gsm8k_*.jsonl'))},
        selected_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(run.glob('*_selected.jsonl'))},
        train_count=len(train), validation_count=len(validation), excluded=excluded,
        validation_policy='Fixed shuffled train tail, disjoint from all training and official test questions',
        test_policy='Official 1319 evaluated only after validation checkpoint selection'))
    return train, validation


def validation_is_better(current, best):
    """Choose only by validation correct count; retain the earlier epoch on ties."""
    if best is not None and current['n'] != best['n']:
        raise ValueError('Validation sample count changed')
    return best is None or current['correct'] > best['correct']


def restore_adapter(model, path):
    import torch
    from peft.utils.save_and_load import (get_peft_model_state_dict, load_peft_weights,
                                         set_peft_model_state_dict)
    weights = load_peft_weights(str(path), device='cpu')
    current = get_peft_model_state_dict(model)
    if set(current) != set(weights) or any(current[k].shape != weights[k].shape for k in current):
        raise RuntimeError('Selected checkpoint does not match the live adapters')
    loaded = set_peft_model_state_dict(model, weights)
    if loaded.unexpected_keys or any('lora_' in k for k in loaded.missing_keys):
        raise RuntimeError('Selected adapter was not fully restored')
    restored = get_peft_model_state_dict(model)
    if any(not torch.equal(p.detach().cpu(), weights[k].to(dtype=p.dtype)) for k, p in restored.items()):
        raise RuntimeError('Selected checkpoint tensor verification failed')


def check_reference_config(config, old):
    keys = ['model_id', 'precision', 'enable_thinking', 'seed', 'system_prompt',
            'max_prompt_length', 'max_completion_length', 'eval_batch_size',
            'lora_rank', 'lora_alpha', 'lora_dropout', 'target_modules',
            'learning_rate', 'num_generations', 'generation_batch_size',
            'gradient_accumulation_steps', 'per_device_train_batch_size',
            'temperature', 'top_p', 'top_k', 'beta', 'loss_type', 'warmup_ratio']
    changed = [k for k in keys if config[k] != old[k]]
    if changed:
        raise ValueError(f'Unexpected changes beyond data size / epochs: {changed}')


def evaluate_selected(model, tokenizer, config, data_dir, run, deadline):
    # Import lazily: offline splitting and training do not read test score artifacts.
    from eval_full import compare, full_test_rows
    from rl.qwen3_moe_grpo.train import evaluate
    source = Path(config['comparison_source_run'])
    reference = Path(config['comparison_full_eval_run'])
    reference_summary = json.loads((reference / 'summary.json').read_text())
    if not reference_summary.get('completed') or reference_summary['full']['n'] != 1319:
        raise ValueError('A completed full-set reference evaluation is required')
    rows = full_test_rows(tokenizer, config, read_rows(Path(data_dir) / 'gsm8k_test.jsonl'),
                         read_rows(source / 'test_selected.jsonl'))
    cached_rows = read_rows(reference / 'baseline' / 'test_selected.jsonl')
    if len(rows) != 1319 or rows != cached_rows:
        raise ValueError('Full test prompts/order differ from the reference evaluation')
    (run / 'test_selected.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    metrics, after = evaluate(model, tokenizer, rows, config, run, 'final', deadline)
    baseline = read_rows(reference / 'baseline' / 'baseline_predictions.jsonl')
    previous = read_rows(reference / 'final' / 'final_predictions.jsonl')
    return dict(final=metrics, versus_base=compare(baseline, after),
                versus_round1=compare(previous, after), baseline_reused_from=str(reference),
                evaluation_policy='Same 1319 prompts/order, greedy, non-thinking, 512 tokens; selected by validation only')
