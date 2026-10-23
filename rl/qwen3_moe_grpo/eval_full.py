"""Evaluation only: all original GSM8K test questions, before/after a saved adapter."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

from common import event, load_config, number_value, wilson, write_json
from rl.qwen3_moe_grpo.train import evaluate
from rl.qwen7b_grpo.prepare_data import FILES, validate


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def full_test_rows(tokenizer, config, source, previous):
    """Never drop/truncate test questions; preserve original 128 batching first."""
    if len({r['question'] for r in source}) != len(source):
        raise ValueError('Duplicate test questions')
    rows = {}
    for index, item in enumerate(source):
        answer = item['answer'].rsplit('####', 1)[-1].strip()
        if number_value(answer) is None:
            raise ValueError(f'Invalid answer test:{index}')
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': config['system_prompt']},
             {'role': 'user', 'content': item['question']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        length = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        if length > config['max_prompt_length']:
            raise ValueError(f'test:{index} has {length} tokens; cannot silently exclude it')
        rows[f'test:{index}'] = dict(id=f'test:{index}', prompt=prompt, solution=answer)
    ids = [r['id'] for r in previous]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate IDs in prior test subset')
    if any(rows.get(r['id']) != r for r in previous):
        raise ValueError('Original test prompt/answer mismatch')
    remaining = [key for key in rows if key not in set(ids)]
    random.Random(config['seed'] + 1).shuffle(remaining)
    return [rows[key] for key in ids + remaining]


def compare(before, after):
    if not before or [r['id'] for r in before] != [r['id'] for r in after]:
        raise ValueError('Missing or unpaired predictions')
    if len({r['id'] for r in before}) != len(before):
        raise ValueError('Duplicate evaluation IDs')
    if any(a['solution'] != b['solution'] for a, b in zip(before, after)):
        raise ValueError('Paired solutions differ')
    n = len(before)
    a = sum(r['correct'] for r in before)
    b = sum(r['correct'] for r in after)
    improved = sum(not x['correct'] and y['correct'] for x, y in zip(before, after))
    regressed = sum(x['correct'] and not y['correct'] for x, y in zip(before, after))
    discordant = improved + regressed
    # Exact two-sided paired test; log space avoids overflow for large samples.
    p = 1.0 if not discordant else min(1.0, 2 * sum(math.exp(
        math.lgamma(discordant + 1) - math.lgamma(k + 1) -
        math.lgamma(discordant - k + 1) - discordant * math.log(2))
        for k in range(min(improved, regressed) + 1)))
    return dict(n=n, baseline_correct=a, final_correct=b,
                baseline_accuracy=a/n, final_accuracy=b/n,
                accuracy_delta_pp=100*(b-a)/n,
                baseline_wilson_95_ci=wilson(a, n), final_wilson_95_ci=wilson(b, n),
                wrong_to_right=improved, right_to_wrong=regressed,
                mcnemar_exact_two_sided_p=p)


def aggregate(run, source_run):
    before = read_rows(run / 'baseline' / 'baseline_predictions.jsonl')
    after = read_rows(run / 'final' / 'final_predictions.jsonl')
    if len(before) != FILES['test'][0] or len(after) != FILES['test'][0]:
        raise ValueError('Full 1319-question predictions are required')
    prior = read_rows(source_run / 'test_selected.jsonl')
    prior_ids = {r['id'] for r in prior}
    if not prior_ids.issubset({r['id'] for r in before}):
        raise ValueError('Prior test subset missing from full evaluation')
    result = dict(completed=True, evaluation_only=True, source_run=str(source_run),
                  model='Qwen/Qwen3-30B-A3B', full=compare(before, after),
                  original_subset=compare([r for r in before if r['id'] in prior_ids],
                                          [r for r in after if r['id'] in prior_ids]),
                  newly_evaluated=compare([r for r in before if r['id'] not in prior_ids],
                                          [r for r in after if r['id'] not in prior_ids]),
                  baseline=json.loads((run/'baseline'/'baseline_metrics.json').read_text()),
                  final=json.loads((run/'final'/'final_metrics.json').read_text()),
                  decoding='greedy, non-thinking, one completion per question; max 512 tokens',
                  note='Evaluation only; no parameter updates or test-based checkpoint selection.')
    for label, records in [('baseline', before), ('final', after)]:
        old = {r['id']: r for r in read_rows(source_run / f'{label}_predictions.jsonl')}
        result[f'{label}_original_subset_reproduction'] = dict(
            correctness_changes=sum(r['correct'] != old[r['id']]['correct']
                                    for r in records if r['id'] in old),
            completion_text_changes=sum(r['completion'] != old[r['id']]['completion']
                                        for r in records if r['id'] in old))
    write_json(run/'summary.json', result)
    event(run, 'complete', **result)
    print('FULL_EVAL_COMPLETE ' + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-file', type=Path, required=True)
    parser.add_argument('--phase', choices=['preflight', 'baseline', 'final', 'aggregate'], required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    run, source_run = args.run_dir.resolve(), args.source_run.resolve()
    if run == source_run or source_run in run.parents:
        raise ValueError('Evaluation must use a separate output directory')
    if args.phase == 'aggregate':
        aggregate(run, source_run)
        return
    phase_run = run/args.phase
    phase_run.mkdir(parents=True, exist_ok=False)
    try:
        source_summary = json.loads((source_run/'summary.json').read_text())
        if not source_summary['completed'] or source_summary['optimizer_steps'] != 128:
            raise ValueError('Expected the completed 128-step training run')
        config = load_config(source_run/'config.json')
        if config['max_completion_length'] != 512 or config['eval_batch_size'] != 4:
            raise ValueError('Unexpected original decoding configuration')
        validate(args.data_file, *FILES['test'])
        os.environ.setdefault('UNSLOTH_MOE_BACKEND', 'unsloth_triton')
        os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')
        from unsloth import FastLanguageModel
        import torch
        from transformers import set_seed
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('One BF16 CUDA GPU is required')
        if torch.cuda.get_device_properties(0).total_memory / 2**30 < 78:
            raise RuntimeError('An 80GB-class GPU is required')
        if time.time() >= args.deadline:
            raise TimeoutError('Evaluation deadline already reached')
        set_seed(config['seed'])
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = True
        versions = {p: importlib.metadata.version(p) for p in
                    ['torch', 'transformers', 'trl', 'unsloth', 'unsloth_zoo', 'peft', 'datasets']}
        original_versions = json.loads((source_run/'environment.json').read_text())['versions']
        if versions != original_versions:
            raise RuntimeError(f'Environment changed: {versions} != {original_versions}')
        write_json(phase_run/'environment.json', dict(versions=versions, gpu=torch.cuda.get_device_name()))
        event(run, 'loading_model', phase=args.phase)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=config['max_prompt_length'] + config['max_completion_length'],
            dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False,
            full_finetuning=False, fast_inference=False, device_map={'': 0}, trust_remote_code=False)
        if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
            raise RuntimeError('Unexpected quantized model')
        dtypes = Counter()
        for name, p in model.named_parameters():
            if p.device.type != 'cuda':
                raise RuntimeError(f'Unexpected offload: {name}')
            dtypes[str(p.dtype)] += p.numel()
        if dtypes.get('torch.bfloat16', 0) < 29_000_000_000:
            raise RuntimeError(f'Unexpected base precision: {dtypes}')
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        rows = full_test_rows(tokenizer, config, read_rows(args.data_file),
                             read_rows(source_run/'test_selected.jsonl'))
        payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
        (phase_run/'test_selected.jsonl').write_text(payload)
        write_json(phase_run/'config.json', config)
        write_json(phase_run/'manifest.json', dict(test_count=len(rows),
            source_sha256=FILES['test'][1], selected_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            source_run=str(source_run), phase=args.phase, training_updates=0))
        if args.phase != 'baseline':
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict, get_peft_model_state_dict
            model = FastLanguageModel.get_peft_model(model, r=config['lora_rank'],
                lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                target_modules=config['target_modules'], bias='none',
                use_gradient_checkpointing=True, random_state=config['seed'])
            weights = load_peft_weights(str(source_run/'adapter'), device='cpu')
            current = get_peft_model_state_dict(model)
            if set(weights) != set(current):
                raise RuntimeError('Saved/live adapter key mismatch')
            if any(weights[k].shape != current[k].shape for k in weights):
                raise RuntimeError('Saved/live adapter shape mismatch')
            loaded = set_peft_model_state_dict(model, weights)
            if loaded.unexpected_keys or any('lora_' in k for k in loaded.missing_keys):
                raise RuntimeError(f'Adapter not fully restored: {loaded}')
            restored = get_peft_model_state_dict(model)
            if any(not torch.equal(restored[k].detach().cpu(), weights[k].to(dtype=restored[k].dtype))
                   for k in weights):
                raise RuntimeError('Restored adapter differs from saved tensors')
            expert_count = sum(p.numel() for k, p in restored.items() if '.experts.' in k)
            total_count = sum(p.numel() for p in restored.values())
            original_parameters = json.loads((source_run/'parameters.json').read_text())
            if expert_count != original_parameters['trainable_expert_parameters'] or total_count != original_parameters['trainable']:
                raise RuntimeError('Attention/expert adapter count mismatch')
            write_json(phase_run/'adapter_verified.json', dict(all_tensors_equal=True,
                tensor_count=len(weights), total_parameters=total_count, expert_parameters=expert_count))
            del weights, current, restored
        model.requires_grad_(False)
        FastLanguageModel.for_inference(model)
        if args.phase == 'preflight':
            rows = rows[:8]
        _, predictions = evaluate(model, tokenizer, rows, config, phase_run, args.phase, args.deadline)
        if args.phase == 'preflight':
            old = {r['id']: r for r in read_rows(source_run/'final_predictions.jsonl')}
            changes = sum(r['correct'] != old[r['id']]['correct'] for r in predictions)
            if changes:
                raise RuntimeError(f'Saved adapter failed 8-question reproduction: {changes} changed scores')
            write_json(phase_run/'gate.json', dict(passed=True, evaluated=8, correctness_changes=changes))
        event(run, 'phase_complete', phase=args.phase, evaluated=len(predictions))
    except BaseException as exc:
        event(run, 'failed', phase=args.phase, error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
