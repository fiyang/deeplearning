"""Screen 1024 unseen GSM8K training questions; inference only, fixed adapter."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from common import ROOT, wilson, write_json
from probe_training_signal import group_metrics, summarize, training_pools
from round2 import restore_adapter


def select_unseen(tokenizer, model_config, source_run, data_dir, screen):
    _, eligible, pools = training_pools(tokenizer, model_config, source_run, data_dir)
    if len(eligible) != screen['expected_pool_size']:
        raise ValueError('Eligible pool differs from the reviewed screening population')
    n = screen['samples']
    if not 1 <= n <= min(1024, len(eligible)):
        raise ValueError('Screen requires 1 to 1024 distinct eligible questions')
    rows = random.Random(screen['seed']).sample(eligible, n)
    return rows, pools


def classify(group, known_review_ids=()):
    """Keep strict-reward differences separate from valid numeric differences."""
    valid_rewards = {r for a, r in zip(group['parsed_answers'], group['rewards']) if a is not None}
    numeric_mixed = valid_rewards == {0, 1}
    flags = []
    if group['valid_answer_count'] != 4:
        flags.append('invalid_format')
    if any(group['truncated']):
        flags.append('truncated')
    if group['id'] in known_review_ids:
        flags.append('known_question_ambiguity')
    return dict(valid_numeric_mixed=numeric_mixed,
                format_only_mixed=group['category'] == 'mixed' and not numeric_mixed,
                clean_numeric_mixed=numeric_mixed and not flags,
                review_flags=flags)


def screen_summary(groups, expected, seconds, known_review_ids=()):
    ids = [g['id'] for g in groups]
    if len(set(ids)) != len(ids) or len(groups) > expected:
        raise ValueError('Duplicate or excess screened question IDs')
    stats = summarize(groups)
    classified = [classify(g, known_review_ids) for g in groups]
    counts = {key: sum(c[key] for c in classified)
              for key in ['valid_numeric_mixed', 'format_only_mixed', 'clean_numeric_mixed']}
    flags = Counter(flag for c in classified for flag in c['review_flags'])
    n = len(groups)
    return dict(completed=n == expected, optimizer_updates=0, screened_questions=n,
                expected_questions=expected, completions=n * 4, seconds=seconds, **stats,
                **{key + '_count': value for key, value in counts.items()},
                **{key + '_rate': value / n for key, value in counts.items()},
                valid_numeric_mixed_wilson_95_ci=wilson(counts['valid_numeric_mixed'], n),
                review_flag_counts=dict(flags),
                valid_numeric_mixed_ids=[g['id'] for g, c in zip(groups, classified) if c['valid_numeric_mixed']],
                clean_numeric_mixed_ids=[g['id'] for g, c in zip(groups, classified) if c['clean_numeric_mixed']],
                numeric_mixed_groups_per_hour=counts['valid_numeric_mixed'] / max(seconds / 3600, 1e-9),
                note='Fixed-model screening, not test accuracy. Clean candidates still require question/label review. No training is launched.')


def budget_deadline(screen, budget, price, now, started_at, platform_deadline):
    values = [budget, price, now, started_at, platform_deadline]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('Budget, price and times must be positive and finite')
    if budget > screen['max_compute_budget_cny'] or price > screen['max_price_cny_per_hour']:
        raise ValueError('Budget/price exceeds this screening plan')
    if started_at > now or platform_deadline > started_at + screen['platform_time_limit_seconds']:
        raise ValueError('Invalid instance start time or platform deadline beyond seven-hour plan')
    reserve = screen['shutdown_reserve_seconds']
    deadline = min(now + screen['job_time_limit_seconds'],
                   started_at + budget / price * 3600 - reserve,
                   platform_deadline - reserve)
    if deadline - now < 600:
        raise ValueError('Less than ten minutes remain for this screening')
    return int(deadline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config_screen.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'qwen7b_grpo/data')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    screen = json.loads(args.config.read_text())
    source_run = Path(screen['source_run'])
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    groups = []
    try:
        if screen['num_generations'] != 4 or screen['samples'] != 1024:
            raise ValueError('Expected the 1024-question, four-answer screening plan')
        if not math.isfinite(args.deadline) or args.deadline <= time.time() + 600:
            raise ValueError('A future screening deadline is required')
        if shutil.disk_usage(run).free < 2**30:
            raise RuntimeError('At least 1 GiB free disk space is required')
        source_summary = json.loads((source_run / 'summary.json').read_text())
        config = json.loads((source_run / 'config.json').read_text())
        if not source_summary.get('completed') or source_summary.get('optimizer_steps') != 512:
            raise ValueError('Expected the completed second-round adapter')
        if config['precision'] != 'bf16' or config['enable_thinking'] or config['max_completion_length'] != 512:
            raise ValueError('Unexpected source precision or generation contract')
        from rl.qwen7b_grpo.prepare_data import FILES, validate
        for split in ('train', 'test'):
            validate(args.data_dir / f'gsm8k_{split}.jsonl', *FILES[split])
        source_manifest = json.loads((source_run / 'data_manifest.json').read_text())
        for name in ('train_selected.jsonl', 'validation_selected.jsonl'):
            if hashlib.sha256((source_run / name).read_bytes()).hexdigest() != source_manifest['selected_sha256'][name]:
                raise ValueError('Saved train/validation split hash changed')
        os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')
        os.environ.setdefault('UNSLOTH_MOE_BACKEND', 'unsloth_triton')
        from unsloth import FastLanguageModel
        import torch
        from transformers import set_seed
        versions = {p: importlib.metadata.version(p) for p in
                    ['torch', 'transformers', 'trl', 'unsloth', 'unsloth_zoo', 'peft', 'datasets']}
        if versions != json.loads((source_run / 'environment.json').read_text())['versions']:
            raise RuntimeError('Training environment versions changed')
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(0).total_memory / 2**30 < 78:
            raise RuntimeError('Expected one 80GB-class GPU')
        set_seed(screen['seed'])
        torch.set_num_threads(8)
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path, max_seq_length=config['max_prompt_length'] + config['max_completion_length'],
            dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
            fast_inference=False, device_map={'': 0}, trust_remote_code=False)
        if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
            raise RuntimeError('Unexpected quantized base')
        model = FastLanguageModel.get_peft_model(model, r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'], target_modules=config['target_modules'], bias='none',
            use_gradient_checkpointing=True, random_state=config['seed'])
        restore_adapter(model, source_run / 'adapter')
        model.requires_grad_(False)
        FastLanguageModel.for_inference(model)
        model.eval()
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        rows, pools = select_unseen(tokenizer, config, source_run, args.data_dir, screen)
        selected = run / 'selected_train_questions.jsonl'
        selected.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        write_json(run / 'config.json', screen)
        write_json(run / 'manifest.json', dict(source_run=str(source_run), source_adapter='adapter',
            source_summary_sha256=hashlib.sha256((source_run / 'summary.json').read_bytes()).hexdigest(),
            selected_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
            pools=pools, versions=versions, seed=screen['seed'], optimizer_updates=0,
            sampling={k: screen[k] for k in ('temperature', 'top_p', 'top_k', 'num_generations')},
            max_completion_length=512, precision='bf16', enable_thinking=False))
        with (run / 'rollout_groups.jsonl').open('w') as output, (run / 'numeric_mixed_candidates.jsonl').open('w') as candidates:
            for index, row in enumerate(rows):
                if time.time() >= args.deadline - 120:
                    raise TimeoutError('Screening deadline reserve reached')
                group_seed = screen['seed'] + index
                set_seed(group_seed)
                inputs = tokenizer([row['prompt']] * 4, return_tensors='pt', padding=True,
                                   add_special_tokens=False).to('cuda')
                with torch.inference_mode():
                    tokens = model.generate(**inputs, do_sample=True, use_cache=True,
                        temperature=screen['temperature'], top_p=screen['top_p'], top_k=screen['top_k'],
                        num_return_sequences=1, max_new_tokens=512,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                generated = tokens[:, inputs.input_ids.shape[1]:]
                texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                group = dict(id=row['id'], seed=group_seed, solution=row['solution'], completions=texts,
                    **group_metrics(texts, row['solution']),
                    truncated=[tokenizer.eos_token_id not in ids for ids in generated.tolist()])
                group.update(classify(group, screen['known_review_ids']))
                output.write(json.dumps(group, ensure_ascii=False) + '\n'); output.flush()
                if group['valid_numeric_mixed']:
                    candidates.write(json.dumps(dict(group, prompt=row['prompt']), ensure_ascii=False) + '\n')
                    candidates.flush()
                groups.append(group)
                elapsed = time.monotonic() - started
                status = dict(stage='screening', completed_questions=len(groups), total_questions=len(rows),
                              seconds=elapsed, estimated_remaining_seconds=elapsed / len(groups) * (len(rows) - len(groups)))
                write_json(run / 'status.json', status)
                print('SCREEN_PROGRESS ' + json.dumps(status), flush=True)
                if len(groups) % 16 == 0:
                    write_json(run / 'partial_summary.json', screen_summary(groups, len(rows), elapsed, screen['known_review_ids']))
        result = screen_summary(groups, len(rows), time.monotonic() - started, screen['known_review_ids'])
        result['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        write_json(run / 'summary.json', result)
        write_json(run / 'status.json', dict(stage='complete', **result))
        print('SCREEN_COMPLETE ' + json.dumps(result), flush=True)
    except BaseException as error:
        if groups:
            write_json(run / 'partial_summary.json', screen_summary(groups, screen['samples'], time.monotonic() - started, screen['known_review_ids']))
        write_json(run / 'status.json', dict(stage='failed', completed=False, completed_questions=len(groups),
                                            error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
