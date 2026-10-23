"""Inference-only pilot: old/new training questions under two sampling settings.

This measures the fixed round-2 adapter now. It cannot reproduce historical
training rollouts. No optimizer or training method is used.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
import unicodedata

from common import ROOT, extract_answer, number_value, score, write_json
from round2 import read_rows, restore_adapter

SETTINGS = [dict(name='current', temperature=0.7, top_p=0.8, top_k=20),
            dict(name='broader', temperature=0.9, top_p=0.95, top_k=20)]


def normalized_text(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def group_metrics(texts, solution):
    if len(texts) != 4:
        raise ValueError('Exactly four sampled completions are required')
    rewards = [int(score(text, solution)) for text in texts]
    answers = [extract_answer(text) for text in texts]
    count = sum(rewards)
    return dict(correct_count=count,
                category='all_correct' if count == 4 else 'all_zero_reward' if count == 0 else 'mixed',
                rewards=rewards, unique_exact_texts=len(set(texts)),
                unique_normalized_texts=len({normalized_text(t) for t in texts}),
                valid_answer_count=sum(a is not None for a in answers),
                unique_valid_answers=len({a for a in answers if a is not None}),
                all_same_valid_answer=all(a is not None for a in answers) and len(set(answers)) == 1,
                normalized_pair_duplicate_rate=sum(normalized_text(texts[i]) == normalized_text(texts[j])
                    for i in range(4) for j in range(i + 1, 4)) / 6,
                parsed_answers=[str(a) if a is not None else None for a in answers])


def summarize(groups):
    if not groups:
        raise ValueError('No sampled groups')
    n = len(groups)
    categories = Counter(g['category'] for g in groups)
    return dict(n=n, category_counts={k: categories[k] for k in ['all_correct', 'all_zero_reward', 'mixed']},
                category_rates={k: categories[k] / n for k in ['all_correct', 'all_zero_reward', 'mixed']},
                sample_accuracy=sum(g['correct_count'] for g in groups) / (4 * n),
                exact_all_four_same_rate=sum(g['unique_exact_texts'] == 1 for g in groups) / n,
                normalized_all_four_same_rate=sum(g['unique_normalized_texts'] == 1 for g in groups) / n,
                normalized_any_duplicate_rate=sum(g['unique_normalized_texts'] < 4 for g in groups) / n,
                mean_unique_normalized_texts=sum(g['unique_normalized_texts'] for g in groups) / n,
                normalized_pair_duplicate_rate=sum(g['normalized_pair_duplicate_rate'] for g in groups) / n,
                all_same_valid_answer_rate=sum(g['all_same_valid_answer'] for g in groups) / n,
                invalid_answer_rate=1 - sum(g['valid_answer_count'] for g in groups) / (4 * n),
                truncated_rate=sum(sum(g['truncated']) for g in groups) / (4 * n))


def training_pools(tokenizer, config, source_run, data_dir):
    """Rebuild the same train-only candidate pool for pilots and larger screens."""
    old = read_rows(source_run / 'train_selected.jsonl')
    validation = read_rows(source_run / 'validation_selected.jsonl')
    blocked_ids = {r['id'] for r in old + validation}
    test_questions = {r['question'] for r in read_rows(data_dir / 'gsm8k_test.jsonl')}
    source = read_rows(data_dir / 'gsm8k_train.jsonl')
    # Exclude question-text duplicates too, rather than relying only on IDs.
    blocked_questions = {source[int(r['id'].split(':')[1])]['question'] for r in old + validation}
    eligible, seen = [], set()
    for i, item in enumerate(source):
        question = item['question']
        if f'train:{i}' in blocked_ids or question in test_questions | blocked_questions | seen:
            continue
        seen.add(question)
        solution = item['answer'].rsplit('####', 1)[-1].strip()
        if number_value(solution) is None:
            continue
        prompt = tokenizer.apply_chat_template(
            [dict(role='system', content=config['system_prompt']), dict(role='user', content=question)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if len(tokenizer(prompt, add_special_tokens=False)['input_ids']) <= config['max_prompt_length']:
            eligible.append(dict(id=f'train:{i}', prompt=prompt, solution=solution))
    return old, eligible, dict(seen_training_pool=len(old), unseen_eligible_pool=len(eligible),
                              held_out_validation_count=len(validation))


def choose_rows(tokenizer, config, source_run, data_dir, per_cohort, seed):
    old, eligible, pools = training_pools(tokenizer, config, source_run, data_dir)
    if per_cohort > min(len(old), len(eligible)):
        raise ValueError('Not enough disjoint eligible training questions')
    result = [dict(row, cohort=cohort) for cohort, pool, offset in [('seen_train', old, 0), ('unseen_train', eligible, 1)]
              for row in random.Random(seed + offset).sample(pool, per_cohort)]
    return result, dict(pools, per_cohort=per_cohort)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'qwen7b_grpo/data')
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--per-cohort', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    all_groups = []
    try:
        if not 1 <= args.per_cohort <= 64:
            raise ValueError('Reviewed pilot is limited to 64 questions per cohort')
        config = json.loads((args.source_run / 'config.json').read_text())
        source_summary = json.loads((args.source_run / 'summary.json').read_text())
        if not source_summary.get('completed') or source_summary.get('optimizer_steps') != 512:
            raise ValueError('Expected the completed second-round adapter')
        os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')
        os.environ.setdefault('UNSLOTH_MOE_BACKEND', 'unsloth_triton')
        from unsloth import FastLanguageModel
        import torch
        from transformers import set_seed
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(0).total_memory / 2**30 < 78:
            raise RuntimeError('Expected one 80GB-class GPU')
        set_seed(args.seed)
        torch.set_num_threads(8)
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path, max_seq_length=config['max_prompt_length'] + config['max_completion_length'],
            dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False,
            full_finetuning=False, fast_inference=False, device_map={'': 0}, trust_remote_code=False)
        if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
            raise RuntimeError('Quantized model unexpectedly loaded')
        model = FastLanguageModel.get_peft_model(model, r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'], target_modules=config['target_modules'], bias='none',
            use_gradient_checkpointing=True, random_state=config['seed'])
        restore_adapter(model, args.source_run / 'adapter')
        FastLanguageModel.for_inference(model)
        model.eval()
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        rows, pools = choose_rows(tokenizer, config, args.source_run, args.data_dir, args.per_cohort, args.seed)
        selected_path = run / 'selected_train_questions.jsonl'
        selected_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        write_json(run / 'manifest.json', dict(source_run=str(args.source_run), source_adapter='adapter',
            settings=SETTINGS, pools=pools, seed=args.seed, num_generations=4,
            precision='bf16', enable_thinking=False, optimizer_updates=0,
            selected_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),
            purpose='Current fixed adapter sampling audit, not historical rollout reconstruction'))
        with (run / 'rollout_groups.jsonl').open('w') as out:
            for setting in SETTINGS:
                for index, row in enumerate(rows):
                    if time.time() >= args.deadline - 120:
                        raise TimeoutError('Diagnostic deadline reserve reached')
                    # Set once per group, never once per completion. Four rows draw
                    # from one RNG stream; the paired setting uses the same seed.
                    group_seed = args.seed + index
                    set_seed(group_seed)
                    inputs = tokenizer([row['prompt']] * 4, return_tensors='pt', padding=True,
                                       add_special_tokens=False).to('cuda')
                    with torch.inference_mode():
                        tokens = model.generate(**inputs, do_sample=True, use_cache=True,
                            temperature=setting['temperature'], top_p=setting['top_p'], top_k=setting['top_k'],
                            num_return_sequences=1, max_new_tokens=config['max_completion_length'],
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    generated = tokens[:, inputs.input_ids.shape[1]:]
                    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                    group = dict(id=row['id'], cohort=row['cohort'], setting=setting['name'], seed=group_seed,
                        solution=row['solution'], completions=texts, **group_metrics(texts, row['solution']),
                        truncated=[tokenizer.eos_token_id not in ids for ids in generated.tolist()])
                    out.write(json.dumps(group, ensure_ascii=False) + '\n')
                    out.flush()
                    all_groups.append(group)
                    status = dict(stage='sampling', completed_groups=len(all_groups),
                                  total_groups=len(rows) * len(SETTINGS), seconds=time.monotonic() - started)
                    write_json(run / 'status.json', status)
                    print(json.dumps(status), flush=True)
        results = {s['name']: {cohort: summarize([g for g in all_groups if g['setting'] == s['name'] and g['cohort'] == cohort])
                   for cohort in ['seen_train', 'unseen_train']} for s in SETTINGS}
        summary = dict(completed=True, optimizer_updates=0, groups=len(all_groups),
                       completions=len(all_groups) * 4, results=results, pools=pools,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                       seconds=time.monotonic() - started,
                       note='Small fixed-model pilot; not historical duplication rate or a final model accuracy evaluation')
        write_json(run / 'summary.json', summary)
        write_json(run / 'status.json', dict(stage='complete', **summary))
        print('SIGNAL_PROBE_COMPLETE ' + json.dumps(summary), flush=True)
    except BaseException as error:
        write_json(run / 'status.json', dict(stage='failed', completed=False,
            completed_groups=len(all_groups), error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
