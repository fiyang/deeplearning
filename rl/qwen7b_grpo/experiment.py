"""One bounded Qwen2.5-7B LoRA/GRPO experiment with paired held-out evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import time

from rewards import correctness_reward, extract_answer, number_value, score

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def event(run, stage, **extra):
    value = dict(stage=stage, timestamp=time.time(), **extra)
    write_json(run / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare_data(tokenizer, config, run):
    from datasets import Dataset
    selected = {}
    all_questions = set()
    for split, size in [('test', config['test_samples']), ('train', config['train_samples'])]:
        source = ROOT / 'data' / f'gsm8k_{split}.jsonl'
        raw = load_rows(source)
        indices = list(range(len(raw)))
        random.Random(config['seed'] + (1 if split == 'test' else 0)).shuffle(indices)
        rows = []
        for idx in indices:
            sample = raw[idx]
            if sample['question'] in all_questions:
                continue
            target = sample['answer'].rsplit('####', 1)[-1].strip()
            if number_value(target) is None:
                raise ValueError(f'Invalid ground truth at {split}:{idx}')
            prompt = [{'role': 'system', 'content': config['system_prompt']},
                      {'role': 'user', 'content': sample['question']}]
            token_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            if len(token_ids) > config['max_prompt_length']:
                continue  # Never silently cut a question in half.
            rows.append(dict(id=f'{split}:{idx}', prompt=prompt, solution=target))
            all_questions.add(sample['question'])
            if len(rows) == size:
                break
        if len(rows) != size:
            raise ValueError(f'Insufficient eligible {split} examples')
        selected[split] = rows
        with (run / f'{split}_selected.jsonl').open('w') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(run / 'data_manifest.json', {
        'source': 'https://github.com/openai/grade-school-math/tree/master/grade_school_math/data',
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / 'data').glob('gsm8k_*.jsonl'))},
        'selected_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(run.glob('*_selected.jsonl'))},
        'train_count': len(selected['train']), 'test_count': len(selected['test']),
        'selection': 'Seeded shuffle, no duplicate questions, discard overlength prompts',
        'test_policy': 'Same fixed test subset before/after; never used for rewards or checkpoint selection',
    })
    return Dataset.from_list(selected['train']), selected['test']


def wilson(correct, count):
    z = 1.96
    p = correct / count
    d = 1 + z * z / count
    center = (p + z * z / (2 * count)) / d
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / d
    return [center - radius, center + radius]


def evaluate(model, tokenizer, rows, config, run, label):
    import torch
    model.eval()
    started = time.monotonic()
    records = []
    batch_size = config['eval_batch_size']
    output_path = run / f'{label}_predictions.jsonl'
    with output_path.open('w') as out:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts = [tokenizer.apply_chat_template(r['prompt'], tokenize=False, add_generation_prompt=True)
                     for r in batch]
            inputs = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=config['max_completion_length'],
                                        do_sample=False, use_cache=True,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
            completion_ids = result[:, inputs.input_ids.shape[1]:]
            outputs = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            for row, text, ids in zip(batch, outputs, completion_ids, strict=True):
                tokens = ids.tolist()
                eos_seen = tokenizer.eos_token_id in tokens
                length = tokens.index(tokenizer.eos_token_id) + 1 if eos_seen else len(tokens)
                record = dict(id=row['id'], solution=row['solution'], completion=text,
                              correct=bool(score(text, row['solution'])),
                              valid_answer=extract_answer(text) is not None,
                              completion_tokens=length, truncated=not eos_seen)
                records.append(record)
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()
            event(run, label, evaluated=len(records), total=len(rows),
                  accuracy=sum(r['correct'] for r in records) / len(records))
    n = len(records)
    correct = sum(r['correct'] for r in records)
    elapsed = time.monotonic() - started
    summary = dict(label=label, n=n, correct=correct, accuracy=correct/n,
                   wilson_95_ci=wilson(correct, n),
                   valid_answer_rate=sum(r['valid_answer'] for r in records)/n,
                   truncated_rate=sum(r['truncated'] for r in records)/n,
                   seconds=elapsed, completion_tokens_per_second=sum(r['completion_tokens'] for r in records)/elapsed,
                   decoding='greedy, one completion per question',
                   max_completion_length=config['max_completion_length'])
    write_json(run / f'{label}_metrics.json', summary)
    print('EVAL_COMPLETE ' + json.dumps(summary), flush=True)
    torch.cuda.empty_cache()
    return summary, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'config.json').read_text())
    if args.smoke:
        config.update(train_samples=8, test_samples=8, epochs=1, max_completion_length=64,
                      gradient_accumulation_steps=4, save_steps=2, train_time_limit_seconds=600,
                      eval_batch_size=4)
    write_json(run / 'config.json', dict(config, model_path=args.model_path, smoke=args.smoke))
    event(run, 'initializing')
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from trl import GRPOConfig, GRPOTrainer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA and BF16 support are required')
        set_seed(config['seed'])
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
        versions = {p: importlib.metadata.version(p)
                    for p in ['torch', 'trl', 'transformers', 'peft', 'accelerate', 'datasets']}
        write_json(run / 'environment.json', dict(versions=versions, gpu=torch.cuda.get_device_name(),
                   gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory))
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        train, test = prepare_data(tokenizer, config, run)
        event(run, 'loading_model', train_examples=len(train), test_examples=len(test))
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            trust_remote_code=False, device_map={'': 0}, low_cpu_mem_usage=True)
        baseline, before = evaluate(model, tokenizer, test, config, run, 'baseline')

        class Progress(TrainerCallback):
            def __init__(self):
                self.started = time.monotonic()
                self.stopped_early = False
                self.interrupted = False

            def on_step_end(self, args, state, control, **kwargs):
                elapsed = time.monotonic() - self.started
                if self.interrupted or elapsed >= config['train_time_limit_seconds']:
                    self.stopped_early = True
                    control.should_training_stop = True
                    control.should_save = True
                event(run, 'training', step=state.global_step, max_steps=state.max_steps,
                      epoch=state.epoch, training_seconds=elapsed,
                      peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                      stopping_for_budget_or_signal=self.stopped_early)

            def on_log(self, args, state, control, logs=None, **kwargs):
                for key in ('loss', 'grad_norm'):
                    value = (logs or {}).get(key)
                    if isinstance(value, (float, int)) and not math.isfinite(value):
                        raise FloatingPointError(f'Non-finite {key}: {value}')
                record = dict(step=state.global_step, timestamp=time.time(), **(logs or {}))
                with (run / 'metrics.jsonl').open('a') as out:
                    out.write(json.dumps(record) + '\n')
                print('METRICS ' + json.dumps(record), flush=True)

        progress = Progress()
        signal.signal(signal.SIGTERM, lambda *_: setattr(progress, 'interrupted', True))
        lora = LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'],
                          lora_dropout=config['lora_dropout'], bias='none', task_type='CAUSAL_LM',
                          target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                          'gate_proj', 'up_proj', 'down_proj'])
        training_args = GRPOConfig(
            output_dir=str(run / 'checkpoints'), num_train_epochs=config['epochs'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            num_generations=config['num_generations'],
            max_prompt_length=config['max_prompt_length'],
            max_completion_length=config['max_completion_length'],
            learning_rate=config['learning_rate'], lr_scheduler_type='cosine',
            warmup_ratio=config['warmup_ratio'], max_grad_norm=config['max_grad_norm'],
            beta=config['beta'], loss_type=config['loss_type'], scale_rewards='group',
            temperature=config['temperature'], top_p=config['top_p'],
            bf16=True, fp16=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            use_vllm=False, optim='adamw_torch',
            logging_steps=1, save_steps=config['save_steps'], save_total_limit=config['save_total_limit'],
            eval_strategy='no', report_to='none', seed=config['seed'], data_seed=config['seed'],
            dataloader_num_workers=0, remove_unused_columns=False, disable_tqdm=True)
        trainer = GRPOTrainer(model=model, args=training_args, reward_funcs=correctness_reward,
                              train_dataset=train, processing_class=tokenizer,
                              peft_config=lora, callbacks=[progress])
        params = dict(total=sum(p.numel() for p in trainer.model.parameters()),
                      trainable=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))
        write_json(run / 'parameters.json', params)
        event(run, 'training_start', **params)
        progress.started = time.monotonic()
        result = trainer.train()
        trainer.save_model(str(run / 'adapter'))
        tokenizer.save_pretrained(run / 'adapter')
        trainer.save_state()
        write_json(run / 'train_metrics.json', result.metrics)
        final, after = evaluate(trainer.model, tokenizer, test, config, run, 'final')
        if [r['id'] for r in before] != [r['id'] for r in after]:
            raise RuntimeError('Evaluation pairing mismatch')
        summary = dict(
            model=config['model_id'], algorithm='GRPO + BF16 LoRA',
            baseline=baseline, final=final,
            accuracy_delta_pp=100*(final['accuracy']-baseline['accuracy']),
            wrong_to_right=sum(not a['correct'] and b['correct'] for a,b in zip(before,after)),
            right_to_wrong=sum(a['correct'] and not b['correct'] for a,b in zip(before,after)),
            completed_epoch=not progress.stopped_early, smoke=args.smoke,
            optimizer_steps=trainer.state.global_step,
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            note='One seed and a small fixed test subset; GSM8K may have appeared in pretraining. '
                 'No checkpoint selection on test. This does not establish general capability improvement.')
        write_json(run / 'summary.json', summary)
        event(run, 'complete' if not progress.stopped_early else 'budget_limited_complete',
              summary_file=str(run / 'summary.json'), accuracy_delta_pp=summary['accuracy_delta_pp'])
    except BaseException as error:
        event(run, 'failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
