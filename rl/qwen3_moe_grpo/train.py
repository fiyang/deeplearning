"""One bounded Qwen3-30B-A3B BF16 LoRA/GRPO run. CUDA validation is mandatory."""
import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

from common import (ROOT, correctness_reward, event, expected_steps, extract_answer,
                    load_config, score, select_rows, wilson, write_json)


def evaluate(model, tokenizer, rows, config, run, label, deadline):
    import torch
    model.eval()
    records = []
    started = time.monotonic()
    with (run / f'{label}_predictions.jsonl').open('w') as out:
        for start in range(0, len(rows), config['eval_batch_size']):
            if time.time() >= deadline:
                raise TimeoutError('Job deadline reached during evaluation')
            batch = rows[start:start + config['eval_batch_size']]
            inputs = tokenizer([r['prompt'] for r in batch], return_tensors='pt',
                               padding=True, add_special_tokens=False).to('cuda')
            with torch.inference_mode():
                result = model.generate(**inputs, do_sample=False, use_cache=True,
                         max_new_tokens=config['max_completion_length'],
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tokens = result[:, inputs.input_ids.shape[1]:]
            texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            for row, text, token_ids in zip(batch, texts, tokens.tolist(), strict=True):
                ended = tokenizer.eos_token_id in token_ids
                length = token_ids.index(tokenizer.eos_token_id) + 1 if ended else len(token_ids)
                record = dict(id=row['id'], solution=row['solution'], completion=text,
                              correct=bool(score(text, row['solution'])),
                              valid_answer=extract_answer(text) is not None,
                              completion_tokens=length, truncated=not ended)
                records.append(record)
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()
            event(run, label, evaluated=len(records), total=len(rows),
                  accuracy=sum(r['correct'] for r in records) / len(records))
    elapsed = time.monotonic() - started
    correct, n = sum(r['correct'] for r in records), len(records)
    result = dict(correct=correct, n=n, accuracy=correct / n, wilson_95_ci=wilson(correct, n),
                  valid_answer_rate=sum(r['valid_answer'] for r in records) / n,
                  truncated_rate=sum(r['truncated'] for r in records) / n,
                  seconds=elapsed, completion_tokens_per_second=
                  sum(r['completion_tokens'] for r in records) / elapsed,
                  decoding='greedy, non-thinking, one completion per question')
    write_json(run / f'{label}_metrics.json', result)
    print('EVAL_COMPLETE ' + json.dumps({'label': label, **result}), flush=True)
    torch.cuda.empty_cache()
    return result, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-dir', default=str(ROOT.parent / 'qwen7b_grpo' / 'data'))
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--deadline', required=True, type=float, help='Unix time; set by run.sh')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    config = load_config()
    write_json(run / 'config.json', dict(config, smoke=args.smoke, deadline=args.deadline,
                                       model_path=args.model_path))
    event(run, 'initializing')
    try:
        # Must precede transformers/trl imports so patches apply to both train and generate.
        os.environ.setdefault('UNSLOTH_MOE_BACKEND', 'unsloth_triton')
        os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')
        from unsloth import FastLanguageModel
        import torch
        from datasets import Dataset
        from transformers import TrainerCallback, set_seed
        from trl import GRPOConfig, GRPOTrainer

        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Exactly one visible BF16 CUDA GPU is required')
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total_gib < 78:
            raise RuntimeError(f'Expected an 80GB-class GPU, found {total_gib:.1f} GiB')
        if time.time() >= args.deadline:
            raise TimeoutError('Job deadline already reached')
        if shutil.disk_usage(run).free < config['minimum_free_disk_gib'] * 2**30:
            raise RuntimeError('Insufficient disk headroom for adapter checkpoints')
        set_seed(config['seed'])
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
        versions = {p: importlib.metadata.version(p) for p in
                    ['torch', 'transformers', 'trl', 'unsloth', 'unsloth_zoo', 'peft', 'datasets']}
        write_json(run / 'environment.json', dict(versions=versions,
                   gpu=torch.cuda.get_device_name(), gpu_total_gib=total_gib,
                   moe_backend=os.environ['UNSLOTH_MOE_BACKEND'],
                   compile_disable=os.environ['UNSLOTH_COMPILE_DISABLE']))
        event(run, 'loading_model')
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=config['max_prompt_length'] + config['max_completion_length'],
            dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False,
            full_finetuning=False, fast_inference=False, device_map={'': 0},
            trust_remote_code=False)
        if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
            raise RuntimeError('Quantized model unexpectedly loaded')
        dtype_counts = Counter()
        for name, parameter in model.named_parameters():
            if parameter.device.type != 'cuda':
                raise RuntimeError(f'Unexpected CPU/disk offload: {name}')
            dtype_counts[str(parameter.dtype)] += parameter.numel()
        if dtype_counts.get('torch.bfloat16', 0) < 29_000_000_000:
            raise RuntimeError(f'Unexpected base parameter precision: {dtype_counts}')
        write_json(run / 'base_dtypes.json', dict(dtype_counts))
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        train_rows, test_rows = select_rows(tokenizer, config, args.data_dir, run)
        if args.smoke:
            test_rows = test_rows[:config['smoke_eval_samples']]
        FastLanguageModel.for_inference(model)
        baseline, before = evaluate(model, tokenizer, test_rows, config, run, 'baseline', args.deadline)
        model = FastLanguageModel.get_peft_model(model, r=config['lora_rank'],
                lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                target_modules=config['target_modules'], bias='none',
                use_gradient_checkpointing=True, random_state=config['seed'])
        FastLanguageModel.for_training(model)
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in name for name, _ in trainable):
            raise RuntimeError('Only LoRA tensors may be trainable')
        expert_count = sum(p.numel() for name, p in trainable if '.experts.' in name)
        if not expert_count:
            raise RuntimeError('MoE expert adapters were not attached; refusing silent attention-only run')
        # Verify a real optimizer update using a small attention B tensor, initially zero.
        probe_name, probe = next((name, p) for name, p in trainable
                                if 'q_proj' in name and 'lora_B' in name)
        probe_before = probe.detach().float().cpu().clone()
        expert_probe_name, expert_probe = next((name, p) for name, p in trainable
                                              if '.experts.' in name and 'lora_B' in name)
        expert_probe_before = expert_probe.detach().float().cpu().clone()
        write_json(run / 'parameters.json', {
            'total': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for _, p in trainable),
            'trainable_expert_parameters': expert_count,
            'names_shapes': {name: list(p.shape) for name, p in trainable},
            'update_probe': probe_name,
            'expert_update_probe': expert_probe_name,
        })

        class Progress(TrainerCallback):
            def __init__(self):
                self.started = time.monotonic()
                self.stopped = False
                self.interrupted = False

            def on_step_end(self, training_args, state, control, **kwargs):
                if self.interrupted or time.time() >= args.deadline - config['final_evaluation_reserve_seconds']:
                    self.stopped = True
                    control.should_training_stop = True
                    control.should_save = not args.smoke
                event(run, 'training', step=state.global_step, max_steps=state.max_steps,
                      seconds=time.monotonic() - self.started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      stopping_for_deadline_or_signal=self.stopped)

            def on_log(self, training_args, state, control, logs=None, **kwargs):
                for name, value in (logs or {}).items():
                    if isinstance(value, (float, int)) and not math.isfinite(value):
                        raise FloatingPointError(f'Non-finite metric {name}={value}')
                row = dict(step=state.global_step, timestamp=time.time(), **(logs or {}))
                with (run / 'metrics.jsonl').open('a') as out:
                    out.write(json.dumps(row) + '\n')
                print('METRICS ' + json.dumps(row), flush=True)

        progress = Progress()
        signal.signal(signal.SIGTERM, lambda *_: setattr(progress, 'interrupted', True))
        training_args = GRPOConfig(
            output_dir=str(run / 'checkpoints'), num_train_epochs=config['epochs'],
            max_steps=config['smoke_steps'] if args.smoke else -1,
            per_device_train_batch_size=config['per_device_train_batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            generation_batch_size=config['generation_batch_size'], num_generations=config['num_generations'],
            max_prompt_length=config['max_prompt_length'], max_completion_length=config['max_completion_length'],
            learning_rate=config['learning_rate'], lr_scheduler_type='cosine', warmup_ratio=config['warmup_ratio'],
            max_grad_norm=config['max_grad_norm'], beta=config['beta'], loss_type=config['loss_type'],
            scale_rewards='group', temperature=config['temperature'], top_p=config['top_p'], top_k=config['top_k'],
            bf16=True, fp16=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False}, use_vllm=False, optim='adamw_torch',
            logging_steps=1, save_strategy='no' if args.smoke else 'steps',
            save_steps=config['save_steps'], save_total_limit=config['save_total_limit'], save_only_model=True,
            eval_strategy='no', report_to='none', seed=config['seed'], data_seed=config['seed'],
            dataloader_num_workers=0, remove_unused_columns=False, disable_tqdm=True)
        trainer = GRPOTrainer(model=model, args=training_args, reward_funcs=correctness_reward,
                              train_dataset=Dataset.from_list(train_rows), processing_class=tokenizer,
                              callbacks=[progress])
        if getattr(trainer, 'ref_model', None) is not None:
            raise RuntimeError('A separate reference model would exceed the memory plan')
        event(run, 'training_start', expected_steps=config['smoke_steps'] if args.smoke else expected_steps(config))
        progress.started = time.monotonic()
        output = trainer.train()
        training_seconds = time.monotonic() - progress.started
        delta = (probe.detach().float().cpu() - probe_before).abs().max().item()
        expert_delta = (expert_probe.detach().float().cpu() - expert_probe_before).abs().max().item()
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError('No finite nonzero LoRA update was observed')
        if not math.isfinite(expert_delta) or expert_delta <= 0:
            raise RuntimeError('No finite nonzero MoE expert LoRA update was observed')
        if not args.smoke:
            trainer.save_model(str(run / 'adapter'))
            tokenizer.save_pretrained(run / 'adapter')
            trainer.save_state()
        write_json(run / 'train_metrics.json', output.metrics)
        FastLanguageModel.for_inference(trainer.model)
        final, after = evaluate(trainer.model, tokenizer, test_rows, config, run, 'final', args.deadline)
        if [r['id'] for r in before] != [r['id'] for r in after]:
            raise RuntimeError('Evaluation pairing mismatch')
        requested_steps = config['smoke_steps'] if args.smoke else expected_steps(config)
        summary = dict(model=config['model_id'], algorithm='BF16 LoRA + GRPO', smoke=args.smoke,
                       completed=not progress.stopped and trainer.state.global_step == requested_steps,
                       optimizer_steps=trainer.state.global_step, baseline=baseline, final=final,
                       accuracy_delta_pp=100 * (final['accuracy'] - baseline['accuracy']),
                       wrong_to_right=sum(not a['correct'] and b['correct'] for a, b in zip(before, after)),
                       right_to_wrong=sum(a['correct'] and not b['correct'] for a, b in zip(before, after)),
                       adapter_updated=delta > 0, adapter_max_delta=delta,
                       expert_adapter_updated=expert_delta > 0, expert_adapter_max_delta=expert_delta,
                       trainable_expert_parameters=expert_count, training_seconds=training_seconds,
                       evaluation_seconds=baseline['seconds'] + final['seconds'],
                       gpu_total_gib=total_gib, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                       note='One seed; small paired test; no test checkpoint selection. Smoke is not a quality result.')
        write_json(run / 'summary.json', summary)
        event(run, 'complete' if summary['completed'] else 'incomplete', **summary)
        if not summary['completed']:
            raise RuntimeError('Stopped before all requested optimizer steps; see summary.json')
    except BaseException as error:
        event(run, 'failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
