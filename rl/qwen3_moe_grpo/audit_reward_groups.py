"""Recover binary four-completion group counts from archived GRPO metrics.

This recovers counts, never prompt IDs or completion text. It fails on ambiguous
counts rather than inventing a per-question history. Definitions match TRL 0.22.2.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path


def candidates(row):
    result = set()
    for successes in itertools.product(range(5), repeat=4):
        mean = sum(successes) / 16
        zero_fraction = sum(k in (0, 4) for k in successes) / 4
        # Unbiased sample std for k ones and 4-k zeros, averaged over groups.
        group_std = sum(math.sqrt(k * (4 - k) / 12) for k in successes) / 4
        # Two generation batches of 8 completions per optimizer update.
        batch_std = sum(math.sqrt(k * (8 - k) / 56)
                        for k in [sum(successes[:2]), sum(successes[2:])]) / 2
        expected = dict(reward=mean, frac_reward_zero_std=zero_fraction,
                        reward_std=group_std)
        if 'rewards/correctness_reward/std' in row:
            expected['rewards/correctness_reward/std'] = batch_std
        if all(math.isclose(row[key], value, abs_tol=1e-6, rel_tol=1e-6)
               for key, value in expected.items()):
            result.add(tuple(successes.count(k) for k in range(5)))
    return sorted(result)


def summarize(counts):
    total = sum(counts)
    categories = dict(all_correct=counts[4], all_zero_reward=counts[0],
                      mixed=sum(counts[1:4]))
    return dict(total_prompt_visits=total, correct_completions=sum(k * n for k, n in enumerate(counts)),
                group_counts_by_correct_completions=dict(enumerate(counts)),
                categories={key: dict(count=n, percent=100 * n / total)
                            for key, n in categories.items()})


def audit(metrics_path, config_path):
    config = json.loads(Path(config_path).read_text())
    required = dict(num_generations=4, generation_batch_size=8,
                    per_device_train_batch_size=1, gradient_accumulation_steps=16)
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('This reconstruction requires the verified single-GPU batch layout')
    if config['train_samples'] % 4:
        raise ValueError('Partial optimizer batch is unsupported')
    steps_per_epoch = config['train_samples'] // 4
    with Path(metrics_path).open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    rows = [row for row in rows if 'reward' in row]
    if [r['step'] for r in rows] != list(range(1, steps_per_epoch * config['epochs'] + 1)):
        raise ValueError('Expected all optimizer steps exactly once')
    totals, epoch_counts, per_step = [0] * 5, {}, []
    for row in rows:
        matches = candidates(row)
        if len(matches) != 1:
            raise ValueError(f"Step {row['step']}: {len(matches)} possible count vectors")
        counts = matches[0]
        epoch = (row['step'] - 1) // steps_per_epoch + 1
        accumulator = epoch_counts.setdefault(epoch, [0] * 5)
        for k, count in enumerate(counts):
            totals[k] += count
            accumulator[k] += count
        per_step.append(dict(step=row['step'], epoch=epoch, counts=list(counts)))
    return dict(
        method='Unique integer reconstruction from binary reward mean, group std, zero-std fraction and generation-batch std',
        logging_reference='https://github.com/huggingface/trl/blob/v0.22.2/trl/trainer/grpo_trainer.py',
        assumptions='Single GPU; 4 completions per prompt; 4 prompt visits per optimizer update; binary correctness reward; equal-size generation batches',
        source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in [metrics_path, config_path]},
        optimizer_steps=len(rows), unique_training_questions=config['train_samples'],
        count_unit='Prompt visits, not distinct question IDs; each question is visited once per epoch',
        overall=summarize(totals), epochs={str(k): summarize(v) for k, v in epoch_counts.items()},
        prompt_identity_recoverable=False, completion_text_available=False,
        text_duplicate_rate=None,
        limitations=['Reward ties do not imply identical texts.',
                     'Zero reward includes malformed final answers as well as wrong numeric answers.',
                     'Exact question identities and epoch-to-epoch transitions cannot be recovered.'],
        per_step=per_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.metrics, args.config)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['overall', 'epochs']}, ensure_ascii=False, indent=2))
