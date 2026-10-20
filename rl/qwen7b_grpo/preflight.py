"""Synthetic integration check only; these scores are NOT experiment results."""
import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from tokenizers import Tokenizer, models, pre_tokenizers
from trl import GRPOConfig, GRPOTrainer

root = Path(__file__).resolve().parent
vocab = {word: i for i, word in enumerate(
    ['<pad>', '<eos>', '<unk>', '<bos>', 'user', 'assistant', ':', 'Compute',
     '2', '+', '.', '4', 'The', 'answer', 'is', '<answer>', '</answer>'])}
backend = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
backend.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
    eos_token='<eos>', unk_token='<unk>', bos_token='<bos>', padding_side='left')
tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
model = Qwen2ForCausalLM(Qwen2Config(
    vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id)).to(device='cuda', dtype=torch.bfloat16)
dataset = Dataset.from_list([{'prompt': [{'role': 'user', 'content': 'Compute 2+2.'}]} for _ in range(4)])

def synthetic_reward(completions, **kwargs):
    return [float(i % 2) for i in range(len(completions))]

trainer = GRPOTrainer(
    model=model, processing_class=tokenizer, train_dataset=dataset, reward_funcs=synthetic_reward,
    peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'),
    args=GRPOConfig(output_dir=str(root/'runs'/'preflight'), max_steps=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, num_generations=4,
        max_prompt_length=64, max_completion_length=16, beta=0.04, loss_type='grpo',
        learning_rate=1e-4, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        use_vllm=False, report_to='none', logging_steps=1, save_strategy='no', disable_tqdm=True))
before = {n:p.detach().clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
result = trainer.train()
delta = sum((p.detach()-before[n]).float().abs().sum().item()
            for n,p in trainer.model.named_parameters() if n in before)
assert delta > 0, 'Adapters did not update'
assert torch.isfinite(torch.tensor(result.metrics['train_loss'])), 'Non-finite loss'
summary = dict(synthetic_integration_check=True, adapter_update_l1=delta,
               gpu=torch.cuda.get_device_name(), torch=torch.__version__, metrics=result.metrics)
(root/'runs'/'preflight_result.json').write_text(json.dumps(summary, indent=2))
print('PREFLIGHT_PASSED ' + json.dumps(summary), flush=True)
