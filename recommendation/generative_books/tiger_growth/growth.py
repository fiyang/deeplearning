"""Double T5 residual/FFN widths while preserving the FP32 evaluation function.

Attention heads and d_kv stay fixed. Tied output embeddings require the decoder's
final norm scale to be divided by sqrt(2), compensating T5's model_dim**-0.5.
Dropout is disabled for the equivalence check; independent training dropout breaks
the initial symmetry between copied channels. BF16 rounding need not be identical.
"""
import copy
import math

import torch
from transformers import T5ForConditionalGeneration


def expand_tiger(base):
    cfg = copy.deepcopy(base.net.config)
    if (cfg.d_model != 128 or cfg.d_ff != 1024 or cfg.feed_forward_proj != 'relu'
            or not cfg.tie_word_embeddings):
        raise ValueError('Only the validated tied-embedding 128/1024 ReLU TIGER can grow')
    grown = copy.deepcopy(base)
    cfg.d_model, cfg.d_ff = 256, 2048
    parameter = next(base.parameters())
    grown.net = T5ForConditionalGeneration(cfg).to(device=parameter.device, dtype=parameter.dtype)
    original, destination = base.net.state_dict(), grown.net.state_dict()
    mapped = {}
    for name, target in destination.items():
        value = original[name]
        if value.shape == target.shape:
            result = value.clone()
        elif name in ('shared.weight', 'encoder.embed_tokens.weight',
                      'decoder.embed_tokens.weight', 'lm_head.weight'):
            result = value.repeat(1, 2)
        elif value.ndim == 1 and ('layer_norm.weight' in name):
            result = value.repeat(2)
            if name == 'decoder.final_layer_norm.weight':
                result = result / math.sqrt(2)
        elif name.endswith(('Attention.q.weight', 'Attention.k.weight', 'Attention.v.weight')):
            result = value.repeat(1, 2) / 2
        elif name.endswith('Attention.o.weight'):
            result = value.repeat(2, 1)
        elif name.endswith(('DenseReluDense.wi.weight', 'DenseReluDense.wo.weight')):
            result = value.repeat(2, 2) / 2
        else:
            raise ValueError(f'Unmapped tensor: {name} {value.shape} -> {target.shape}')
        if result.shape != target.shape:
            raise ValueError(f'Bad expansion shape: {name}')
        mapped[name] = result
    grown.net.load_state_dict(mapped, strict=True)
    return grown


@torch.no_grad()
def check_equivalence(base, grown, history, users, targets):
    """Teacher-forced logits expose all vocabulary scores, including tied output."""
    base.eval()
    grown.eval()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        src = base.source(history, users)
        labels = base.item_tokens[targets + 1]
        a = base.net(input_ids=src, attention_mask=src.ne(0), labels=labels)
        b = grown.net(input_ids=src, attention_mask=src.ne(0), labels=labels)
        error = float((a.logits.float() - b.logits.float()).abs().max())
        torch.testing.assert_close(a.logits, b.logits, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(a.loss, b.loss, rtol=2e-5, atol=2e-5)
        return {'fp32_max_abs_logit_error': error, 'base_loss': float(a.loss),
                'grown_loss': float(b.loss), 'checked_requests': len(history)}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def check_initial_validation(parent, grown):
    """Predeclared BF16 gate: <=0.10 pp Recall@10, <=0.00025 NDCG@10 drift."""
    for key, tolerance in [('recall@10', .001), ('ndcg@10', .00025)]:
        if abs(parent[key] - grown[key]) > tolerance:
            raise RuntimeError(f'Initial BF16 validation drift too large: {key}')
