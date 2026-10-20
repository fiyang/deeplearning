"""Field self-attention before the shared-input ESMM towers.

This is a field-interaction experiment, not a chronological behavior encoder.
The original weighted, untruncated within-field pooling is retained.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from base_model import MixedWidthEmbedding, make_tower, product_bce_logits


def present_fields(part, field_count):
    """Presence depends on known, nonzero-weight tokens, not embedding values."""
    tokens, offsets, weights = part
    valid = (tokens != 0) & (weights != 0)
    prefix = torch.cat([offsets.new_zeros(1), valid.cumsum(0)])
    return (prefix[offsets[1:].long()] > prefix[offsets[:-1].long()]).reshape(-1, field_count)


class FieldAttention(nn.Module):
    def __init__(self, field_dims, width=64, heads=4, dropout=0.1):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError('Attention width must be divisible by number of heads')
        self.field_dims = list(field_dims)
        # Field-specific projections distinguish semantic fields without time positions.
        self.projections = nn.ModuleList([nn.Linear(d, width, bias=False) for d in field_dims])
        self.pre_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.residual_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, pooled, present):
        parts = pooled.split(self.field_dims, dim=-1)
        x = torch.stack([projection(part) for projection, part in zip(self.projections, parts)], dim=1)
        x = x.masked_fill(~present.unsqueeze(-1), 0)
        # MultiheadAttention must have at least one unmasked key, even for an
        # entirely missing example. Its dummy output is explicitly discarded.
        safe_present = present.clone()
        safe_present[:, 0] |= ~present.any(dim=1)
        normalized = self.pre_norm(x)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     key_padding_mask=~safe_present, need_weights=False)
        output = self.output_norm(x + self.residual_dropout(attended))
        return output.masked_fill(~present.unsqueeze(-1), 0)


class Model(nn.Module):
    def __init__(self, config, vocabulary=None):
        super().__init__()
        if config['kind'] != 'esmm' or config.get('attention_layers', 1) != 1:
            raise ValueError('This experiment supports ESMM with one attention layer')
        self.config = config
        self.fields = config['field_count']
        field_dims = [config['id_embedding_dim'] if field in ['101', '205'] else config['embedding_dim']
                      for field in config['fields']]
        self.pooled_dim = sum(field_dims)
        self.input_dim = self.fields * config['attention_dim']
        self.embedding = MixedWidthEmbedding(config, vocabulary)
        self.feature_encoder = FieldAttention(field_dims, config['attention_dim'],
                                              config['attention_heads'], config['attention_dropout'])
        self.ctr = make_tower(self.input_dim, config['hidden'], config['dropout'])
        self.cvr = make_tower(self.input_dim, config['hidden'], config['dropout'])
        for tower, rate in [(self.ctr, config['ctr_prior']), (self.cvr, config['cvr_prior'])]:
            rate = min(max(rate, 1e-7), 1 - 1e-7)
            with torch.no_grad():
                tower[-1].bias.copy_(torch.tensor([0., math.log(rate / (1 - rate))]))

    def encode(self, common, sample):
        pooled = self.embedding(common, sample)
        present = present_fields(common, self.fields) | present_fields(sample, self.fields)
        return self.feature_encoder(pooled, present)

    def forward(self, common, sample):
        features = self.encode(common, sample).flatten(1)
        a, b = self.ctr(features), self.cvr(features)
        return a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]

    def loss(self, logits, labels):
        a, b = logits
        click, conversion = labels.unbind(1)
        return F.binary_cross_entropy_with_logits(a, click) + product_bce_logits(a, b, conversion)

    def probabilities(self, common, sample):
        a, b = self(common, sample)
        p, q = a.sigmoid(), b.sigmoid()
        return torch.stack([p, q, p * q], dim=1)
