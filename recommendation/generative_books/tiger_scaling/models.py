"""Controlled small generative recommenders, not official paper reproductions."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model, T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class RQVAE(nn.Module):
    def __init__(self, input_dim=768, k=512, levels=3, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(),
                                     nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128),
                                     nn.ReLU(), nn.Linear(128, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(),
                                     nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 512),
                                     nn.ReLU(), nn.Linear(512, input_dim))
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(k, latent) * .01)
                                          for _ in range(levels)])

    def quantize(self, z):
        residual, quantized, loss, codes = z, torch.zeros_like(z), z.new_zeros(()), []
        for cb in self.codebooks:
            dist = residual.square().sum(-1, keepdim=True) + cb.square().sum(-1) - 2 * residual @ cb.T
            idx = dist.argmin(-1)
            q = cb[idx]
            loss = loss + F.mse_loss(q, residual.detach()) + .25 * F.mse_loss(residual, q.detach())
            residual = residual - q.detach()
            quantized = quantized + q
            codes.append(idx)
        return quantized, loss, torch.stack(codes, -1)

    def forward(self, x):
        z = self.encoder(x)
        q, qloss, codes = self.quantize(z)
        rec = F.mse_loss(self.decoder(z + (q - z).detach()), x)
        return rec + qloss, rec, codes


class SemanticGenerator(nn.Module):
    """Shared vocabulary and sparse legal-prefix index for both architectures."""
    def __init__(self, codes, k=512, user_buckets=2000, collision_size=None):
        super().__init__()
        self.sizes = (k, k, k, k if collision_size is None else collision_size)
        self.token_offsets = (3, 3+k, 3+2*k, 3+3*k)
        self.user_offset = 3 + sum(self.sizes)
        codes = np.asarray(codes, dtype=np.int64)
        if (codes.ndim != 2 or codes.shape[1] != 4 or codes.min() < 0
                or bool((codes >= np.asarray(self.sizes)).any())):
            raise ValueError("Expected four unique bounded codes per item")
        if len(np.unique(codes, axis=0)) != len(codes):
            raise ValueError("Semantic IDs must identify real items uniquely")
        self.k, self.user_buckets = k, user_buckets
        self.vocab_size = self.user_offset + user_buckets
        table = np.zeros((len(codes) + 1, 4), dtype=np.int64)
        table[1:] = codes + np.asarray(self.token_offsets)
        self.register_buffer("item_tokens", torch.from_numpy(table), persistent=False)
        prefix = np.zeros(len(codes), dtype=np.int64)
        for depth in range(4):
            prefix = prefix * self.sizes[depth] + codes[:, depth]
            self.register_buffer(f"legal_{depth}", torch.from_numpy(np.unique(prefix)), persistent=False)
            self.register_buffer(f"choices_{depth}", torch.arange(self.sizes[depth]), persistent=False)
        self.register_buffer("leaf_items", torch.from_numpy(np.argsort(prefix)), persistent=False)


    def source(self, history, user_hash):
        return torch.cat([(self.user_offset + user_hash).unsqueeze(-1),
                          self.item_tokens[history].flatten(1)], -1)

    def select(self, logp, scores, prefixes, seen, depth, beam_size):
        batch, beams = prefixes.shape
        size = self.sizes[depth]
        keys = prefixes.unsqueeze(-1) * size + getattr(self, f"choices_{depth}")
        legal = getattr(self, f"legal_{depth}")
        index = torch.searchsorted(legal, keys)
        safe = index.clamp_max(len(legal) - 1)
        valid = (index < len(legal)) & legal[safe].eq(keys)
        if depth == 3:
            item = self.leaf_items[safe]
            valid &= ~seen.gather(1, item.reshape(batch, -1)).reshape_as(item)
        joint = (scores.unsqueeze(-1) + logp).masked_fill(~valid, -torch.inf)
        scores, selected = joint.flatten(1).topk(min(beam_size, beams * size), -1)
        parent, token = selected // size, selected % size
        prefixes = keys.flatten(1).gather(1, selected)
        if depth == 3:
            item = item.flatten(1).gather(1, selected).masked_fill(~torch.isfinite(scores), -1)
            return parent, token, prefixes, scores, item
        return parent, token, prefixes, scores, None


class Tiger(SemanticGenerator):
    def __init__(self, codes, k=512, user_buckets=2000, collision_size=None,
                 d_model=256, d_ff=2048):
        super().__init__(codes, k, user_buckets, collision_size)
        cfg = T5Config(vocab_size=self.vocab_size, d_model=d_model, d_kv=64, d_ff=d_ff,
                       num_layers=4, num_decoder_layers=4, num_heads=6, dropout_rate=.1,
                       pad_token_id=0, decoder_start_token_id=1, eos_token_id=2,
                       feed_forward_proj="relu", use_cache=False)
        self.net = T5ForConditionalGeneration(cfg)

    def forward(self, history, user_hash, targets):
        src = self.source(history, user_hash)
        return self.net(input_ids=src, attention_mask=src.ne(0),
                        labels=self.item_tokens[targets + 1]).loss

    @torch.no_grad()
    def recommend(self, history, user_hash, seen, beam_size=20):
        src = self.source(history, user_hash)
        mask = src.ne(0)
        memory = self.net.encoder(input_ids=src, attention_mask=mask).last_hidden_state
        batch = len(src)
        seq = torch.full((batch, 1, 1), 1, device=src.device, dtype=torch.long)
        scores = torch.zeros(batch, 1, device=src.device)
        prefixes = torch.zeros(batch, 1, device=src.device, dtype=torch.long)
        for depth in range(4):
            beams = seq.shape[1]
            enc = BaseModelOutput(last_hidden_state=memory.repeat_interleave(beams, 0))
            logits = self.net(encoder_outputs=enc, attention_mask=mask.repeat_interleave(beams, 0),
                              decoder_input_ids=seq.reshape(batch * beams, -1), use_cache=False).logits[:, -1].float()
            offset, size = self.token_offsets[depth], self.sizes[depth]
            logp = logits.log_softmax(-1)[:, offset:offset+size]
            parent, token, prefixes, scores, item = self.select(
                logp.reshape(batch, beams, size), scores, prefixes, seen, depth, beam_size)
            seq = torch.cat([seq.gather(1, parent.unsqueeze(-1).expand(-1, -1, seq.shape[-1])),
                             (offset + token).unsqueeze(-1)], -1)
        return item, scores


class DecoderOnly(SemanticGenerator):
    def __init__(self, codes, k=512, user_buckets=2000, collision_size=None):
        super().__init__(codes, k, user_buckets, collision_size)
        cfg = GPT2Config(vocab_size=self.vocab_size, n_positions=256, n_embd=256,
                         n_layer=5, n_head=4, n_inner=1024, resid_pdrop=.1, embd_pdrop=.1,
                         attn_pdrop=.1, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                         use_cache=True, attn_implementation="sdpa")
        self.net = GPT2Model(cfg)

    def prefix(self, history, user_hash):
        src = self.source(history, user_hash)
        return torch.cat([src, torch.ones(len(src), 1, device=src.device, dtype=torch.long)], -1)

    def forward(self, history, user_hash, targets):
        labels = self.item_tokens[targets + 1]
        prefix = self.prefix(history, user_hash)
        inputs = torch.cat([prefix, labels[:, :-1]], -1)
        mask = inputs.ne(0)
        position = (mask.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.net(input_ids=inputs, attention_mask=mask, position_ids=position,
                          use_cache=False).last_hidden_state[:, -4:]
        logits = F.linear(hidden, self.net.wte.weight)
        return F.cross_entropy(logits.float().reshape(-1, self.vocab_size), labels.flatten())

    @torch.no_grad()
    def recommend(self, history, user_hash, seen, beam_size=20):
        src = self.prefix(history, user_hash)
        mask = src.ne(0)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        output = self.net(input_ids=src, attention_mask=mask, position_ids=positions, use_cache=True)
        cache = output.past_key_values
        logits = F.linear(output.last_hidden_state[:, -1], self.net.wte.weight).float()
        batch, beams = len(src), 1
        scores = torch.zeros(batch, 1, device=src.device)
        prefixes = torch.zeros(batch, 1, device=src.device, dtype=torch.long)
        for depth in range(4):
            offset, size = self.token_offsets[depth], self.sizes[depth]
            logp = logits.log_softmax(-1)[:, offset:offset+size]
            parent, token, prefixes, scores, item = self.select(
                logp.reshape(batch, beams, size), scores, prefixes, seen, depth, beam_size)
            if depth == 3:
                break
            parent_flat = (parent + torch.arange(batch, device=src.device)[:, None] * beams).flatten()
            if hasattr(cache, "reorder_cache"):
                cache.reorder_cache(parent_flat)
            else:
                cache = tuple(tuple(t.index_select(0, parent_flat) for t in layer) for layer in cache)
            mask = mask.index_select(0, parent_flat)
            position = mask.long().sum(-1, keepdim=True)
            mask = torch.cat([mask, torch.ones(len(mask), 1, device=src.device, dtype=torch.bool)], -1)
            selected = (offset + token).reshape(-1, 1)
            output = self.net(input_ids=selected, attention_mask=mask, position_ids=position,
                              past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = F.linear(output.last_hidden_state[:, -1], self.net.wte.weight).float()
            beams = parent.shape[1]
        return item, scores


def create_model(name, codes):
    if name == "tiger":
        return Tiger(codes, collision_size=1024)
    if name == "decoder_only":
        return DecoderOnly(codes, collision_size=1024)
    raise ValueError(name)
