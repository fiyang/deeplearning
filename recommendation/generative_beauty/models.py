"""Small educational models; not an official TIGER implementation."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class RQVAE(nn.Module):
    def __init__(self, input_dim=768, k=256, levels=3, latent=32):
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


class SASRec(nn.Module):
    def __init__(self, n_items, width=128, layers=2, max_history=20, dropout=.1):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, width, padding_idx=0)
        self.pos = nn.Embedding(max_history, width)
        layer = nn.TransformerEncoderLayer(width, 4, 512, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item.weight, std=.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, history):
        length = history.ne(0).sum(-1)
        assert bool((length > 0).all())
        x = self.dropout(self.item(history) + self.pos(torch.arange(history.shape[1], device=history.device)))
        causal = torch.ones(history.shape[1], history.shape[1], device=history.device, dtype=torch.bool).triu(1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=history.eq(0))
        h = self.norm(x[torch.arange(len(x), device=x.device), length - 1])
        return h @ self.item.weight[1:].T


class Tiger(nn.Module):
    def __init__(self, codes, width=128, layers=4, dropout=.1, k=256, user_buckets=2000):
        super().__init__()
        self.k, self.user_buckets = k, user_buckets
        codes = torch.as_tensor(codes, dtype=torch.long)
        assert codes.ndim == 2 and codes.shape[1] == 4
        assert int(codes.min()) >= 0 and int(codes.max()) < k
        offset = 3 + torch.arange(4) * k
        table = torch.zeros(len(codes) + 1, 4, dtype=torch.long)
        table[1:] = codes + offset
        self.register_buffer("item_tokens", table)
        config = T5Config(vocab_size=3 + 4 * k + user_buckets, d_model=width,
                          d_kv=64, d_ff=1024, num_layers=layers,
                          num_decoder_layers=layers, num_heads=6,
                          dropout_rate=dropout, pad_token_id=0, decoder_start_token_id=1,
                          eos_token_id=2, feed_forward_proj="relu", use_cache=False)
        self.net = T5ForConditionalGeneration(config)
        self._build_trie(codes.cpu().numpy())

    def _build_trie(self, codes):
        """Last transition stores zero-based item IDs; intermediate IDs are local to depth."""
        prefixes = {(): 0}
        for depth in range(4):
            transitions = np.full((len(prefixes), self.k), -1, dtype=np.int64)
            next_prefixes = {}
            for item, code in enumerate(codes):
                prefix, child = tuple(code[:depth]), tuple(code[:depth + 1])
                if child not in next_prefixes:
                    next_prefixes[child] = len(next_prefixes)
                value = item if depth == 3 else next_prefixes[child]
                old = transitions[prefixes[prefix], code[depth]]
                if old != -1 and old != value:
                    raise ValueError("Semantic IDs must uniquely identify actual items")
                transitions[prefixes[prefix], code[depth]] = value
            self.register_buffer(f"transition_{depth}", torch.from_numpy(transitions))
            prefixes = next_prefixes

    def source(self, history, user_hash):
        return torch.cat([(3 + 4 * self.k + user_hash).unsqueeze(-1),
                          self.item_tokens[history].flatten(1)], -1)

    def forward(self, history, user_hash, targets):
        src = self.source(history, user_hash)
        labels = self.item_tokens[targets + 1]
        return self.net(input_ids=src, attention_mask=src.ne(0), labels=labels).loss

    @torch.no_grad()
    def recommend(self, history, user_hash, seen, beam_size=20):
        """Constrained beam search, preserving full-vocabulary log probabilities."""
        src = self.source(history, user_hash)
        src_mask = src.ne(0)
        memory = self.net.encoder(input_ids=src, attention_mask=src_mask).last_hidden_state
        batch = len(src)
        seq = torch.full((batch, 1, 1), 1, device=src.device, dtype=torch.long)
        scores = torch.zeros(batch, 1, device=src.device)
        nodes = torch.zeros(batch, 1, device=src.device, dtype=torch.long)
        for depth in range(4):
            beams = seq.shape[1]
            enc = BaseModelOutput(last_hidden_state=memory.repeat_interleave(beams, 0))
            logits = self.net(encoder_outputs=enc,
                              attention_mask=src_mask.repeat_interleave(beams, 0),
                              decoder_input_ids=seq.reshape(batch * beams, -1),
                              use_cache=False).logits[:, -1].float()
            logp = logits.log_softmax(-1)[:, 3 + depth * self.k:3 + (depth + 1) * self.k]
            logp = logp.reshape(batch, beams, self.k)
            children = getattr(self, f"transition_{depth}")[nodes.clamp_min(0)]
            valid = children.ge(0)
            if depth == 3:
                valid &= ~seen.gather(1, children.clamp_min(0).reshape(batch, -1)).reshape_as(children)
            joint = (scores.unsqueeze(-1) + logp).masked_fill(~valid, -torch.inf)
            top = min(beam_size, joint.shape[1] * self.k)
            scores, choice = joint.flatten(1).topk(top, -1)
            parent, token = choice // self.k, choice % self.k
            seq = torch.cat([seq.gather(1, parent.unsqueeze(-1).expand(-1, -1, seq.shape[-1])),
                             (3 + depth * self.k + token).unsqueeze(-1)], -1)
            nodes = children.flatten(1).gather(1, choice)
        return nodes.masked_fill(~torch.isfinite(scores), -1), scores
