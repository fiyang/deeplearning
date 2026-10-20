"""PyTorch port of xidongbo/AITM's two-task AITM.py (MIT; see official/).

Outputs are p(click|exposure) and p(purchase|exposure), NOT a CVR tower.
Unknown row 0 is trainable, as in the author's embedding_lookup.
"""
import math
import torch
from torch import nn


class AITM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList([nn.Embedding(n, config['embedding_dim'])
                                         for n in config['vocabulary_sizes']])
        width = len(self.embeddings) * config['embedding_dim']
        self.click_tower = self.make_tower(width)
        self.purchase_tower = self.make_tower(width)
        self.transfer = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(.3))
        self.attention = nn.ParameterList([nn.Parameter(torch.empty(32, 32)) for _ in range(3)])
        self.click_head = nn.Linear(32, 1)
        self.purchase_head = nn.Linear(32, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for table in self.embeddings:
            nn.init.normal_(table.weight, std=.01)
        for matrix in self.attention:
            nn.init.normal_(matrix, std=.01)

    @staticmethod
    def make_tower(width):
        layers = []
        for hidden, dropout in zip([128, 64, 32], [.1, .3, .3]):
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden
        return nn.Sequential(*layers)

    def aggregate(self, purchase, info):
        inputs = torch.stack([purchase, info], dim=1)
        q, k, v = [inputs @ matrix for matrix in self.attention]
        # The author's attention is a two-way gate, not a Transformer QK^T.
        weights = ((q * k).sum(-1) / math.sqrt(32)).softmax(dim=1)
        return (weights.unsqueeze(-1) * v).sum(dim=1)

    def forward(self, ids):
        features = torch.cat([table(ids[:, i].long()) for i, table in enumerate(self.embeddings)], dim=1)
        click = self.click_tower(features)
        purchase = self.purchase_tower(features)
        mixed = self.aggregate(purchase, self.transfer(click))
        return torch.cat([self.click_head(click), self.purchase_head(mixed)], dim=1)

    def probabilities(self, ids):
        return self(ids).sigmoid()

    def loss_parts(self, logits, labels):
        probabilities = logits.sigmoid()
        # tf.losses.log_loss uses epsilon inside each logarithm, default 1e-7.
        bce = -(labels * torch.log(probabilities + 1e-7)
                + (1 - labels) * torch.log(1 - probabilities + 1e-7)).mean(0).sum()
        constraint = self.config['constraint_weight'] * torch.relu(probabilities[:, 1] - probabilities[:, 0]).mean()
        # In AITM.py only embedding tables and the three attention matrices
        # receive tf.contrib.layers.l2_regularizer; Dense kernels do not.
        regularized = [table.weight for table in self.embeddings] + list(self.attention)
        regularizer = .5 * self.config['l2'] * sum(x.square().sum() for x in regularized)
        return dict(bce=bce, constraint=constraint, regularizer=regularizer)

    def loss(self, logits, labels):
        return sum(self.loss_parts(logits, labels).values())


class TF1Adam(torch.optim.Optimizer):
    """Dense Adam with TF1's epsilon placement, including embedding gradients."""
    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, step=0))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError('Closure is not used by this experiment')
        for group in self.param_groups:
            group['step'] += 1
            beta1, beta2 = group['betas']
            rate = group['lr'] * math.sqrt(1 - beta2 ** group['step']) / (1 - beta1 ** group['step'])
            params, gradients, first, second = [], [], [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError('Expected dense gradient (L2 covers every embedding row)')
                state = self.state[p]
                if not state:
                    state['m'], state['v'] = torch.zeros_like(p), torch.zeros_like(p)
                params.append(p); gradients.append(p.grad)
                first.append(state['m']); second.append(state['v'])
            torch._foreach_mul_(first, beta1)
            torch._foreach_add_(first, gradients, alpha=1 - beta1)
            torch._foreach_mul_(second, beta2)
            torch._foreach_addcmul_(second, gradients, gradients, value=1 - beta2)
            denominator = torch._foreach_sqrt(second)
            torch._foreach_add_(denominator, group['eps'])
            torch._foreach_addcdiv_(params, first, denominator, value=-rate)
