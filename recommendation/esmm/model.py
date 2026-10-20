"""ESMM and an independently trained CTR + clicked-only CVR reference."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def product_bce_logits(ctr_logit, cvr_logit, joint_label):
    # log(1 - a*b) = log((1-a) + a*(1-b)); stable even at extreme logits.
    log_yes = F.logsigmoid(ctr_logit) + F.logsigmoid(cvr_logit)
    log_no = torch.logaddexp(F.logsigmoid(-ctr_logit),
                            F.logsigmoid(ctr_logit) + F.logsigmoid(-cvr_logit))
    return -(joint_label * log_yes + (1 - joint_label) * log_no).mean()


class FieldEncoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=.01)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, categories, tags):
        ordinary = self.embedding(categories).flatten(1)
        tag_sum = self.embedding(tags).sum(1)
        return torch.cat((ordinary, tag_sum), dim=1)


def tower(input_dim, hidden, dropout):
    layers = []
    for width in hidden:
        layers += [nn.Linear(input_dim, width), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        input_dim = width
    layers.append(nn.Linear(input_dim, 2))
    return nn.Sequential(*layers)


class FunnelModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kind = config["kind"]
        assert self.kind in ("esmm", "independent")
        self.encoder = FieldEncoder(config["vocabulary_size"], config["embedding_dim"])
        if self.kind == "independent":
            self.cvr_encoder = FieldEncoder(config["vocabulary_size"], config["embedding_dim"])
            self.cvr_encoder.load_state_dict(self.encoder.state_dict())
        self.input_dim = (config["category_count"] + 1) * config["embedding_dim"]
        self.ctr = tower(self.input_dim, config["hidden"], config["dropout"])
        self.cvr = tower(self.input_dim, config["hidden"], config["dropout"])
        for head, rate in [(self.ctr, config["ctr_prior"]), (self.cvr, config["cvr_prior"])]:
            rate = min(max(rate, 1e-6), 1 - 1e-6)
            with torch.no_grad():
                head[-1].bias.copy_(torch.tensor([0., math.log(rate / (1 - rate))]))

    def forward(self, categories, tags):
        x = self.encoder(categories, tags)
        cvr_x = self.cvr_encoder(categories, tags) if self.kind == "independent" else x
        ctr = self.ctr(x); cvr = self.cvr(cvr_x)
        return ctr[:, 1] - ctr[:, 0], cvr[:, 1] - cvr[:, 0]

    def loss(self, logits, labels):
        a, b = logits
        click, like = labels.unbind(1)
        ctr_loss = F.binary_cross_entropy_with_logits(a, click)
        if self.kind == "esmm":
            other = product_bce_logits(a, b, click * like)
        else:
            clicked = click == 1
            other = F.binary_cross_entropy_with_logits(b[clicked], like[clicked]) if clicked.any() else b.sum() * 0
        return ctr_loss + other

    def probabilities(self, categories, tags):
        a, b = self(categories, tags)
        ctr, cvr = a.sigmoid(), b.sigmoid()
        return torch.stack((ctr, cvr, ctr * cvr), dim=1)
