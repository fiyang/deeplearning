"""Full-feature MLP, explicit interactions, and a shared-embedding DeepFM variant."""
import torch
from torch import nn

from feature_model import FeatureRecommender


def fm_second_order(weighted_embeddings):
    """Sum <v_i*x_i, v_j*x_j> over all distinct active feature pairs."""
    return .5 * (weighted_embeddings.sum(dim=1).square()
                 - weighted_embeddings.square().sum(dim=1)).sum(dim=-1)


class InteractionMLP(FeatureRecommender):
    def __init__(self, user_count, item_count, user_categorical, item_metadata,
                 embedding_dim=64, hidden_dim=64):
        super().__init__(user_count, item_count, user_categorical, item_metadata,
                         "history", embedding_dim, hidden_dim)
        old_hidden = self.hidden
        self.base_input_dim = self.input_dim
        self.input_dim += 2 * embedding_dim + 2
        self.hidden = nn.Linear(self.input_dim, hidden_dim)
        # Identical initial predictions to the baseline under the same seed.
        # Extra interaction weights are learned from zero.
        with torch.no_grad():
            self.hidden.weight.zero_()
            self.hidden.weight[:, :self.base_input_dim].copy_(old_hidden.weight)
            self.hidden.bias.copy_(old_hidden.bias)

    def forward(self, users, items, histories, numeric):
        categories = self.user_categorical[users]
        u, i = self.user_embedding(users), self.item_embedding(items)
        mask = (histories != 0).unsqueeze(-1)
        h = (self.item_embedding(histories) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        user_item, history_item = u * i, h * i
        parts = [u, i, self.gender_embedding(categories[:, 0]), self.age_embedding(categories[:, 1]),
                 self.occupation_embedding(categories[:, 2]), self.item_metadata[items], h, numeric,
                 user_item, history_item, user_item.sum(dim=1, keepdim=True),
                 history_item.sum(dim=1, keepdim=True)]
        return self.output(torch.relu(self.hidden(torch.cat(parts, dim=1)))).squeeze(-1)


class DeepFM(nn.Module):
    """Linear + second-order FM + one-hidden-layer MLP, sharing all latent factors.

    Five categorical fields and 41 numeric/multi-hot fields use 64-d factors.
    Past positive movie IDs share candidate-movie factors. Their weights sum to
    one. FM sees the individual weighted history features; the deep branch sees
    their sum, keeping the input size independent of history length.
    This is an adaptation to our MovieLens features, not paper hyperparameters.
    """
    def __init__(self, user_count, item_count, user_categorical, item_metadata,
                 embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.register_buffer("user_categorical", torch.as_tensor(user_categorical, dtype=torch.long).clone())
        self.register_buffer("item_metadata", torch.as_tensor(item_metadata, dtype=torch.float32).clone())
        counts = {"user": user_count + 1, "item": item_count + 1, "gender": 3, "age": 8, "occupation": 22}
        for name, count in counts.items():
            setattr(self, f"{name}_embedding", nn.Embedding(count, embedding_dim, padding_idx=0))
            setattr(self, f"{name}_linear", nn.Embedding(count, 1, padding_idx=0))
        self.numeric_embedding = nn.Parameter(torch.empty(41, embedding_dim))
        self.numeric_linear = nn.Parameter(torch.zeros(41))
        self.bias = nn.Parameter(torch.zeros(()))
        self.input_dim = (5 + 1 + 41) * embedding_dim
        self.hidden = nn.Linear(self.input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                if name.endswith("_linear"):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.normal_(module.weight, std=.05)
                with torch.no_grad():
                    module.weight[0].zero_()
        nn.init.normal_(self.numeric_embedding, std=.05)

    def components(self, users, items, histories, numeric):
        categories = self.user_categorical[users]
        values = torch.cat([self.item_metadata[items], numeric], dim=1)
        indexes = {"user": users, "item": items, "gender": categories[:, 0],
                   "age": categories[:, 1], "occupation": categories[:, 2]}
        categorical = torch.stack([getattr(self, f"{name}_embedding")(index)
                                   for name, index in indexes.items()], dim=1)
        numerical = values.unsqueeze(-1) * self.numeric_embedding.unsqueeze(0)
        history_mask = (histories != 0).to(values.dtype)
        history_weights = history_mask / history_mask.sum(dim=1, keepdim=True).clamp_min(1)
        history_vectors = self.item_embedding(histories) * history_weights.unsqueeze(-1)
        fm = fm_second_order(torch.cat([categorical, numerical, history_vectors], dim=1))
        linear = self.bias + (values * self.numeric_linear).sum(dim=1)
        for name, index in indexes.items():
            linear = linear + getattr(self, f"{name}_linear")(index).squeeze(-1)
        linear = linear + (self.item_linear(histories).squeeze(-1) * history_weights).sum(dim=1)
        deep_input = torch.cat([categorical.flatten(1), history_vectors.sum(dim=1), numerical.flatten(1)], dim=1)
        deep = self.output(torch.relu(self.hidden(deep_input))).squeeze(-1)
        return linear, fm, deep

    def forward(self, users, items, histories, numeric):
        linear, fm, deep = self.components(users, items, histories, numeric)
        return linear + fm + deep


def make_auc_model(config, user_categorical, item_metadata):
    args = (config["user_count"], config["item_count"], user_categorical, item_metadata)
    kwargs = dict(embedding_dim=config["embedding_dim"], hidden_dim=config["hidden_dim"])
    if config["architecture"] == "mlp":
        return FeatureRecommender(*args, mode="history", **kwargs)
    if config["architecture"] == "interaction":
        return InteractionMLP(*args, **kwargs)
    if config["architecture"] == "deepfm":
        return DeepFM(*args, **kwargs)
    raise ValueError(config["architecture"])


def restore_auc(checkpoint):
    state = checkpoint["model_state"]
    model = make_auc_model(checkpoint["config"], state["user_categorical"], state["item_metadata"])
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
