"""One-hidden-layer ranker with optional metadata and pooled positive history."""
import torch
from torch import nn

from model import ShallowRecommender


class FeatureRecommender(nn.Module):
    def __init__(self, user_count, item_count, user_categorical, item_metadata,
                 mode="history", embedding_dim=64, hidden_dim=64):
        super().__init__()
        if mode not in ["metadata", "history"]:
            raise ValueError(mode)
        self.mode = mode
        self.user_embedding = nn.Embedding(user_count + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_count + 1, embedding_dim, padding_idx=0)
        self.gender_embedding = nn.Embedding(3, 4, padding_idx=0)
        self.age_embedding = nn.Embedding(8, 8, padding_idx=0)
        self.occupation_embedding = nn.Embedding(22, 8, padding_idx=0)
        self.register_buffer("user_categorical", torch.as_tensor(user_categorical, dtype=torch.long).clone())
        self.register_buffer("item_metadata", torch.as_tensor(item_metadata, dtype=torch.float32).clone())
        self.input_dim = 2 * embedding_dim + 4 + 8 + 8 + 19
        if mode == "history":
            self.input_dim += embedding_dim + 22
        self.hidden = nn.Linear(self.input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=.05)
                with torch.no_grad():
                    module.weight[0].zero_()

    def forward(self, users, items, histories=None, numeric=None):
        categorical = self.user_categorical[users]
        parts = [self.user_embedding(users), self.item_embedding(items),
                 self.gender_embedding(categorical[:, 0]), self.age_embedding(categorical[:, 1]),
                 self.occupation_embedding(categorical[:, 2]), self.item_metadata[items]]
        if self.mode == "history":
            if histories is None or numeric is None:
                raise ValueError("History model requires past-item IDs and history statistics")
            mask = (histories != 0).unsqueeze(-1)
            pooled = (self.item_embedding(histories) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            parts.extend([pooled, numeric])
        return self.output(torch.relu(self.hidden(torch.cat(parts, dim=1)))).squeeze(-1)


def make_model(config, user_categorical=None, item_metadata=None):
    if config["mode"] == "id":
        return ShallowRecommender(config["user_count"], config["item_count"],
                                  config["embedding_dim"], config["hidden_dim"])
    return FeatureRecommender(config["user_count"], config["item_count"], user_categorical, item_metadata,
                              config["mode"], config["embedding_dim"], config["hidden_dim"])


def restore(checkpoint):
    state = checkpoint["model_state"]
    model = make_model(checkpoint["config"], state.get("user_categorical"), state.get("item_metadata"))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
