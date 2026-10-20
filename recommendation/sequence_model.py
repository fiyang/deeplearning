"""Mean/Transformer sequence features appended to a warm-started DeepFM ranker."""
import torch
from torch import nn

from auc_models import DeepFM, fm_second_order
from sequence_data import TIME_BUCKETS


class SequenceRanker(DeepFM):
    def __init__(self, user_count, item_count, user_categorical, item_metadata,
                 encoder="transformer", embedding_dim=64, hidden_dim=64,
                 sequence_length=50, heads=4, layers=2, ff_dim=128, dropout=.1):
        super().__init__(user_count, item_count, user_categorical, item_metadata, embedding_dim, hidden_dim)
        if encoder not in ["mean", "transformer"]:
            raise ValueError(encoder)
        self.encoder_kind = encoder
        self.base_input_dim = self.input_dim
        self.input_dim += embedding_dim
        old_hidden = self.hidden
        self.hidden = nn.Linear(self.input_dim, hidden_dim)
        with torch.no_grad():
            self.hidden.weight.zero_()
            self.hidden.weight[:, :self.base_input_dim].copy_(old_hidden.weight)
            self.hidden.bias.copy_(old_hidden.bias)
        self.rating_embedding = nn.Embedding(6, embedding_dim, padding_idx=0)
        self.time_embedding = nn.Embedding(TIME_BUCKETS, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(sequence_length + 1, embedding_dim, padding_idx=0)
        self.movie_feature_projection = nn.Linear(19, embedding_dim, bias=False)
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.token_dropout = nn.Dropout(dropout)
        for module in [self.rating_embedding, self.time_embedding, self.position_embedding]:
            nn.init.normal_(module.weight, std=.02)
            with torch.no_grad():
                module.weight[0].zero_()
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads, dim_feedforward=ff_dim,
                                        dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)
        ] if encoder == "transformer" else [])
        self.interest_norm = nn.LayerNorm(embedding_dim)

    def warm_start(self, checkpoint):
        current = self.state_dict()
        for name, value in checkpoint["model_state"].items():
            if name == "hidden.weight":
                assert value.shape[1] == self.base_input_dim
                current[name].zero_()
                current[name][:, :self.base_input_dim].copy_(value)
            else:
                assert current[name].shape == value.shape, name
                current[name].copy_(value)
        self.load_state_dict(current, strict=True)

    def extract_interest(self, sequence_items, ratings, time_buckets, positions):
        sequence_items = sequence_items.long()
        valid = sequence_items != 0
        x = (self.item_embedding(sequence_items) + self.rating_embedding(ratings.long())
             + self.time_embedding(time_buckets.long()) + self.position_embedding(positions.long())
             + self.movie_feature_projection(self.item_metadata[sequence_items]))
        x = self.token_dropout(self.token_norm(x))
        x = x * valid.unsqueeze(-1)
        # Unmask a zero dummy token for empty windows to prevent all-masked softmax NaNs.
        padding_mask = ~valid
        padding_mask = padding_mask.clone()
        padding_mask[~valid.any(dim=1), 0] = False
        # All input events are strictly in the past. Bidirectional attention
        # within this known history is valid; no target/held-out token is included.
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        interest = self.interest_norm(pooled)
        return interest * valid.any(dim=1, keepdim=True)

    def score_with_interest(self, users, items, positive_histories, numeric, interest):
        categories = self.user_categorical[users]
        values = torch.cat([self.item_metadata[items], numeric], dim=1)
        indexes = {"user": users, "item": items, "gender": categories[:, 0],
                   "age": categories[:, 1], "occupation": categories[:, 2]}
        categorical = torch.stack([getattr(self, f"{name}_embedding")(index) for name, index in indexes.items()], dim=1)
        numerical = values.unsqueeze(-1) * self.numeric_embedding.unsqueeze(0)
        mask = (positive_histories != 0).to(values.dtype)
        weights = mask / mask.sum(dim=1, keepdim=True).clamp_min(1)
        vectors = self.item_embedding(positive_histories) * weights.unsqueeze(-1)
        fm = fm_second_order(torch.cat([categorical, numerical, vectors], dim=1))
        linear = self.bias + (values * self.numeric_linear).sum(dim=1)
        for name, index in indexes.items():
            linear = linear + getattr(self, f"{name}_linear")(index).squeeze(-1)
        linear = linear + (self.item_linear(positive_histories).squeeze(-1) * weights).sum(dim=1)
        deep_input = torch.cat([categorical.flatten(1), vectors.sum(dim=1), numerical.flatten(1), interest], dim=1)
        deep = self.output(torch.relu(self.hidden(deep_input))).squeeze(-1)
        return linear + fm + deep

    def forward(self, users, items, positive_histories, numeric, sequence_items, ratings, time_buckets, positions):
        interest = self.extract_interest(sequence_items, ratings, time_buckets, positions)
        return self.score_with_interest(users, items, positive_histories, numeric, interest)


def make_sequence_model(config, categories, item_metadata):
    return SequenceRanker(config["user_count"], config["item_count"], categories, item_metadata,
                           encoder=config["encoder"], embedding_dim=config["embedding_dim"], hidden_dim=config["hidden_dim"],
                           sequence_length=config["sequence_length"], heads=config["heads"], layers=config["layers"],
                           ff_dim=config["ff_dim"], dropout=config["dropout"])


def restore_sequence(checkpoint):
    state = checkpoint["model_state"]
    model = make_sequence_model(checkpoint["config"], state["user_categorical"], state["item_metadata"])
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
