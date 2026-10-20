"""用户与电影 ID embedding + 一个隐藏层的前馈网络。"""
import torch
from torch import nn


class ShallowRecommender(nn.Module):
    def __init__(self, user_count, item_count, embedding_dim=64, hidden_dim=64):
        super().__init__()
        # 索引 0 保留给未知 ID；本实验仅评估训练期已出现的用户和物品。
        self.user_embedding = nn.Embedding(user_count + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_count + 1, embedding_dim, padding_idx=0)
        self.hidden = nn.Linear(2 * embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def forward(self, user_ids, item_ids):
        features = torch.cat([self.user_embedding(user_ids), self.item_embedding(item_ids)], dim=-1)
        return self.output(torch.relu(self.hidden(features))).squeeze(-1)
