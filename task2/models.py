"""Custom pooling/expected-rating head and hierarchical collaborative regression."""
import torch
from torch import nn
import torch.nn.functional as F

from task2.common import NUMERIC_NAMES


class TextRatingModel(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 5))
        self.register_buffer("ratings", torch.arange(1., 6.))

    def forward(self, input_ids, attention_mask, token_type_ids, body_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        mask = (body_mask * attention_mask).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.)
        pooled = torch.where((mask.sum(1) > 0), pooled, hidden[:, 0])
        logits = self.head(torch.cat([hidden[:, 0], pooled], dim=1))
        return (logits.float().softmax(-1) * self.ratings).sum(-1), logits

    @staticmethod
    def loss(prediction, logits, rating, ce_weight):
        return F.mse_loss(prediction, rating) + ce_weight * F.cross_entropy(logits.float(), rating.long() - 1)


class RelationRatingModel(nn.Module):
    def __init__(self, cardinalities, numeric_size, global_mean, dimension=32, drop_user=False, drop_parent=False):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, dimension, padding_idx=0) for n in cardinalities])
        self.biases = nn.ModuleList([nn.Embedding(n, 1, padding_idx=0) for n in cardinalities])
        self.metadata = nn.Sequential(nn.Linear(numeric_size, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("global_mean", torch.tensor(global_mean, dtype=torch.float32))
        self.drop_user, self.drop_parent = drop_user, drop_parent
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
            with torch.no_grad():
                embedding.weight[0].zero_()
        for bias in self.biases:
            nn.init.zeros_(bias.weight)
        nn.init.zeros_(self.metadata[-1].weight)
        nn.init.zeros_(self.metadata[-1].bias)

    def forward(self, entities, numeric):
        entities = entities.clone()
        numeric = numeric.clone()
        for index, drop, prefix in [(0, self.drop_user, "user_"), (2, self.drop_parent, "parent_")]:
            if drop:
                entities[:, index] = 0
                columns = [i for i, name in enumerate(NUMERIC_NAMES) if name.startswith(prefix)]
                numeric[:, columns] = 0.
        vectors = [layer(entities[:, i]) for i, layer in enumerate(self.embeddings)]
        bias = sum(layer(entities[:, i]).squeeze(-1) for i, layer in enumerate(self.biases))
        interaction = (vectors[0] * (vectors[1] + vectors[2])).sum(-1)
        return self.global_mean + bias + interaction + self.metadata(numeric).squeeze(-1)
