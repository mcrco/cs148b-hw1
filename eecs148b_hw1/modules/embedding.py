import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        data = torch.zeros((self.num_embeddings, self.embedding_dim), dtype=self.dtype, device=self.device)
        nn.init.trunc_normal_(data, mean=0, std=1, a=-3, b=3)
        self.lut = nn.Parameter(data)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.lut[token_ids.long()]
