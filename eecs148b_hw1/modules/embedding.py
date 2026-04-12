import torch
import torch.nn as nn

from eecs148b_hw1.modules.common import init_normal_params


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # Initialize embeddings ~ N(0, 1) truncated to [-3, 3].
        self.weight = init_normal_params(
            (self.num_embeddings, self.embedding_dim), 0, 1, -3, 3, dtype=dtype, device=device
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids.long()]
