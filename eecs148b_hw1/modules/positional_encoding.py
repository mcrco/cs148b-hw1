import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, use_positional_embeddings: bool = True, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_positional_embeddings = use_positional_embeddings
        self.device = device
        self.dtype = dtype

        embeddings = torch.zeros((self.max_seq_len, self.d_model), device=self.device, dtype=self.dtype)
        if self.use_positional_embeddings:
            # Denominator is 10000^(2i / d_model) = exp(log(1000) * 2i / model).
            indices = torch.arange(0.0, d_model, 2.0, device=self.device, dtype=self.dtype)
            denoms = torch.exp(math.log(10000.0) * indices / self.d_model)

            # Make numerators shape (max_seq_len, 1) so that we can index and
            # divide by denoms to auto broadcast to shape (max_seq_len, d_model // 2).
            numers = torch.arange(max_seq_len, device=self.device, dtype=self.dtype).unsqueeze(1)

            # Even terms are simple.
            embeddings[:, 0::2] = torch.sin(numers / denoms)
            # Might not have last odd term depending on whether or not d_model is even.
            if self.d_model % 2 != 0:
                embeddings[:, 1::2] = torch.cos(numers / denoms[:-1])
            else:
                embeddings[:, 1::2] = torch.cos(numers / denoms)
        self.register_buffer("embeddings", embeddings, persistent=False)

    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        return self.get_buffer("embeddings")[token_positions]
