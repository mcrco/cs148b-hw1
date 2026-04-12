import math

import torch
import torch.nn as nn

from eecs148b_hw1.modules.activation import softmax
from eecs148b_hw1.modules.linear import Linear


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    # Q: B x m x d_k
    # K: B x n x d_k
    # V: B x n x d_v
    d_k = K.shape[-1]
    logits = Q @ K.transpose(-1, -2) / math.sqrt(d_k)  # should be shape B x m x n
    if mask is not None:
        mask_logits = torch.where(mask, 0, -torch.inf)  # B x m x n
        logits += mask_logits
    probs = softmax(logits, dim=-1)  # B x m x n
    attention = probs @ V  # B x m x d_v
    return attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype

        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_model // self.num_heads

        self.W_q = Linear(self.d_model, self.d_model, device=self.device, dtype=self.dtype)
        self.W_k = Linear(self.d_model, self.d_model, device=self.device, dtype=self.dtype)
        self.W_v = Linear(self.d_model, self.d_model, device=self.device, dtype=self.dtype)
        self.W_o = Linear(self.d_model, self.d_model, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        # For each of the linear layer outputs, we have B x S x d_model.
        # We want to reshape the concatenated projections into projections per head, which is B x S x H x d_k/d_v.
        # Then, we want to do attention, but the expected dimensions are B x ... x S x d_k/d_v, so we transpose again.
        Q = self.W_q(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Causal mask should have mask[i, j] = 1 for i <= j, which is lower triangle of ones.
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool))
        # Unsqueeze first two dims so it can broadcast to B x H x S x S.
        mask = mask.reshape(1, 1, seq_len, seq_len)
        attention = scaled_dot_product_attention(Q, K, V, mask)
        # Right now, the output is B x H x S x D.
        # We want to go back to the origianl B x S x d_model.
        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.W_o(attention)
