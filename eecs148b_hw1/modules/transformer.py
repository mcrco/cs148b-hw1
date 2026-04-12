import torch
import torch.nn as nn

from eecs148b_hw1.modules.attention import MultiHeadSelfAttention
from eecs148b_hw1.modules.ffn import FeedForwardNetwork
from eecs148b_hw1.modules.layernorm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dtype=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dtype = dtype
        self.device = device

        self.ln1 = LayerNorm(d_model, dtype=dtype, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dtype=dtype, device=device)
        self.ln2 = LayerNorm(d_model, dtype=dtype, device=device)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.ln1(x))
        z = y + self.ffn(self.ln2(y))
        return z
