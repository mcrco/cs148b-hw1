import torch
import torch.nn as nn

from eecs148b_hw1.modules.linear import Linear
from eecs148b_hw1.modules.transformer import TransformerBlock

from .embedding import Embedding
from .layernorm import LayerNorm
from .positional_encoding import SinusoidalPositionalEncoding


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        self.token_embeddings = Embedding(self.vocab_size, self.d_model, dtype=self.dtype, device=self.device)
        self.positional_embeddings = SinusoidalPositionalEncoding(
            self.d_model, self.context_length, dtype=self.dtype, device=self.device
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(self.d_model, self.num_heads, self.d_ff, dtype=self.dtype, device=self.device)
                for _ in range(self.num_layers)
            ]
        )
        self.ln_final = LayerNorm(self.d_model, dtype=self.dtype, device=self.device)
        self.lm_head = Linear(self.d_model, self.vocab_size, dtype=self.dtype, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embeddings(x)
        positional_embeddings = self.positional_embeddings(torch.arange(x.shape[-1]))
        embeddings = token_embeddings + positional_embeddings
        out = embeddings
        for block in self.layers:
            out = block(out)
        out = self.ln_final(out)
        logits = self.lm_head(out)
        return logits
