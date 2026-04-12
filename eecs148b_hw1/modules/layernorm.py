import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Initialize affine transformation weights to 1 and bias to 0.
        weights = torch.ones(self.d_model, dtype=self.dtype, device=self.device)
        bias = torch.zeros(self.d_model, dtype=self.dtype, device=self.device)
        self.weight = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save curent dtype and cast to float32.
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # layernorm(a_i) = (a_i - mu(a)) / sqrt(var + eps) * g_i + b_i.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        out = norm * self.weight + self.bias

        # Cast back to original dtype.
        out = out.to(in_dtype)
        return out
