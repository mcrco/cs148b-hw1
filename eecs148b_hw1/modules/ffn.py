import torch
import torch.nn as nn

from .linear import Linear


def relu(x: torch.Tensor) -> torch.Tensor:
    # Apparently torch.where is differentiable so this works.
    return torch.where(x > 0.0, x, 0.0).to(x.dtype)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dtype=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = Linear(self.d_model, self.d_ff, dtype=dtype, device=device)
        self.fc2 = Linear(self.d_ff, self.d_model, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(relu(self.fc1(x)))
