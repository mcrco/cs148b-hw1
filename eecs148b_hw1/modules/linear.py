import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shape = (out_features, in_features)
        self.device = device
        self.dtype = dtype

        # Initialize weights ~ N(0, 2/(d_in + d_out)) truncated to within 3 sigma.
        data = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        mean = 0
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(data, mean=mean, std=std, a=-3 * std, b=3 * std)
        self.W = nn.Parameter(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
