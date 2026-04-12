import math

import torch
import torch.nn as nn

from eecs148b_hw1.modules.common import init_normal_params


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shape = (out_features, in_features)
        self.device = device
        self.dtype = dtype

        # Initialize weights ~ N(0, 2/(d_in + d_out)) truncated to within 3 sigma.
        mean = 0
        std = math.sqrt(2 / (self.in_features + self.out_features))
        self.weight = init_normal_params(
            self.shape, mean=mean, std=std, a=-3 * std, b=3 * std, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
