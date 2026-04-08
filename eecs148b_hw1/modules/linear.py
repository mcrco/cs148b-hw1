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

        data = torch.zeros(self.shape, dtype=self.dtype)
        nn.init.trunc_normal_(data)
        self.W = nn.Parameter(data)

    def forward(self, x: torch.Tensor):
        return x @ self.W.T
