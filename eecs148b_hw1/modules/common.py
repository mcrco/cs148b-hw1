import torch
import torch.nn as nn


def init_normal_params(shape: tuple[int, ...], mean: float, std: float, a: float, b: float, dtype=None, device=None):
    data = torch.zeros(shape, dtype=dtype, device=device)
    nn.init.trunc_normal_(data, mean=mean, std=std, a=a, b=b)
    return nn.Parameter(data)
