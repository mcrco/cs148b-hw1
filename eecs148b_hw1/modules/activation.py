import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    # Apparently torch.where is differentiable so this works.
    return torch.where(x > 0.0, x, 0.0).to(x.dtype)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # log sum exp trick:
    # exp(v_i) / sum(exp(v_i)) = exp(v_i - max(v)) / sum(exp(v_i - max(v)))
    # because we just multiplied numerator and denominator by exp(-max(v)).
    x = x - x.amax(dim=dim, keepdim=True)
    numers = torch.exp(x)
    denoms = numers.sum(dim=dim, keepdim=True)
    return numers / denoms
