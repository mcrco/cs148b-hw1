import math

import torch
import torch.nn as nn

from eecs148b_hw1.modules.activation import softmax


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
