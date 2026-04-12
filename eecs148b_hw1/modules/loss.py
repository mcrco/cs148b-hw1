import torch
import torch.nn as nn


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # From log sum exp trick,
    # softmax(logits[i]) = exp(logits[i] - max(logits)) / sum(exp(logits[j] - max(logits))).
    # Then, log of that is
    # log softmax(logits[i]) = log exp(logits[i] - max(logits)) - log(sum(exp(logits[j] - max(logits)))).
    # In cross entropy, we are focused on
    # -log p(targets[i]) = -log softmax(logits[targets[i]])
    # = -(log exp(logits[i] - max(logits)) - log(sum(exp(logits[j] - max(logits)))))
    # = -(logits[i] - max(logits) - log(sum(exp(logits[j] - max(logits)))))
    max_logits = logits.amax(dim=-1, keepdim=True)
    log_probs = logits - max_logits - torch.log(torch.exp(logits - max_logits).sum(dim=-1, keepdim=True))
    losses = -log_probs[torch.arange(logits.shape[0]), targets]
    return losses.mean()
