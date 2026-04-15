import numpy as np
import numpy.typing as npt
import torch


def get_sequence(x: npt.NDArray, start: int, context_length: int, device: str) -> torch.Tensor:
    return torch.from_numpy(x[start : start + context_length].copy()).to(device).to(torch.long)


def get_batch(x: npt.NDArray, indices: npt.NDArray, context_length: int, device: str):
    inputs = torch.stack([get_sequence(x, i, context_length, device) for i in indices])
    outputs = torch.stack([get_sequence(x, i + 1, context_length, device) for i in indices])
    return inputs, outputs


def get_random_batch(x: npt.NDArray, batch_size: int, context_length: int, device: str):
    indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    return get_batch(x, indices, context_length, device)
