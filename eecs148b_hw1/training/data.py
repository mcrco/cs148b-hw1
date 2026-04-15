import numpy as np
import numpy.typing as npt
import torch


def get_batch(x: npt.NDArray, indices: npt.NDArray, context_length: int, device: str):
    # Indices start off as something like [[0], [1], [2], [3]].
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
    # Offsets are something like [[0, 1, 2, 3]].
    offsets = np.arange(context_length, dtype=np.int64).reshape(1, -1)

    # Indices + offsets broadcasts to [[0, 0 + 1, 0 + 2, 0 + 3], [1 + 0, 1 + 1, ...]].
    input_tokens = np.asarray(x[indices + offsets])
    output_tokens = np.asarray(x[indices + offsets + 1])

    inputs = torch.from_numpy(input_tokens).to(device=device, dtype=torch.long)
    outputs = torch.from_numpy(output_tokens).to(device=device, dtype=torch.long)
    return inputs, outputs


def get_random_batch(x: npt.NDArray, batch_size: int, context_length: int, device: str):
    indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    return get_batch(x, indices, context_length, device)
