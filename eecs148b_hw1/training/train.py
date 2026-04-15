import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

import wandb
from eecs148b_hw1.modules.lm import TransformerLM
from eecs148b_hw1.modules.loss import cross_entropy_loss, perplexity

from .data import get_batch


def get_lr(lr: float, step: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr * float(step) / float(warmup_steps)
    return lr


def get_batch_indices(
    dataset: npt.NDArray,
    context_length: int,
    batch_size: int,
) -> list[npt.NDArray]:
    # For dataset length L and context length C, inputs are indices [i, i + C)
    # and ouputs are indices [i + 1, i + C + 1), so the maximum starting index
    # for any sequence in a batch is L - C - 1.
    max_index = len(dataset) - context_length - 1
    indices = np.random.permutation(max_index + 1)
    return [indices[i : min(max_index + 1, i + batch_size)] for i in range(0, max_index, batch_size)]


def train(
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    epochs: int,
    lr: float,
    batch_size: int,
    warmup_steps: int,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    train_dataset: npt.NDArray,
    val_dataset: npt.NDArray,
    wandb_run_name: str,
    dtype: torch.dtype = torch.float,
    device: str = "cuda",
):
    wandb.init(project="cs148b", name=wandb_run_name)

    model = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length, num_layers, dtype=dtype, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=epsilon
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: get_lr(lr, step, warmup_steps))

    # Progress bar logs in terms of training batches for more detail.
    total_batches = epochs * len(get_batch_indices(train_dataset, context_length, batch_size))
    with tqdm(total=total_batches) as pbar:
        for epoch in range(epochs):
            # Training
            pbar.set_description(f"Epoch: {epoch}/{epochs} (training)")
            model.train()
            train_batch_indices_list = get_batch_indices(train_dataset, context_length, batch_size)
            for indices in train_batch_indices_list:
                x, y = get_batch(train_dataset, indices, context_length, device)
                logits = model(x)
                loss = cross_entropy_loss(logits, y)
                wandb.log({"train/loss": loss}, step=pbar.n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.update(1)

            # Validation
            pbar.set_description(f"Epoch: {epoch}/{epochs} (validation)")
            model.eval()
            val_batch_indices_list = get_batch_indices(val_dataset, context_length, batch_size)
            val_loss = 0
            val_pplx = 0
            for indices in val_batch_indices_list:
                x, y = get_batch(val_dataset, indices, context_length, device)
                logits = model(x)
                val_loss += torch.sum(cross_entropy_loss(logits, y)).item()
                val_pplx += torch.sum(perplexity(logits, y)).item()
            wandb.log({"val/loss": val_loss, "val/perplexity": val_pplx}, step=pbar.n)

    # Save model.
    weights_path = f"weights/{wandb_run_name}.pth"
    weights_path = Path(weights_path)
    Path(weights_path.parent).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}")
