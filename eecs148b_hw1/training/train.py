import math
from pathlib import Path

import numpy.typing as npt
import torch
from tqdm import tqdm

import wandb
from eecs148b_hw1.modules.lm import TransformerLM
from eecs148b_hw1.modules.loss import cross_entropy_loss, perplexity

from .data import get_random_batch


def get_lr_multiplier(step: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(warmup_steps)
    return 1.0


def get_batches_per_epoch(
    dataset: npt.NDArray,
    context_length: int,
    batch_size: int,
) -> int:
    num_valid_starts = len(dataset) - context_length
    if num_valid_starts <= 0:
        raise ValueError("Dataset must be longer than the context length.")
    # Sample enough random batches to cover roughly one non-overlapping pass worth of tokens.
    return max(1, math.ceil(num_valid_starts / (batch_size * context_length)))


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
    train_batches_per_epoch: int | None = None,
    val_batches_per_epoch: int | None = None,
    use_layernorm: bool = True,
    use_positional_embeddings: bool = True,
    dtype: torch.dtype = torch.float,
    device: str = "cuda",
    load_weights: str | Path | None = None,
):
    wandb.init(project="cs148b", name=wandb_run_name)

    model = TransformerLM(
        d_model,
        num_heads,
        d_ff,
        vocab_size,
        context_length,
        num_layers,
        use_layernorm=use_layernorm,
        use_positional_embeddings=use_positional_embeddings,
        dtype=dtype,
        device=device,
    )
    if load_weights is not None:
        load_weights = Path(load_weights)
        if not load_weights.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {load_weights}")
        state = torch.load(load_weights, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {load_weights}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=epsilon
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: get_lr_multiplier(step, warmup_steps))

    if train_batches_per_epoch is None:
        train_batches_per_epoch = get_batches_per_epoch(train_dataset, context_length, batch_size)
    if val_batches_per_epoch is None:
        val_batches_per_epoch = get_batches_per_epoch(val_dataset, context_length, batch_size)

    # Progress bar logs in terms of training batches for more detail.
    total_batches = epochs * train_batches_per_epoch
    with tqdm(total=total_batches) as pbar:
        for epoch in range(epochs):
            # Training
            pbar.set_description(f"Epoch: {epoch}/{epochs} (training)")
            model.train()
            for _ in range(train_batches_per_epoch):
                x, y = get_random_batch(train_dataset, batch_size, context_length, device)
                logits = model(x)
                loss = cross_entropy_loss(logits, y)
                wandb.log({"train/loss": loss.item()}, step=pbar.n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.update(1)

            # Validation
            pbar.set_description(f"Epoch: {epoch}/{epochs} (validation)")
            model.eval()
            val_loss = 0
            val_pplx = 0
            with torch.no_grad():
                for _ in range(val_batches_per_epoch):
                    x, y = get_random_batch(val_dataset, batch_size, context_length, device)
                    logits = model(x)
                    val_loss += cross_entropy_loss(logits, y).item()
                    val_pplx += perplexity(logits, y).item()
            wandb.log(
                {
                    "val/loss": val_loss / val_batches_per_epoch,
                    "val/perplexity": val_pplx / val_batches_per_epoch,
                },
                step=pbar.n,
            )

            # Save model.
            weights_path = f"weights/{wandb_run_name}/epoch-{epoch}.pth"
            weights_path = Path(weights_path)
            Path(weights_path.parent).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights_path)
            print(f"Saved weights to {weights_path}")
