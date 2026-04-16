"""Train the Transformer LM on TinyStories token ids stored as NumPy arrays."""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from eecs148b_hw1.training.train import train as run_train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train on TinyStories (tokenized .npy files).")
    p.add_argument("--train-path", type=Path, default=Path("data/train.npy"), help="Path to training token array.")
    p.add_argument("--val-path", type=Path, default=Path("data/val.npy"), help="Path to validation token array.")
    p.add_argument("--vocab-size", type=int, default=10_000, help="Vocabulary size (must match BPE training).")

    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducible initialization and batch sampling.")
    p.add_argument(
        "--train-batches-per-epoch",
        type=int,
        default=None,
        help="Random training batches to sample per epoch. Default: approximate one non-overlapping pass.",
    )
    p.add_argument(
        "--val-batches-per-epoch",
        type=int,
        default=None,
        help="Random validation batches to sample per epoch. Default: approximate one non-overlapping pass.",
    )
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.01)

    p.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Model and optimizer compute dtype.",
    )
    p.add_argument("--device", default="cuda", help='Device, e.g. "cuda", "cuda:0", or "cpu".')

    p.add_argument(
        "--wandb-run-name",
        default=None,
        help="Weights are saved as weights/<name>.pth. Default: tinystories-YYYYMMDD-HHMMSS.",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (sets WANDB_MODE=disabled).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Reduce length of dataset to just the context length for debugging model.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    set_seed(args.seed)

    train_path = args.train_path.resolve()
    val_path = args.val_path.resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    train_data = np.load(train_path, mmap_mode="r")
    val_data = np.load(val_path, mmap_mode="r")
    if args.debug:
        train_data = np.load(train_path, mmap_mode="r")[: args.context_length + 2]
        val_data = np.load(train_path, mmap_mode="r")[: args.context_length + 2]

    print(f"Training data token count: {len(train_data)}.")
    print(f"Validation data token count: {len(val_data)}.")
    print(f"Using seed: {args.seed}.")

    wandb_run_name = args.wandb_run_name
    if wandb_run_name is None:
        wandb_run_name = f"tinystories-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    run_train(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        train_dataset=train_data,
        val_dataset=val_data,
        wandb_run_name=wandb_run_name,
        train_batches_per_epoch=args.train_batches_per_epoch,
        val_batches_per_epoch=args.val_batches_per_epoch,
        dtype=dtype,
        device=args.device,
    )


if __name__ == "__main__":
    main()
