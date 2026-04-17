import argparse
from pathlib import Path

import torch
from torch import Tensor

from eecs148b_hw1.bpe.tokenizer import BPETokenizer
from eecs148b_hw1.modules.activation import softmax
from eecs148b_hw1.modules.lm import TransformerLM

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def mask_for_top_p(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
    indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
    logits = logits.clone()
    logits[indices_to_remove] = float("-inf")
    return logits


def sample_next_token(logits: Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    if top_p < 1.0:
        logits = mask_for_top_p(logits, top_p)
    probs = softmax(logits, dim=-1)
    if bool((probs <= 0).all()):
        return int(torch.argmax(logits, dim=-1).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_completion(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    context_length: int,
    temperature: float,
    top_p: float,
    endoftext_id: int | None = None,
) -> str:
    if endoftext_id is None:
        endoftext_id = tokenizer.vocab_idx[b"<|endoftext|>"]

    device = next(model.parameters()).device
    model.eval()
    ids: list[int] = tokenizer.encode(prompt, progress_bar=False)
    generated = 0

    with torch.inference_mode():
        while generated < context_length:
            context = ids
            if len(context) > model.context_length:
                context = context[-model.context_length :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1, :].float()
            next_id = sample_next_token(next_logits, temperature, top_p)
            ids.append(next_id)
            generated += 1
            if next_id == endoftext_id:
                break

    return tokenizer.decode(ids)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with a trained TransformerLM.")
    p.add_argument("--weights", type=Path, required=True, help="Path to a .pth state dict.")
    p.add_argument("--prompt", type=str, default="", help="Prompt text to continue.")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Cap on newly generated tokens.")
    p.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (<=0 for greedy).")
    p.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling threshold in (0, 1]; 1 disables.")
    p.add_argument("--vocab", type=Path, default=Path("data/vocab.pkl"))
    p.add_argument("--merges", type=Path, default=Path("data/merges.pkl"))
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument(
        "--no-layernorm",
        action="store_true",
        help="Load a checkpoint trained with all LayerNorm modules replaced by nn.Identity.",
    )
    p.add_argument(
        "--no-positional-embeddings",
        action="store_true",
        help="Load a checkpoint trained with positional embeddings zeroed out.",
    )
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--device", default="cuda", help='Device, e.g. "cuda", "cuda:0", or "cpu".')
    p.add_argument("--seed", type=int, default=None, help="If set, seed PyTorch for sampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    tokenizer = BPETokenizer.from_files(str(args.vocab), str(args.merges), DEFAULT_SPECIAL_TOKENS)

    model = TransformerLM(
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.vocab_size,
        args.context_length,
        args.num_layers,
        use_layernorm=not args.no_layernorm,
        use_positional_embeddings=not args.no_positional_embeddings,
        dtype=dtype,
        device=args.device,
    )
    state = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(state)

    text = generate_completion(
        model,
        tokenizer,
        args.prompt,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text, end="" if text.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
