import pickle

import numpy as np

from eecs148b_hw1.bpe.tokenizer import BPETokenizer
from eecs148b_hw1.bpe.train import train_bpe

TINYSTORIES_TRAIN_PATH_RAW = "data/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VAL_PATH_RAW = "data/TinyStoriesV2-GPT4-valid.txt"
TINYSTORIES_TRAIN_PATH = "data/train.npy"
TINYSTORIES_VAL_PATH = "data/val.npy"
VOCAB_SAVE_PATH = "data/vocab.pkl"
MERGES_SAVE_PATH = "data/merges.pkl"
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]


def train_bpe_tinystories(
    train_path: str = TINYSTORIES_TRAIN_PATH_RAW,
    vocab_size: int = VOCAB_SIZE,
    special_tokens: list[str] = SPECIAL_TOKENS,
):
    print(f"Training BPE on {train_path}, with vocab size {vocab_size}.")
    vocab, merges = train_bpe(train_path, vocab_size, special_tokens)

    with open(VOCAB_SAVE_PATH, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(MERGES_SAVE_PATH, "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)

    longest_token = max(vocab.values(), key=lambda x: len(x))
    print(f"Longest token: {longest_token}")


def convert_txt_to_numpy(
    input_path: str, output_path: str, vocab_path: str, merges_path: str, special_tokens: list[str]
):
    print(f"Running BPE on {input_path}, outputting to {output_path}.")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)
    with open(input_path) as f:
        text = f.read()
        tokenized = np.array(tokenizer.encode(text), dtype=np.uint16)
    np.save(output_path, tokenized)


def convert_tinystories_to_numpy(
    train_path_raw: str = TINYSTORIES_TRAIN_PATH_RAW,
    train_path: str = TINYSTORIES_TRAIN_PATH,
    val_path_raw: str = TINYSTORIES_VAL_PATH_RAW,
    val_path: str = TINYSTORIES_VAL_PATH,
    vocab_path: str = VOCAB_SAVE_PATH,
    merges_path: str = MERGES_SAVE_PATH,
    special_tokens: list[str] = SPECIAL_TOKENS,
):
    convert_txt_to_numpy(train_path_raw, train_path, vocab_path, merges_path, special_tokens)
    convert_txt_to_numpy(val_path_raw, val_path, vocab_path, merges_path, special_tokens)


if __name__ == "__main__":
    train_bpe_tinystories()
    convert_tinystories_to_numpy()
