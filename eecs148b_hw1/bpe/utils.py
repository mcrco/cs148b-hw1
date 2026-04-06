from collections import defaultdict

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pretokenizer = re.compile(PAT)


def str_to_bytes_list(text: str) -> list[bytes]:
    return [bytes([byte]) for byte in text.encode("utf-8")]


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    special_tokens = [re.escape(token) for token in special_tokens]
    return re.split("|".join(special_tokens), text)


def pretokenize(text: str, special_tokens: list[str] | None = None) -> dict[str, int]:
    """
    Uses regex to pretokenize corpus before training as in GPT-2.
    """
    if special_tokens:
        chunks = split_on_special_tokens(text, special_tokens)
    else:
        chunks = [text]
    pretoken_counts = defaultdict(int)
    for chunk in chunks:
        if special_tokens and chunk in special_tokens:
            continue
        for pretoken in pretokenizer.findall(chunk):
            pretoken_counts[pretoken] += 1
    return pretoken_counts


def apply_merges(seq: list[bytes], merges: list[tuple[bytes, bytes]]) -> list[bytes]:
    for first, second in merges:
        i = 0
        while i < len(seq) - 1:
            if seq[i] == first and seq[i + 1] == second:
                seq = seq[:i] + [seq[i] + seq[i + 1]] + seq[i + 2:]
            i += 1
    return seq