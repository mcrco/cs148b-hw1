from collections import defaultdict

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pretokenizer = re.compile(PAT)


def str_to_bytes_list(text: str) -> list[bytes]:
    return [bytes([byte]) for byte in text.encode("utf-8")]


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    # Escape special characters in special tokens + sort descending by length
    # in case one special token is a prefix of another.
    escaped = sorted([re.escape(token) for token in special_tokens], key=len, reverse=True)
    pattern = f"({'|'.join(escaped)})"
    # Return non-empty chunks only.
    return [chunk for chunk in re.split(pattern, text) if chunk != ""]


def get_pretoken_counts(text: str, special_tokens: list[str] | None = None) -> dict[str, int]:
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


def apply_merges(seq: list[bytes], merge_dict: dict[tuple[bytes, bytes], int]) -> list[bytes]:
    while True:
        # Find highest priority merge.
        merge_idx = -1
        for i in range(len(seq) - 1):
            if (seq[i], seq[i + 1]) in merge_dict and (
                merge_idx == -1 or merge_dict[(seq[i], seq[i + 1])] < merge_dict[(seq[merge_idx], seq[merge_idx + 1])]
            ):
                merge_idx = i
        if merge_idx == -1:
            return seq
        # Concat the merged subtokens together.
        seq[merge_idx] = seq[merge_idx] + seq[merge_idx + 1]
        del seq[merge_idx + 1]
