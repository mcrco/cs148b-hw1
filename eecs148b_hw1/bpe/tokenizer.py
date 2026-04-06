import re
from collections import defaultdict
from heapq import heappop, heappush


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_idx = {v: k for k, v in self.vocab.items()}
        self.merges = set(merges)
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        pass

    def encode(self, text: str) -> list[int]:
        """
        Uses internal merges to encode a string into a list of integers.
        """

        tokenized = list(text.encode("utf-8"))
        idx = 1
        while idx < len(tokenized):
            # maybe merge with previous token
            if (tokenized[idx - 1], tokenized[idx]) in self.merges:
                tokenized = (
                    tokenized[: idx - 1] + self.merges[(tokenized[idx - 1], tokenized[idx])] + tokenized[idx + 1 :]
                )
                idx -= 1
            # maybe merge with next token
            elif (tokenized[idx], tokenized[idx + 1]) in self.merges:
                tokenized = tokenized[:idx] + self.merges[(tokenized[idx], tokenized[idx + 1])] + tokenized[idx + 2 :]
            # move on for now, but step 1 allows us to come back if needed
            else:
                idx += 1
        return [self.vocab_idx[token] for token in tokenized]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.vocab[id] for id in ids])
