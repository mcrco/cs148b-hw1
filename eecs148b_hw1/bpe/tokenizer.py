import pickle
from collections.abc import Iterable, Iterator

from tqdm import tqdm

from eecs148b_hw1.bpe.utils import apply_merges, pretokenizer, split_on_special_tokens, str_to_bytes_list


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_idx = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.merge_dict = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens
        self.pretoken_sequence_cache: dict[str, list[bytes]] = {}

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return BPETokenizer(vocab, merges, special_tokens)

    def encode(self, text: str, progress_bar: bool = False) -> list[int]:
        if self.special_tokens:
            chunks = split_on_special_tokens(text, self.special_tokens)
        else:
            chunks = [text]

        ids = []
        for chunk in tqdm(chunks, disable=not progress_bar):
            if self.special_tokens and chunk in self.special_tokens:
                ids.append(self.vocab_idx[chunk.encode("utf-8")])
            else:
                pretokens = pretokenizer.findall(chunk)
                for pretoken in pretokens:
                    if pretoken not in self.pretoken_sequence_cache:
                        seq = str_to_bytes_list(pretoken)
                        seq = apply_merges(seq, self.merge_dict)
                        self.pretoken_sequence_cache[pretoken] = seq
                    seq = self.pretoken_sequence_cache[pretoken]
                    for token in seq:
                        ids.append(self.vocab_idx[token])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[tid] for tid in ids]).decode("utf-8", errors="replace")
