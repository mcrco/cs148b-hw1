import os
from collections import defaultdict
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

from .utils import apply_merges, pretokenize, str_to_bytes_list


@dataclass
class MergeCandidate:
    count: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other: "MergeCandidate") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return True if self.pair > other.pair else False


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str] | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path) as f:
        text = f.read()
    pretoken_counts = pretokenize(text, special_tokens)

    # Set up initial token vocabulary of all possible bytes and map tokens to pretokens.
    possible_bytes = [bytes([i]) for i in range(256)]
    bytes_to_idx: dict[bytes, int] = {byte: i for i, byte in enumerate(possible_bytes)}
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_pretokens: dict[tuple[bytes, bytes], set[str]] = defaultdict(set)
    for pretoken, count in pretoken_counts.items():
        seq = str_to_bytes_list(pretoken)
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += count
            pair_pretokens[pair].add(pretoken)

    # Create max heap of pairs of tokens we can merge, ordered by count + lexicography.
    pair_count_heap: list[MergeCandidate] = [MergeCandidate(count, pair) for pair, count in pair_counts.items()]
    heapify(pair_count_heap)

    # Repeatedly pop pair of tokens with highest frequency and merge until
    # vocab size is reached or no more pairs can be merged.
    merges = []
    effective_vocab_size = vocab_size - len(special_tokens)
    while pair_count_heap and len(bytes_to_idx) < effective_vocab_size:
        best_cand = heappop(pair_count_heap)
        max_pair, max_count = best_cand.pair, best_cand.count

        # Skip stale pair counts/removed pairs.
        if pair_counts[max_pair] != max_count or pair_counts[max_pair] == 0:
            continue

        # Use cached pretokens per pair to only update pair counts based on
        # relevant pretokens.
        new_token = max_pair[0] + max_pair[1]
        updated_pairs: set[tuple[bytes, bytes]] = set()
        for pretoken in pair_pretokens[max_pair]:
            # Process pretoken into existing tokens (including new one).
            seq = apply_merges(str_to_bytes_list(pretoken), merges)

            # If new token is t1 + t2, for any previous sequence prev -> t1 ->
            # t2 -> next, we should decrement the pair counts for (prev, t1) and
            # (t2, next) and increment (prev, t1 + t2) and (t1 + t2, next). We
            # keep track of all of the updated pairs and update the pair count
            # heap at the end instead of adding a new heap entry right away to
            # reduce memory usage.
            for i in range(len(seq) - 2):
                if (seq[i], seq[i + 1]) == max_pair:
                    old_pair = (seq[i + 1], seq[i + 2])
                    new_pair = (new_token, seq[i + 2])
                    updated_pairs.add(old_pair)
                    updated_pairs.add(new_pair)
                    pair_counts[old_pair] -= pretoken_counts[pretoken]
                    pair_counts[new_pair] += pretoken_counts[pretoken]
                    pair_pretokens[old_pair].add(pretoken)
                    pair_pretokens[new_pair].add(pretoken)
                if (seq[i + 1], seq[i + 2]) == max_pair:
                    old_pair = (seq[i], seq[i + 1])
                    new_pair = (seq[i], new_token)
                    updated_pairs.add(old_pair)
                    updated_pairs.add(new_pair)
                    pair_counts[old_pair] -= pretoken_counts[pretoken]
                    pair_counts[new_pair] += pretoken_counts[pretoken]
                    pair_pretokens[old_pair].add(pretoken)
                    pair_pretokens[new_pair].add(pretoken)

        # Update all the pair counts in the heap. We don't need to remove the old pair
        # count since our loop skips stale pair counts.
        for pair in updated_pairs:
            new_merge_cand = MergeCandidate(pair_counts[pair], pair)
            heappush(pair_count_heap, new_merge_cand)

        # Register pair as merge/new token and remove its count.
        merges.append(max_pair)
        bytes_to_idx[new_token] = len(bytes_to_idx)
        pair_counts[(max_pair[0], max_pair[1])] = 0

    # Add special tokens to vocabulary.
    for special_token in special_tokens:
        if special_token not in bytes_to_idx:
            bytes_to_idx[special_token.encode('utf-8')] = len(bytes_to_idx)

    vocab = {bytes_to_idx[byte]: byte for byte in bytes_to_idx}
    return vocab, list(merges)
