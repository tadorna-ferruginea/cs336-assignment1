import regex as re
from collections import Counter
from multiprocessing import Pool
from pretokenization import find_chunk_boundaries


def generate_word_dict(chunk: str, special_tokens: list[str]) -> Counter[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    patterns = "|".join(re.escape(x) for x in special_tokens) + "|" + PAT
    word_count = Counter(re.findall(patterns, chunk))
    return word_count


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

    with Pool(processes=num_processes) as pool:
        word_counts = pool.starmap(generate_word_dict, [(chunk, special_tokens) for chunk in chunks])

    word_count = sum(word_counts, Counter())

    # pre-tokenize
    # bpe loop
    return vocab, merges
