import regex as re  # type: ignore
from collections import Counter, defaultdict
from multiprocessing import Pool
from pretokenization import find_chunk_boundaries


def generate_word_count(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter[str]:
    """
    Read a spefic chunk form file
    split chunk with special_tokens (to drop special_tokens)
    use regular pattern from GPT-2 to match the rest of the chunk to split it into elements(words)
    obtain the word_count.
    """
    # open file and load the desired chunk, then decode it to str
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # first use special_tokens to split the chunk
    split_pattern = "|".join(re.escape(x) for x in special_tokens)
    sentences = re.split(split_pattern, chunk)

    # PAT is the pre-tokenizing pattern from GPT-2
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # use PAT to construct the Counter
    word_count = Counter()
    for sentence in sentences:
        word_count.update(word.group() for word in re.finditer(PAT, sentence))

    return word_count


def merge_pair(tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
    """
    A state machine built to match and merge the pair to obtain a new tokenized sequence.
    """
    result = []
    pre_match = False

    for token in tokens:
        if pre_match:
            if token == pair[1]:
                result.append(new_token)
                pre_match = False
            else:
                if token == pair[0]:
                    result.append(pair[0])
                else:
                    result.extend([pair[0], token])
                    pre_match = False
        else:
            if token == pair[0]:
                pre_match = True
            else:
                result.append(token)
    if pre_match:
        result.append(pair[0])

    return result


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a bpe tokenizer to obtain a vocabulary and merge list from input text
    num_chunks, split_token_in_byte and num_cores are hidden constants to be confirmed befor using.
    """
    # find chunk boundaries
    with open(input_path, "rb") as f:
        num_chunks = 32
        split_token_in_byte = b"<|endoftext|>"
        boundaries = find_chunk_boundaries(f, num_chunks, split_token_in_byte)

    # build parallel task parameter list
    parameter_list = [(input_path, start, end, special_tokens) for start, end in zip(boundaries, boundaries[1:])]

    # parallelly run generate_word_count
    num_cores = 8
    with Pool(processes=num_cores) as pool:
        word_counts = pool.starmap(generate_word_count, parameter_list)

    # collect result form return list of parallel tasks
    # use update method to accelerate
    word_count = Counter()
    for count in word_counts:
        word_count.update(count)

    # build vocabulary
    vocab = {i: bytes([i]) for i in range(256)}

    # tokenize word
    tokenize = {word: list(word.encode()) for word in word_count}

    # initialize pair counter and pair to word embed(used defaultdict for embed)
    pair_count = Counter()
    embed = defaultdict(set)

    for word in word_count:
        tokens = tokenize[word]
        for pair in zip(tokens, tokens[1:]):
            pair_count[pair] += word_count[word]
            embed[pair].add(word)

    # merge and update vocab, merges, tokenize, pair_count, embed
    # until reached designed vocab_size or no more pairs
    merges = []
    while len(vocab) < vocab_size - len(special_tokens):
        # for evil conditions where everything get merged into one piece
        if not pair_count:
            break

        # lexicographic ordering is applied at the max comparing to break tie
        merge = max(pair_count, key=lambda x: (pair_count[x], vocab[x[0]], vocab[x[1]]))

        new_token = len(vocab)
        vocab[new_token] = vocab[merge[0]] + vocab[merge[1]]
        merges.append((vocab[merge[0]], vocab[merge[1]]))

        for word in embed[merge]:
            # reverse the contribution of word containing merge
            tokens = tokenize[word]
            for pair in zip(tokens, tokens[1:]):
                pair_count[pair] -= word_count[word]
                embed[pair].discard(word)
                if embed[pair] == {}:
                    embed.pop(pair, None)

            # merge pair and uppdate tokenize[word]
            # this is the core part and I have wrapped it into a function
            tokenize[word] = merge_pair(tokenize[word], merge, new_token)

            # update pair_count and embed with new tokenize[word]
            tokens = tokenize[word]
            for pair in zip(tokens, tokens[1:]):
                pair_count[pair] += word_count[word]
                embed[pair].add(word)

    # append special_tokens to vocab
    for s in special_tokens:
        vocab[len(vocab)] = s.encode()

    return vocab, merges
