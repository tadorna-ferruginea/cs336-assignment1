import pickle
import os
import regex as re  # type: ignore
from collections import defaultdict
from collections.abc import Iterable, Iterator


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        construct inverse_vocab
        construct the class, add additional special_token to vocab
        """
        self.vocab = vocab

        # construct inverse_vocab
        self.inverse_vocab = {}
        for token, idx in self.vocab.items():
            self.inverse_vocab[idx] = token

        self.merges = merges
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        # cache for _tokenize
        self.tokenize = {}

        # iterate over special_tokens and update vocab and its reverse
        for word in self.special_tokens:
            if word.encode() not in self.inverse_vocab:
                self.inverse_vocab[word.encode()] = len(self.vocab)
                self.vocab[len(self.vocab)] = word.encode()

        # construct the split pattern with capture
        self.split_pattern = (
            "(" + "|".join(re.escape(x) for x in self.special_tokens) + ")"
        )

        # PAT is the pre-tokenizing pattern from GPT-2
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # this is used in encode_iterable
        # generate a prefix set of the special tokens
        # also find the length of the longest special token
        self.prefix_set = set()
        if not self.special_tokens:
            self.max_len = 0
        else:
            self.max_len = len(self.special_tokens[0])-1
        for token in self.special_tokens:
            for length in range(len(token)-1):
                self.prefix_set.add(token[:length])

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        """
        alternative class construction
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _tokenize(self, word_str: str) -> tuple[int, ...]:
        """
        the working function to map word into token sequence
        my favorite state machine
        """
        # use cache
        if word_str in self.tokenize:
            return self.tokenize[word_str]

        word = [bytes([token]) for token in word_str.encode()]
        for merge in self.merges:
            pre_match = False
            new_word = []
            for token in word:
                if pre_match:
                    if token == merge[1]:
                        new_word.append(merge[0] + merge[1])
                        pre_match = False
                    else:
                        if token == merge[0]:
                            new_word.append(token)
                        else:
                            new_word.extend([merge[0], token])
                            pre_match = False
                else:
                    if token == merge[0]:
                        pre_match = True
                    else:
                        new_word.append(token)
            if pre_match:
                new_word.append(merge[0])
            word = new_word

        self.tokenize[word_str] = tuple(self.inverse_vocab[token] for token in word)
        return self.tokenize[word_str]

    def encode(self, text: str) -> list[int]:
        """
        we split the text with special_tokens
        capture the elements with PAT
        then apply merges to each captured part
        connect everything
        """
        # split!
        if not self.special_tokens:
            sentences = [text]
        else:
            sentences = re.split(self.split_pattern, text)

        # creat a list to collect result
        result = []
        # for each sentence, extract all words
        # note that one special token is also splited into a sentence
        for sentence in sentences:
            if sentence in self.special_tokens:
                result.append(self.inverse_vocab[sentence.encode()])
            else:
                for word in (x.group() for x in re.finditer(self.PAT, sentence)):
                    result.extend(self._tokenize(word))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        I don't need to care about how large the iterable element is
        Just let the iterator flow
        """

        # residual is what's left in the previous buffer
        residual = ""
        for chunk in iterable:
            buffer = residual + chunk

            # rule out the special suffix
            # this part is tricky, we only find the split point here,
            # because we need to rule out any special token overlap
            suffix_start = max(0, len(buffer) - self.max_len)
            for split_point in range(suffix_start, len(buffer)):
                if buffer[split_point:] in self.prefix_set:
                    break

            # split y determin boundary
            # we can't use encode directly because we need to deal with the last word
            cumulant_length = 0
            if not self.special_tokens:
                sentences = [text]
            else:
                sentences = re.split(self.split_pattern, text)

            for sentence in sentences[:-1]:
                if sentence in self.special_tokens:
                    yield self.inverse_vocab[sentence.encode()]
                else:
                    for word in (x.group() for x in re.finditer(self.PAT, sentence)):
                        yield from self._tokenize(word)
            # deal with tail: note that the last sentence will never be any special token,
            # due to the behavior of split
            sentence = sentences[-1]
            word_before = None
            for word in (x.group() for x in re.finditer(self.PAT, sentence)):
                if word_before:
                    yield from self._tokenize(word_before)
                word_before = word
            residual = 
                    
我直接split，把最后的组合体和prefix比长度就好了
        # clean up the tail
        yield from self.encode(residual)
