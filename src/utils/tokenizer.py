from typing import List
import torch
import random

class Tokenizer():
    __default_vocab = " '()-abcdefghijklmnopqrstuvwxyzáâäéêëíîóôöúû"

    def __init__(self, vocab: List[str] = [], max_length: int = 10):
        vocab = list(set(vocab + [v for v in self.__default_vocab]))
        vocab = sorted(vocab)
        self.start_token = "S"
        self.end_token = "E"
        self.pad_token = "P"
        self.vocab = [self.pad_token] + vocab + [self.start_token, self.end_token]
        self.max_length = max_length

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def select_random_char(self) -> str:
        return random.choice(self.vocab[1:-1])

    def select_random_n_char(self, k: int = 1) -> str:
        return "".join([self.select_random_char() for _ in range(k)])

    def get_start_token(self) -> str:
        return self.start_token

    def tokenize(self, word: str, length: int = None) -> torch.Tensor:
        if length is None:
            length = len(word) + 2

        current_index = 0

        pad_size = max(length - len(word) - 2, 0)
        slice_size = length - 2

        vec = torch.empty((length,), dtype=torch.long)

        # add start token
        vec[current_index] = self.vocab.index(self.start_token)
        current_index += 1

        for w in word[:slice_size]:
            vec[current_index] = self.vocab.index(w)
            current_index += 1

        # pad last if needed
        for _ in range(pad_size):
            vec[current_index] = self.vocab.index(self.pad_token)
            current_index += 1

        # add end token
        vec[current_index] = self.vocab.index(self.end_token)

        current_index += 1

        return vec

    def detokenize(self, tokenized_word: torch.Tensor) -> str:
        # seq_len,
        word = []
        for t in tokenized_word[1:]: # skip start token
            # TODO consider multiple words
            ch = self.vocab[t]
            if ch == self.vocab[-1]: # <eos> token
                break
            word.append(ch)
        word = "".join(word)
        word = word.replace(self.pad_token, "") # replace <pad> token with empty
        return word

    def __str__(self):
        return ",".join(self.vocab)