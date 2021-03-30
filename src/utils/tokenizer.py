from typing import List
import torch
import random

class Tokenizer():
    __default_vocab = " '()-abcdefghijklmnopqrstuvwxyzé"

    def __init__(self, vocab: List[str] = [], max_length: int = 45):
        vocab = list(set(vocab + [v for v in self.__default_vocab]))
        vocab = sorted(vocab)
        self.vocab = ["P"] + vocab + ["E"]
        self.max_length = max_length

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def select_random_char(self) -> str:
        return random.choice(self.vocab[1:-1])

    def select_random_n_char(self, k: int = 1) -> str:
        return "".join([self.select_random_char() for _ in range(k)])

    def tokenize(self, word: str, length: int = None, add_end_token: bool = False) -> torch.Tensor:
        mlength = length if length else len(word)
        length = mlength-1 if add_end_token else mlength

        vec = torch.empty((1, mlength), dtype=torch.long)
        # pad first
        for i in range(max(length-len(word), 0)):
            vec[:, i] = self.vocab.index("P")

        for i in range(min(length, len(word))):
            w = word[i]
            vec[:, i + max(length - len(word), 0)] = self.vocab.index(w)

        if add_end_token:
            vec[:, mlength-1] = self.vocab.index("E")

        return vec

    def detokenize(self, tokenized_word: torch.Tensor) -> str:
        # seq_len,
        word = []
        for t in tokenized_word:
            ch = self.vocab[t]
            if ch == self.vocab[-1]: # <eos> token
                break
            word.append(ch)
        word = "".join(word)
        word = word.replace(self.vocab[0], "") # replace <pad> token with empty
        return word

    def __str__(self):
        return ",".join(self.vocab)