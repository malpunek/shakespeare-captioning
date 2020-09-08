from itertools import chain
from math import prod
from pathlib import Path

import torch


def get_yn_response(question: str) -> bool:
    while (response := input(question).lower()) not in "yn":
        print(
            "Expecting one of 'YyNn'."
            f"For default press enter. You've typed: {response}"
        )
    return response == "y"


def ask_overwrite(path: str) -> bool:
    """
    Args:
        path (str): Path to file

    Returns:
        bool: True if:
            1) Path does not exist (safe to create the file)
            2) user confirmed overwrite
    """
    if not Path(path).exists():
        return True

    # path exists
    from .config import interactive

    if not interactive:
        return False

    return get_yn_response(f"{path} already present. Overwrite? [y/N]")


class WordIdxMap:
    def __init__(self, words):
        if isinstance(words, dict):
            words = words.keys()

        self.idx2word = list(
            # We want <pad> to be indexed as 0
            chain(
                ["<pad>"],
                words,
                ["<unk>", "<start>", "<end>", "<shake_modern>", "<shake_orig>"],
            )
        )
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def __getitem__(self, x):
        if isinstance(x, torch.Tensor) and prod(x.size()) == 1:
            x = x.item()
        if isinstance(x, int):
            return self.idx2word[x]
        return self.word2idx.get(x, self.word2idx["<unk>"])

    def __len__(self):
        return len(self.idx2word)

    def encode(self, words):
        return (self[w] for w in words)

    def decode(self, encoded_caption):
        return (self[idx] for idx in encoded_caption)

    def prepare_for_training(self, words, max_caption_len, terms=False):
        words = words[: max_caption_len - 2]
        # Dont surround with start end if in terms mode
        start = (self["<start>"],) * (not terms)
        end = (self["<end>"],) * (not terms)
        pad = (self["<pad>"],) * (max_caption_len - len(words) - len(start) - len(end))
        term_len = (len(words),)
        words = chain(start, self.encode(words), end, pad, term_len)
        return list(words)
