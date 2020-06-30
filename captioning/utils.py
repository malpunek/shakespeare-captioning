from itertools import chain
from pathlib import Path

from .config import interactive


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
    if not interactive:
        return False

    while (
        response := input(f"{path} already present. Overwrite? [y/N]").lower()
    ) not in "yn":
        print(
            "Expecting one of 'YyNn'."
            f"For default press enter. You've typed: {response}"
        )
    return response == "y"


class WordIdxMap:
    def __init__(self, word_map):
        self.idx2word = list(
            # We want <pad> to be indexed as 0
            chain(["<pad>"], word_map.keys(), ["<unk>", "<start>", "<end>"])
        )
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def __getitem__(self, x):
        if isinstance(x, int):
            return self.idx2word[x]
        else:
            return self.word2idx[x]
