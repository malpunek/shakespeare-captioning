# %%
import json
import logging
from collections import Counter
from itertools import chain, accumulate

from tabulate import tabulate
from tqdm.auto import tqdm

from ..config import get_zipped_plays_paths, word_map_path
from .extract_tagged_lemmas import caption_to_tagged_lemmas


# %%
def check_word_map_intersection(captions, word_map):

    all_keepers = set(word_map.keys())

    def count(caption):
        keeps = list(caption_to_tagged_lemmas(caption))
        return sum(1 for word in keeps if word in all_keepers)

    cts = [count(line) for line in tqdm(captions)]

    return Counter(cts)


def main():
    with open(word_map_path) as f:
        word_map = json.load(f)

    plays = get_zipped_plays_paths()

    mls, ols = [], []

    for mf_path, of_path in plays:
        with open(mf_path) as mf, open(of_path) as of:
            mls.append(list(mf))
            ols.append(list(of))

    mls = list(chain.from_iterable(mls))
    ols = list(chain.from_iterable(ols))

    intersection = check_word_map_intersection(mls, word_map)
    intersection = dict(sorted(intersection.items()))

    cum_sum = {
        k: v
        for k, v in zip(
            reversed(intersection.keys()), accumulate(reversed(intersection.values()))
        )
    }

    logging.info(
        (
            "\nNumber N of semantic terms/"
            "Number of sentences from plays that produce N semantic terms\n"
            "Backwards cumulative sum in the lower row\n"
        )
        + tabulate([intersection, cum_sum], headers="keys", tablefmt="github")
    )


# %%

if __name__ == "__main__":
    main()
