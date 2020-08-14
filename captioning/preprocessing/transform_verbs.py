# %%
import json
from operator import itemgetter
from pathlib import Path
from typing import Dict, Optional

from nltk.corpus.reader.wordnet import Synset
from tabulate import tabulate
from tqdm.auto import tqdm

from ..config import (
    load_wn,
    logger,
    word_map_path,
    word_occurance_threshold as threshold,
    words_path,
)
from ..utils import ask_overwrite

# %%


def score(synset: Synset, verbs_to_keep: Dict[str, int]):
    """Computes score to rank the most probable synset

    Args:
        synset (Synset): synset to score
        verbs_to_keep (Dict[str, int]): Other verbs in the dataset

    Returns:
        int: Sum of all appearances of related lemmas in the dataset
    """
    return sum(verbs_to_keep.get(lemma, 0) for lemma in synset.lemma_names())


def find_replacement(verb: str, verbs_to_keep: Dict[str, int]) -> Optional[str]:
    """Tries to find the best replacement for a verb that is already in the dataset

    Args:
        verb (str): The verb to find replacement for
        verbs_to_keep (Dict[str, int]): Other verbs in the dataset

    Returns:
        Optional[str]: Best match in verbs_to_keep or None
    """
    wn = load_wn()

    _score = lambda s: score(s, verbs_to_keep)  # noqa: E731
    scores = sorted(((_score(synset), synset) for synset in wn.synsets(verb, wn.VERB)))
    if not scores:
        return None
    top_score, top_synset = scores[0]
    if top_score > 0:
        lemma_score, top_lemma = sorted(
            (verbs_to_keep.get(lm, 0), lm) for lm in top_synset.lemma_names()
        )[-1]
        return top_lemma
    return None


def log_stats(
    len_words_to_keep, len_words, len_verbs_to_keep, len_verbs, len_additional_verbs
):
    len_nouns_to_keep, len_nouns = (
        len_words_to_keep - len_verbs_to_keep,
        len_words - len_verbs,
    )

    headers = ["POS", "Keep", "Total"]

    counts = [
        ["WORD", len_words_to_keep, len_words],
        ["VERB", len_verbs_to_keep, len_verbs],
        ["NOUN", len_nouns_to_keep, len_nouns],
        ["WORD**", len_words_to_keep + len_additional_verbs, "---"],
        ["VERB**", len_verbs_to_keep + len_additional_verbs, "---"],
    ]

    ratios = [
        ["ALL", len_verbs / (len_words - len_verbs)],
        ["KEEP", len_verbs_to_keep / (len_words_to_keep - len_verbs_to_keep)],
        [
            "KEEP**",
            (len_verbs_to_keep + len_additional_verbs)
            / (len_words_to_keep - len_verbs_to_keep),
        ],
    ]

    msg = (
        f"Got {len_additional_verbs} additional verbs to keep\n"
        + tabulate(counts, headers=headers, tablefmt="github")
        + "\n** With additional verbs\nVerbs/Noun ratio:\n"
        + tabulate(ratios, headers=["Set", "Ratio"], tablefmt="github")
    )

    logger.info(msg)


def main():

    if not Path(words_path).exists():
        logger.critical(
            f"{words_path} does not exist. Please extract tagged lemmas first!"
        )
        return

    if not ask_overwrite(word_map_path):
        return

    # _T are tagged with POS tag (cat_NOUN, sleep_VERB)
    with open(words_path) as f:
        words_T = json.load(f)

    verbs = {word[:-5]: cnt for word, cnt in words_T.items() if word[-4:] == "VERB"}

    words_to_keep_T = {w: cnt for w, cnt in words_T.items() if cnt >= threshold}
    verbs_to_keep = {v: cnt for v, cnt in verbs.items() if cnt >= threshold}

    logger.info(
        f"There are {len(verbs_to_keep)} verbs to keep before sense generalization"
    )

    additional_verbs = {
        verb: find_replacement(verb, verbs_to_keep)
        for verb, cnt in tqdm(verbs.items(), desc="Getting additional verbs to keep")
        if cnt < threshold
    }
    additional_verbs = dict(filter(itemgetter(1), additional_verbs.items()))

    log_stats(
        len(words_to_keep_T),
        len(words_T),
        len(verbs_to_keep),
        len(verbs),
        len(additional_verbs),
    )

    word_map_T = {word: word for word in words_to_keep_T}
    word_map_T.update(
        {f"{verb}_VERB": f"{target}_VERB" for verb, target in additional_verbs.items()}
    )

    logger.info(f"Saving WordMap to {word_map_path}!")

    with open(word_map_path, "w") as f:
        json.dump(word_map_T, f, indent=2)


# %%
if __name__ == "__main__":
    main()
