import json
from itertools import chain
from operator import itemgetter
from pathlib import Path

from tqdm.auto import tqdm

from ..config import (
    extended_word_map_path,
    get_zipped_plays_paths,
    logger,
    word_map_path,
    words_path,
)
from ..utils import ask_overwrite
from .extract_tagged_lemmas import TaggerFilterLemmatizer
from .transform_verbs import find_replacement


def file_to_list(path):
    with open(path) as f:
        return list(f)


def strip_POS_tag(word):
    return word[:-5]


def main():

    plays = get_zipped_plays_paths()

    if not ask_overwrite(extended_word_map_path):
        return

    for p in (word_map_path, words_path):
        if not Path(p).exists():
            logger.critical(f"{p} does not exist. Please create word map first!")
            return

    with open(word_map_path) as f:
        word_map_T = json.load(f)

    with open(words_path) as f:
        words_count_T = json.load(f)

    # TODO rerunning this causes KeyError cuz added verbs are not in words_count_T
    verbs_to_keep = {
        word[:-5]: words_count_T[word]
        for word in word_map_T.keys()
        if word[-4:] == "VERB"
    }

    modern_captions = list(
        chain.from_iterable((file_to_list(modern) for modern, _ in plays))
    )
    tfl = TaggerFilterLemmatizer()
    modern_lemmas_T = chain.from_iterable(
        [
            tfl(c)
            for c in tqdm(
                modern_captions, desc="Extracting tagged lemmas from modern plays"
            )
        ]
    )

    modern_verbs = filter(lambda x: x[-4:] == "VERB", modern_lemmas_T)
    modern_verbs = set(map(strip_POS_tag, modern_verbs))
    modern_verbs = list(filter(lambda x: x not in verbs_to_keep, modern_verbs))
    logger.info(f"Trying to preserve {len(modern_verbs)} shakespeare-only verbs")

    additional_verbs = {
        verb: find_replacement(verb, verbs_to_keep)
        for verb in tqdm(modern_verbs, desc="Finding replacements")
    }
    additional_verbs = dict(filter(itemgetter(1), additional_verbs.items()))

    logger.info(f"Found {len(additional_verbs)} additional verbs")

    word_map_T.update(
        {f"{verb}_VERB": f"{target}_VERB" for verb, target in additional_verbs.items()}
    )
    with open(extended_word_map_path, "w") as f:
        json.dump(word_map_T, f, indent=2)


if __name__ == "__main__":
    main()
