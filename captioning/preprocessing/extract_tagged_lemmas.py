# %%
import json
import logging
from collections import Counter
from multiprocessing import Pool
from operator import itemgetter
from typing import Iterable

import contractions
from tqdm.auto import tqdm

from ..config import coco_train_conf, load_nlp, words_path
from ..utils import ask_overwrite

# https://spacy.io/api/annotation
POS_OUT = [
    "PUNCT",
    "ADJ",
    "CCONJ",
    "CONJ",
    "SCONJ",
    "NUM",
    "PRON",
    "ADV",
    "ADP",
    "DET",
    "INTJ",
    "SPACE",
    "SYM",
    "PART",
    "X",
    "AUX",
    # TODO should we keep PROPNS
    "PROPN",
]

POS_IN = [
    "NOUN",
    "VERB",
]


# Local functions (or local lambdas) aren't picklable and thus can't be
# run via multiprocessing.Pool. And I wanted to keep all this logic
# in enclosed scope
# (https://stackoverflow.com/questions/4827432/)
class TaggerFilterLemmatizer:
    """Converts captions to tagged lemmas.
    """

    @staticmethod
    def non_semantic_word_filter(x):
        return x.pos_ in POS_IN

    @staticmethod
    def lemmatize_and_tag(x):
        return f"{x.lemma_.lower()}_{x.pos_}"

    def caption_to_tagged_lemmas(self, caption: str) -> Iterable[str]:
        nlp = load_nlp()
        caption = contractions.fix(caption)
        tokens = nlp(caption)
        # A. Filtering non-semantic words
        tokens = filter(self.non_semantic_word_filter, tokens)
        # B. Lemmatization and tagging.
        tokens = map(self.lemmatize_and_tag, tokens)
        return list(tokens)

    def __call__(self, caption: str) -> Iterable[str]:
        return self.caption_to_tagged_lemmas(caption)


# %%
def main():

    if not ask_overwrite(words_path):
        return

    logging.info("Creating new tagged lemma dict.")

    logging.info("Loading captions...")
    with open(coco_train_conf["captions_path"]) as f:
        caps = json.load(f)["annotations"]
    logging.info("Done!")

    words = Counter()

    with Pool() as p:
        tagged_lemmas = p.imap(
            TaggerFilterLemmatizer(), map(itemgetter("caption"), caps), chunksize=8
        )

        for lemmas in tqdm(
            tagged_lemmas, desc="Extracting & counting words...", total=len(caps)
        ):
            words.update(lemmas)

    with open(words_path, "w") as f:
        json.dump(dict(words), f, indent=2)

    logging.info(f"{len(words)} words saved to {words_path}!")


if __name__ == "__main__":
    main()
