# %%
import json
import logging
from collections import Counter

from tqdm.auto import tqdm

from ..config import captions_path, load_nlp, words_path
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


def caption_to_tagged_lemmas(caption: str):
    nlp = load_nlp()
    tokens = nlp(caption)
    # A. Filtering non-semantic words
    tokens = filter(lambda x: x.pos_ in POS_IN, tokens)
    # B. Lemmatization and tagging.
    tokens = map(lambda t: f"{t.lemma_.lower()}_{t.pos_}", tokens)
    return tokens


def get_captions():
    with open(captions_path) as f:
        return json.load(f)["annotations"]


# %%
def main():

    if not ask_overwrite(words_path):
        return

    logging.info("Creating new tagged lemma dict.")

    logging.info("Loading captions...")
    caps = get_captions()
    logging.info("Done!")

    words = Counter()
    for c in tqdm(caps, desc="Extracting & counting words..."):
        words.update(caption_to_tagged_lemmas(c["caption"]))

    with open(words_path, "w") as f:
        json.dump(dict(words), f)

    logging.info(f"{len(words)} words saved to {words_path}!")


if __name__ == "__main__":
    main()
