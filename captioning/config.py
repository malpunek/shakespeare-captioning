import glob
import logging
from functools import lru_cache

# Paths
mscoco_root_path = "/data/mscoco"
captions_path = f"{mscoco_root_path}/annotations/captions_train2014.json"
words_path = f"{mscoco_root_path}/words.json"
word_map_path = f"{mscoco_root_path}/word_map.json"
word_map_path = f"{mscoco_root_path}/big_word_map.json"

shakespeare_path = "/data/shake"
plays_path = "/data/shake/merged"
nltk_data_path = "/home/malpunek/.nltk_data"

# Data processing metaparameters
word_occurance_threshold = 200

# Various script options

interactive = True
# TODO timestamps in format?
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)


def get_zipped_plays_paths():
    return list(
        zip(
            sorted(glob.glob(f"{plays_path}/*_modern.snt.aligned")),
            sorted(glob.glob(f"{plays_path}/*_original.snt.aligned")),
        )
    )


def setup_nltk():
    import nltk  # noqa: E402

    if nltk.data.path[-1] != nltk_data_path:
        nltk.data.path.append(nltk_data_path)


@lru_cache
def load_wn():
    setup_nltk()
    from nltk.corpus import wordnet

    return wordnet


@lru_cache
def load_nlp(model="en_core_web_lg"):
    import spacy

    logging.info(f"Loading spacy model {model}..")

    def worker():
        try:
            return spacy.load(model)
        except OSError:
            logging.info("Model not found. Downloading...")
            spacy.cli.download(model)
            logging.info("Spacy model downloaded!")
            return spacy.load(model)

    nlp = worker()
    logging.info(f"Spacy model {model} loaded!")
    return nlp


@lru_cache
def load_wn_nlp(model="en_core_web_lg"):
    from spacy_wordnet.wordnet_annotator import WordnetAnnotator

    nlp = load_nlp(model)
    nlp.add_pipe(WordnetAnnotator(nlp.lang), after="tagger")
    return nlp
