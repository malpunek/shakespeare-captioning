import glob
import logging
from functools import lru_cache

import torch

# Paths
mscoco_root_path = "/data/mscoco"
captions_path = f"{mscoco_root_path}/annotations/captions_train2014.json"
words_path = f"{mscoco_root_path}/words.json"
word_map_path = f"{mscoco_root_path}/word_map.json"
word_map_path = f"{mscoco_root_path}/big_word_map.json"

features_path = f"{mscoco_root_path}/extracted_features.hdf5"
feature_ids_path = f"{mscoco_root_path}/extracted_ids.npy"
semantic_captions_path = f"{mscoco_root_path}/annotations/semantic_train2014.json"

shakespeare_path = "/data/shake"
plays_path = "/data/shake/merged"
nltk_data_path = "/home/malpunek/.nltk_data"

# Data processing metaparameters
word_occurance_threshold = 200

gpu_num = 0
device_str = "cpu" if not torch.cuda.is_available() else f"cuda:{gpu_num}"
device = torch.device(device_str)

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


def lgpu():
    """Lists the GPUs found by PyTorch"""
    if not torch.cuda.is_available():
        logging.info("Torch didn't find any available GPUs.")

    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        high, low = torch.cuda.get_device_capability(i)
        message = (
            f"GPU {i}: Device: {torch.cuda.get_device_name(i)}, "
            f"Compute Capability: {high}.{low}"
        )
        logging.info(message)
