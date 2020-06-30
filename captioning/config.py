import argparse
import glob
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--word_occurance_threshold", type=int, default=200)
parser.add_argument("-i", "--interactive", action="store_true")
args = parser.parse_args()

# Data processing metaparameters
word_occurance_threshold = args.word_occurance_threshold

gpu_num = 0
device_str = "cpu" if not torch.cuda.is_available() else f"cuda:{gpu_num}"
device = torch.device(device_str)

# Paths
mscoco_root_path = Path("/data/mscoco")
mscoco_imgs_path = mscoco_root_path / "train2014"
captions_path = mscoco_root_path / "annotations/captions_train2014.json"

computed_root_path = Path("/data/computed_shake")

if not computed_root_path.exists():
    os.makedirs(computed_root_path, exist_ok=True)

features_path = computed_root_path / "extracted_features.hdf5"
feature_ids_path = computed_root_path / "extracted_ids.npy"

thresh_path = computed_root_path / f"thresh_{word_occurance_threshold}"
if not thresh_path.exists():
    os.makedirs(thresh_path, exist_ok=True)

words_path = computed_root_path / "words.json"
word_map_path = thresh_path / "word_map.json"
semantic_captions_path = thresh_path / "semantic_train2014.json"

plays_path = "/data/shake/merged"
nltk_data_path = "/home/malpunek/.nltk_data"


extended_word_map_path = thresh_path / "word_map_extended.json"
encoded_captions_path = thresh_path / "encoded_captions_train2014.json"

# Various script options

interactive = args.interactive
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
        print("Torch didn't find any available GPUs.")
        return

    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        high, low = torch.cuda.get_device_capability(i)
        message = (
            f"GPU {i}: Device: {torch.cuda.get_device_name(i)}, "
            f"Compute Capability: {high}.{low}"
        )
    print(message)


def pick_gpu():
    global device, device_str

    if not interactive:
        device_str = "cpu" if not torch.cuda.is_available() else "cuda:0"

    elif not torch.cuda.is_available():
        device_str = "cpu"

    elif torch.cuda.device_count() == 1:
        device_str = "cuda:0"

    else:
        n_gpu = torch.cuda.device_count()
        while True:
            response = input("Type in the GPU index to use")
            try:
                x = int(response)
                if not 0 <= x < n_gpu:
                    raise ValueError
            except ValueError:
                print(f"Please supply an integer between {0} and {n_gpu - 1}")
                continue
        device_str = f"cuda:{x}"

    device = torch.device(device_str)
    logging.info(f"Using {device}")
    return device
