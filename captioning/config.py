import argparse
import glob
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
from torchvision.transforms import transforms
from torchvision.datasets.coco import CocoCaptions


def lazy(func):
    return lru_cache(maxsize=1)(func)


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--word_occurance_threshold", type=int, default=25)
parser.add_argument("-i", "--interactive", action="store_true")
args = parser.parse_args()

# Data processing metaparameters
word_occurance_threshold = args.word_occurance_threshold

gpu_num = 0
device_str = "cpu" if not torch.cuda.is_available() else f"cuda:{gpu_num}"
device = torch.device(device_str)

first_stage = {"batch_size": 16, "learning_rate": 0.001, "epochs": 10}


max_caption_len = 20

# Paths
mscoco_root_path = Path("/data/mscoco")
computed_root_path = Path("/data/computed_shake")
thresh_path = computed_root_path / f"thresh_{word_occurance_threshold}"

if not computed_root_path.exists():
    os.makedirs(computed_root_path, exist_ok=True)

if not thresh_path.exists():
    os.makedirs(thresh_path, exist_ok=True)

coco_train_conf = {
    "name": "Train",
    "imgs_root_path": mscoco_root_path / "train2014",
    "captions_path": mscoco_root_path / "annotations/captions_train2014.json",
    "features_path": computed_root_path / "train_features.hdf5",
    "semantic_captions_path": thresh_path / "semantic_train2014.json",
    "encoded_captions_path": thresh_path / "encoded_captions_train2014.json",
    "transformed_data_path": thresh_path / "training_full.hdf5",
}

coco_val_conf = {
    "name": "Validation",
    "imgs_root_path": mscoco_root_path / "val2014",
    "captions_path": mscoco_root_path / "annotations/captions_val2014.json",
    "features_path": computed_root_path / "val_features.hdf5",
    "semantic_captions_path": thresh_path / "semantic_val2014.json",
    "encoded_captions_path": thresh_path / "encoded_captions_val2014.json",
}

words_path = computed_root_path / "words.json"
word_map_path = thresh_path / "word_map.json"
extended_word_map_path = thresh_path / "word_map_extended.json"

plays_path = "/data/shake/merged"
nltk_data_path = "/home/malpunek/.nltk_data"


# Various script options

interactive = args.interactive
# TODO timestamps in format?
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)


def setup_nltk():
    import nltk  # noqa: E402

    if nltk.data.path[-1] != nltk_data_path:
        nltk.data.path.append(nltk_data_path)


# ######## Lazy #############


def _get_dataset(dataset_conf):
    t = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return CocoCaptions(
        dataset_conf["imgs_root_path"], dataset_conf["captions_path"], transform=t
    )


@lazy
def coco_captions_train():
    return _get_dataset(coco_train_conf), coco_train_conf


@lazy
def coco_captions_val():
    return _get_dataset(coco_val_conf), coco_val_conf


@lazy
def get_zipped_plays_paths():
    return list(
        zip(
            sorted(glob.glob(f"{plays_path}/*_modern.snt.aligned")),
            sorted(glob.glob(f"{plays_path}/*_original.snt.aligned")),
        )
    )


@lazy
def load_wn():
    setup_nltk()
    from nltk.corpus import wordnet

    return wordnet


@lru_cache(maxsize=8)
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


@lru_cache(maxsize=8)
def load_wn_nlp(model="en_core_web_lg"):
    from spacy_wordnet.wordnet_annotator import WordnetAnnotator

    nlp = load_nlp(model)
    nlp.add_pipe(WordnetAnnotator(nlp.lang), after="tagger")
    return nlp


# ##### UTILS #####


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
        # Not interactive stick with the default
        device_str = "cpu" if not torch.cuda.is_available() else "cuda:0"

    elif not torch.cuda.is_available():
        # Interactive but cuda not available
        device_str = "cpu"

    elif torch.cuda.device_count() == 1:
        # Interactive but only 1 gpu available
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
