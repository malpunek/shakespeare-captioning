import argparse
import glob
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
from torchvision.datasets.coco import CocoCaptions
from torchvision.transforms import transforms


def lazy(func):
    return lru_cache(maxsize=1)(func)


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--word_occurance_threshold", type=int, default=25)
parser.add_argument("-i", "--interactive", action="store_true")
parser.add_argument("-x", "--experiment", type=int, default=0)
args = parser.parse_args()

# Data processing metaparameters
word_occurance_threshold = args.word_occurance_threshold

gpu_num = 0
device_str = "cpu" if not torch.cuda.is_available() else f"cuda:{gpu_num}"
device = torch.device(device_str)

first_stage = {"batch_size": 16, "learning_rate": 0.001, "epochs": 10}
second_stage = {"batch_size": 32, "learning_rate": 0.001, "epochs": 30}
experiment_folder = Path(f"runs/exp_{args.experiment:03d}")

max_caption_len = 20

# Paths
mscoco_root_path = Path("/data/mscoco")
computed_path = Path("/data/computed_shake/")
new_format_path = Path("/data/new_format")


for p in (computed_path, new_format_path, experiment_folder):
    if not p.exists():
        os.makedirs(p, exist_ok=True)

coco_train_conf = {
    "name": "Train",
    "imgs_root_path": mscoco_root_path / "train2014",
    "features": computed_path / "train_features.hdf5",
    "original": mscoco_root_path / "annotations/captions_train2014.json",
    "basic": new_format_path / "train_basic.json",
    "txt": new_format_path / "train.txt",
    "conll": new_format_path / "train.conll",
    "final": new_format_path / "train_final.json",
    "frames": new_format_path / "train_frames.json",
}

coco_val_conf = {
    "name": "Validation",
    "imgs_root_path": mscoco_root_path / "val2014",
    "features": computed_path / "val_features.hdf5",
    "original": mscoco_root_path / "annotations/captions_val2014.json",
    "basic": new_format_path / "val_basic.json",
    "txt": new_format_path / "val.txt",
    "conll": new_format_path / "val.conll",
    "final": new_format_path / "val_final.json",
    "frames": new_format_path / "val_frames.json",
}

shakespare_conf = {
    "name": "Shakespare",
    "basic": new_format_path / "shake_basic.json",
    "txt": new_format_path / "shake.txt",
    "conll": new_format_path / "shake.conll",
    "final": new_format_path / "shake_final.json",
    "frames": new_format_path / "shake_frames.json",
}


plays_path = "/data/shake/merged"
nltk_data_path = "/home/malpunek/.nltk_data"

# Various script options

interactive = args.interactive

logging.basicConfig(format="LIB %(name)s: %(message)s")

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger = logging.getLogger(__package__)
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)


def last_checkpoint_path(stage=1):
    assert stage in (1, 2)
    stage = "model" if stage == 1 else "lang"
    return sorted(experiment_folder.glob(f"{stage}*.pth"))[-1]


def setup_nltk():
    import nltk  # noqa: E402

    if nltk.data.path[-1] != nltk_data_path:
        nltk.data.path.append(nltk_data_path)


image_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ######## Lazy #############
def _get_dataset(dataset_conf):

    return CocoCaptions(
        dataset_conf["imgs_root_path"],
        dataset_conf["original"],
        transform=image_transform,
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


@lazy
def load_framenet():
    setup_nltk()
    from nltk.corpus import framenet

    return framenet


@lru_cache(maxsize=8)
def load_nlp(model="en_core_web_lg"):
    import spacy

    logger.info(f"Loading spacy model {model}..")

    def worker():
        try:
            return spacy.load(model)
        except OSError:
            logger.info("Model not found. Downloading...")
            spacy.cli.download(model)
            logger.info("Spacy model downloaded!")
            return spacy.load(model)

    nlp = worker()
    logger.info(f"Spacy model {model} loaded!")
    return nlp


@lru_cache(maxsize=8)
def load_wn_nlp(model="en_core_web_lg"):
    from spacy_wordnet.wordnet_annotator import WordnetAnnotator

    nlp = load_nlp(model)
    nlp.add_pipe(WordnetAnnotator(nlp.lang), after="tagger")
    return nlp


# ##### TRAINING #####
@lazy
def first_stage_dataset():
    # TODO some other way of switching to QuickCoco
    from .dataset import AllTermsDataset  # , QuickCocoDataset

    args = [
        coco_train_conf["features"],
        coco_train_conf["final"],
        shakespare_conf["final"],
    ]
    # QuickCocoDataset(*args, filter_fn=filter_short,)
    return AllTermsDataset(*args)


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
    logger.info(f"Using {device}")
    return device
