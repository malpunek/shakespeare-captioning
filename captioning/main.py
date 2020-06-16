from collections import Counter
import json
from itertools import islice, chain
from pathlib import Path

import fire
import h5py
import torch
from torchvision import transforms
from torchvision.datasets.coco import CocoCaptions
from tqdm.auto import tqdm

from model import FeatureExtractor


class Commands:
    """
    Args:
        cpu: Use cpu even if GPU is available
        gpu_num: Use the GPU number GPU_NUM. Use `main.py lgpu` to \
list available gpus
    """

    def __init__(self, cpu: bool = False, gpu_num: int = 0):

        device = "cpu" if cpu or not torch.cuda.is_available() else f"cuda:{gpu_num}"
        if device == "cpu":
            print("Using cpu!")
        else:
            print(f"Using {torch.cuda.get_device_name(gpu_num)}")
        self.device = torch.device(device)

    def extract_word_map(self, *captions_files, save_path="word_map.json"):
        captions = chain(
            json.load(open(path, "r")) for path in captions_files
        )  # [[c0_1, c0_2..], [c1_0, c1_1..]..]
        captions = chain.from_iterable(captions)  # flat list
        translation = str.maketrans("'", " ", ",.")
        words = chain(c.translate(translation).split(" ") for c in captions])
        words = map(str.lower, words)
        



if __name__ == "__main__":
    fire.Fire(Commands)
