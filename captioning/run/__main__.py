import json

import torch
from PIL import Image

from ..config import (
    device,
    extended_word_map_path,
    image_transform,
    language_data_path,
    last_checkpoint_path,
)
from ..dataset import BalancedLanguageDataset
from ..model import (
    ImgToTermNet,
    LanguageGenerator,
    SemStyle,
    SentenceDecoderWithAttention,
    TermDecoder,
    TermEncoder,
)
from ..train.second_stage import filter_coco, filter_shake
from ..utils import WordIdxMap


def run_path(model, mapping, img_path):
    with torch.no_grad():
        img = Image.open(img_path)
        img = image_transform(img)
        img = img.to(device).unsqueeze(0)

        terms, _ = model(img, mapping)
        return terms


def get_mappings():
    with open(extended_word_map_path) as f:
        word_map = json.load(f)

    mapping = WordIdxMap(word_map)
    dataset = BalancedLanguageDataset(
        language_data_path,
        filter_shakespear=filter_shake,
        filter_coco=filter_coco,
        to_tensor=True,
    )

    cmapping, tmapping = dataset.get_mappings

    return mapping, cmapping, tmapping


def get_models(mapping, cmapping, tmapping):
    dec = TermDecoder(len(mapping), 2048, 2048)
    dec.load_state_dict(torch.load(last_checkpoint_path(), map_location="cpu"))
    first_stage = ImgToTermNet(dec)
    first_stage = first_stage.to(device)
    first_stage = first_stage.eval()

    enc = TermEncoder(len(tmapping), 2048)
    dec = SentenceDecoderWithAttention(len(cmapping), 2048, len(cmapping))

    lang = LanguageGenerator(enc, dec)
    lang.load_state_dict(torch.load(last_checkpoint_path(2), map_location="cpu"))
    lang.eval()

    return first_stage, lang


def main():

    mmap, cmap, tmap = get_mappings()
    term_gen, lang_gen = get_models(mmap, cmap, tmap)

    model = SemStyle(term_gen, lang_gen, mmap, tmap, cmap)
    model.eval()

    # TODO
    img_path = "/home/malpunek/Downloads/mini_val/COCO_val2014_000000000164.jpg"

    img = Image.open(img_path)
    img = image_transform(img)
    img = img.to(device).unsqueeze(0)

    print(model(img))


if __name__ == "__main__":
    main()
