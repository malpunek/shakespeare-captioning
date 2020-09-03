from pathlib import Path
import json

import torch

from ..config import (
    device,
    extended_word_map_path,
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
from .first_stage import get_image


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


def main(img_dir):

    img_dir = Path(img_dir).expanduser()
    if not img_dir.is_dir():
        raise RuntimeError(f"{str(img_dir)} is not a directory!")
    mmap, cmap, tmap = get_mappings()
    term_gen, lang_gen = get_models(mmap, cmap, tmap)

    model = SemStyle(term_gen, lang_gen, mmap, tmap, cmap)
    model.eval()

    for sub_path in sorted(img_dir.iterdir()):
        if sub_path.suffix not in (".jpg", ".png"):
            continue
        terms, cap, _ = model(get_image(sub_path))
        _, scap, _ = model(get_image(sub_path), True)

        print(f"![Sample image](https://students.mimuw.edu.pl/~sm371229/{sub_path})")
        for meat in (
            terms,
            "Normal: " + " ".join(cap[1:-1]),
            "Styled: " + " ".join(scap[1:-1]),
        ):
            print("-" * 20)
            print(meat)
        print("-" * 20)
        print("\n\n")


if __name__ == "__main__":
    import sys
    import select

    print(f"Evaluating {last_checkpoint_path(2)}")

    print("Provide img dir\n\n")
    i, _, _ = select.select([sys.stdin], [], [], 15)
    img_dir = sys.stdin.readline().strip() if i else "mini_val"
    main(img_dir)
