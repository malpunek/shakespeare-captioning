from pathlib import Path

import torch
from tqdm.auto import tqdm

from ..config import (
    coco_train_conf,
    device,
    last_checkpoint_path,
    shakespare_conf,
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
from ..train.second_stage import filter_short
from .first_stage import get_image


def get_mappings():
    # TODO get rid of redundant mapping
    dataset = BalancedLanguageDataset(
        coco_train_conf["final"],
        shakespare_conf["final"],
        encode=False,
        # filter_fn=filter_fn,
    )

    tmap = dataset.get_term_mapping

    dataset = BalancedLanguageDataset(
        coco_train_conf["final"],
        shakespare_conf["final"],
        encode=False,
        filter_fn=filter_short,
    )
    tmap2, cmapping = dataset.get_term_mapping, dataset.get_cap_mapping

    return cmapping, tmap, tmap2


def get_models(cmapping, tmapping, tmap2):
    dec = TermDecoder(len(tmapping), 2048, 2048)
    dec.load_state_dict(torch.load(last_checkpoint_path(), map_location="cpu"))
    first_stage = ImgToTermNet(dec)
    first_stage = first_stage.to(device)
    first_stage = first_stage.eval()

    enc = TermEncoder(len(tmap2), 2048)
    dec = SentenceDecoderWithAttention(len(cmapping), 2048, len(cmapping))

    lang = LanguageGenerator(enc, dec)
    lang.load_state_dict(torch.load(last_checkpoint_path(2), map_location="cpu"))
    lang.eval()

    return first_stage, lang


def main(img_dir):

    img_dir = Path(img_dir).expanduser()
    if not img_dir.is_dir():
        raise RuntimeError(f"{str(img_dir)} is not a directory!")
    cmap, tmap1, tmap = get_mappings()
    term_gen, lang_gen = get_models(cmap, tmap1, tmap)

    model = SemStyle(term_gen, lang_gen, tmap1, tmap, cmap)
    model.eval()

    for sub_path in tqdm(sorted(img_dir.iterdir()), desc="Computing captions.."):
        if sub_path.suffix not in (".jpg", ".png"):
            continue
        terms, cap, _ = model(get_image(sub_path))
        _, so_cap, _ = model(get_image(sub_path), "<shake_orig>")
        _, sm_cap, _ = model(get_image(sub_path), "<shake_modern>")

        print(f"![Sample image](https://students.mimuw.edu.pl/~sm371229/{sub_path})")
        for meat in (
            terms,
            "Normal: " + " ".join(cap[1:-1]),
            "Styled Original: " + " ".join(so_cap[1:-1]),
            "Styled Modern: " + " ".join(sm_cap[1:-1]),
        ):
            print("-" * 20)
            print(meat)
        print("-" * 20)
        print("\n\n")


if __name__ == "__main__":
    import sys
    import select

    print(f"Evaluating {last_checkpoint_path(), last_checkpoint_path(2)}")
    print("Provide img dir")
    i, _, _ = select.select([sys.stdin], [], [], 15)
    img_dir = sys.stdin.readline().strip() if i else "mini_val"
    img_dir = img_dir or "mini_val"
    print(f"Evaluating images from {img_dir}")
    main(img_dir)
