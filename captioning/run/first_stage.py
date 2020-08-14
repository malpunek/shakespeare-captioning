import json
from pathlib import Path

import torch
from PIL import Image

from ..config import (
    device,
    extended_word_map_path,
    image_transform,
    last_checkpoint_path,
)
from ..model import ImgToTermNet, TermDecoder
from ..utils import WordIdxMap


def run_path(model, mapping, img_path):
    with torch.no_grad():
        img = Image.open(img_path)
        img = image_transform(img)
        img = img.to(device).unsqueeze(0)

        terms, _ = model(img, mapping)
        return terms


def main():
    with open(extended_word_map_path) as f:
        word_map = json.load(f)

    mapping = WordIdxMap(word_map)
    vocab_size = len(word_map) + 4  # <start>, <unk>, <pad>, <end>

    dec = TermDecoder(vocab_size, 2048, 2048)
    dec.load_state_dict(torch.load(last_checkpoint_path(), map_location="cpu"))
    model = ImgToTermNet(dec)
    model = model.to(device)
    model = model.eval()

    try:
        while path := input(
            "Enter path to either (1): image (2): folder with images. To exit leave empty or press Ctrl+D\n"  # noqa
        ):
            path = Path(path).expanduser()
            if not path.exists():
                print(f"{path} does not exist")
                continue

            if path.is_file():
                terms = run_path(model, mapping, path)
                print(f"Terms for {path}: {terms}")
                continue

            if path.is_dir():
                for sub_path in sorted(path.iterdir()):
                    if sub_path.suffix in [".jpg", ".png"]:
                        terms = run_path(model, mapping, sub_path)
                        print(f"Terms for {sub_path}: {terms}")

    except EOFError:
        pass
    finally:
        print("Goodbye my love!")


if __name__ == "__main__":
    main()
