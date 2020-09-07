from pathlib import Path

import torch
from PIL import Image

from ..config import (
    coco_train_conf,
    device,
    image_transform,
    last_checkpoint_path,
    shakespare_conf,
)
from ..dataset import SemStyleDataset
from ..train.misc import filter_fn
from ..model import ImgToTermNet, TermDecoder


def get_image(img_path):
    img = Image.open(img_path)
    img = image_transform(img)
    return img.to(device).unsqueeze(0)


def run_path(model, mapping, img_path):
    with torch.no_grad():
        terms, _ = model(get_image(img_path), mapping)
        return terms


def main():

    mapping = SemStyleDataset(
        coco_train_conf["final"], shakespare_conf["final"], filter_fn=filter_fn
    ).get_term_mapping
    vocab_size = len(mapping)

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
