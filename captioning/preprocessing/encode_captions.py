import json
import logging
from pathlib import Path

from tqdm.auto import tqdm
from itertools import chain
from functools import partial

from ..config import (
    extended_word_map_path,
    max_caption_len,
    coco_train_conf,
    coco_val_conf,
)
from ..utils import WordIdxMap, ask_overwrite


def encode_caption(mapping, caption):
    terms = caption.split()
    terms = terms[: max_caption_len - 2]
    start = (mapping["<start>"],)
    end = (mapping["<end>"],)
    pad = (mapping["<pad>"] for _ in range(max_caption_len - 2 - len(terms)))
    term_len = (len(terms),)
    terms = chain(start, mapping.encode(terms), end, pad, term_len)
    return list(terms)


def map_annotation_using_mapping(mapping, ann):
    res = ann.copy()
    res["caption"] = encode_caption(mapping, res["caption"])
    return res


def encode_semantic_captions(conf):
    if not ask_overwrite(conf["encoded_captions_path"]):
        return

    if not Path(extended_word_map_path).exists():
        logging.critical(
            f"{extended_word_map_path} does not exist. Please create word map first!"
        )
        return
    if not Path(conf["semantic_captions_path"]).exists():
        logging.critical(
            f"{conf['semantic_captions_path']} does not exist."
            "Please extract semantic captions first!"
        )
        return

    with open(extended_word_map_path) as f:
        word_map = json.load(f)

    with open(conf["semantic_captions_path"]) as f:
        caps = json.load(f)

    mapping = WordIdxMap(word_map)
    map_annotations = partial(map_annotation_using_mapping, mapping)

    caps["annotations"] = list(
        tqdm(
            map(map_annotations, caps["annotations"]),
            total=len(caps["annotations"]),
            desc="Encoding captions",
        )
    )
    with open(conf["encoded_captions_path"], "w") as f:
        json.dump(caps, f, indent=2)


def main():
    for conf in (coco_train_conf, coco_val_conf):
        encode_semantic_captions(conf)


if __name__ == "__main__":
    main()
