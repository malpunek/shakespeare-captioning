import json
import logging
from pathlib import Path

from tqdm.auto import tqdm
from itertools import chain
from functools import partial

from ..config import (
    encoded_captions_path,
    extended_word_map_path,
    max_caption_len,
    semantic_captions_path,
)
from ..utils import WordIdxMap, ask_overwrite


def encode_caption(mapping, caption):
    terms = caption.split()
    terms = terms[: max_caption_len - 2]
    start = (mapping["<start>"],)
    pad = (mapping["<pad>"] for _ in range(max_caption_len - 2 - len(terms)))
    end = (mapping["<end>"], len(terms))
    terms = chain(start, (mapping[term] for term in terms), pad, end)
    return list(terms)


def map_annotation_using_mapping(mapping, ann):
    res = ann.copy()
    res["caption"] = encode_caption(mapping, res["caption"])
    return res


def main():

    if not ask_overwrite(encoded_captions_path):
        return

    if not Path(extended_word_map_path).exists():
        logging.critical(
            f"{extended_word_map_path} does not exist. Please create word map first!"
        )
        return
    if not Path(semantic_captions_path).exists():
        logging.critical(
            f"{semantic_captions_path} does not exist."
            "Please extract semantic captions first!"
        )
        return

    with open(extended_word_map_path) as f:
        word_map = json.load(f)

    with open(semantic_captions_path) as f:
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
    with open(encoded_captions_path, "w") as f:
        json.dump(caps, f, indent=2)


if __name__ == "__main__":
    main()
