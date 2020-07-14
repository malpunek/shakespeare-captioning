# %%
import json
import logging
from functools import partial
from pathlib import Path

from tqdm.auto import tqdm

from ..config import coco_train_conf, coco_val_conf, extended_word_map_path
from ..utils import ask_overwrite
from .extract_tagged_lemmas import caption_to_tagged_lemmas


def caption_to_terms(caption, word_map):
    def map_words(terms):
        for term in terms:
            if (substitute := word_map.get(term, None)) is not None:
                yield substitute

    return list(map_words(caption_to_tagged_lemmas(caption)))


def term_annotation(word_map, ann):
    res = ann.copy()
    terms = caption_to_terms(res["caption"], word_map)
    terms = " ".join(terms)
    res["caption"] = terms
    return res


def to_semantic_terms(conf):

    if not Path(extended_word_map_path).exists():
        logging.critical(
            f"{extended_word_map_path} does not exist. Please extract tagged lemmas first!"  # noqa
        )
        return

    if not ask_overwrite(conf["semantic_captions_path"]):
        return

    with open(conf["captions_path"]) as caps_f, open(
        extended_word_map_path
    ) as word_map_f:
        caps = json.load(caps_f)
        word_map = json.load(word_map_f)

    map_annotations = partial(term_annotation, word_map)
    terms_annotations = list(
        tqdm(
            map(map_annotations, caps["annotations"]),
            total=len(caps["annotations"]),
            desc="Mapping annotations to semantic term format",
        )
    )

    caps["annotations"] = terms_annotations
    with open(conf["semantic_captions_path"], "w") as f:
        json.dump(caps, f, indent=2)

    return caps


def main():
    for conf in (coco_train_conf, coco_val_conf):
        to_semantic_terms(conf)


if __name__ == "__main__":
    main()
