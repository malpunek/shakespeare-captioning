# %%
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from tqdm.auto import tqdm

from ..config import coco_train_conf, coco_val_conf, extended_word_map_path, logger
from ..utils import ask_overwrite
from .extract_tagged_lemmas import TaggerFilterLemmatizer

tfl = TaggerFilterLemmatizer()


def map_words(terms, word_map):
    for term in terms:
        if (substitute := word_map.get(term, None)) is not None:
            yield substitute


def caption_to_terms(caption, word_map):
    return list(map_words(tfl(caption), word_map))


def term_annotation(word_map, ann):
    res = ann.copy()
    terms = caption_to_terms(res["caption"], word_map)
    terms = " ".join(terms)
    res["caption"] = terms
    return res


def to_semantic_terms(conf):

    if not Path(extended_word_map_path).exists():
        logger.critical(
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

    with Pool() as p:
        map_annotations = partial(term_annotation, word_map)

        terms_annotations = p.imap(map_annotations, caps["annotations"])
        terms_annotations = tqdm(
            terms_annotations,
            total=len(caps["annotations"]),
            desc="Mapping annotations to semantic term format",
        )
        terms_annotations = list(terms_annotations)

    caps["annotations"] = terms_annotations
    with open(conf["semantic_captions_path"], "w") as f:
        json.dump(caps, f, indent=2)

    return caps


def main():
    for conf in (coco_train_conf, coco_val_conf):
        to_semantic_terms(conf)


if __name__ == "__main__":
    main()
