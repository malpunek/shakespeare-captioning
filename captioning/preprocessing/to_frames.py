import json
import string
from collections import Counter
from itertools import chain
from operator import itemgetter
from pathlib import Path

import contractions
from tqdm.auto import tqdm

from ..config import (
    coco_train_conf,
    coco_val_conf,
    get_zipped_plays_paths,
    shakespare_conf,
)


def cap_to_ascii(cap):
    cap = contractions.fix(cap)
    cap = "".join(filter(lambda x: x in string.printable, cap)).strip()
    return cap.replace("\n", " ")


def make_coco_basic(in_path, out_path):
    if Path(out_path).exists():
        return

    with open(in_path) as f:
        anns = json.load(f)["annotations"]

    coco = [
        {"caption": cap_to_ascii(ann["caption"]), "img_id": ann["image_id"]}
        for ann in tqdm(anns, desc="Creating coco basic..")
    ]
    # Filtration is needed to work nicely with opensesame
    coco = list(filter(lambda x: not x["caption"][0].isdigit(), coco))
    with open(out_path, "wt") as f:
        json.dump(
            coco, f, indent=2,
        )


def make_shake_basic(out_path):
    if Path(out_path).exists():
        return

    modern_lines, old_lines = [], []

    for mf_path, of_path in get_zipped_plays_paths():
        with open(mf_path) as mf, open(of_path) as of:
            modern_lines.append(list(mf))
            old_lines.append(list(of))

    modern_lines = list(chain.from_iterable(modern_lines))
    old_lines = list(chain.from_iterable(old_lines))

    with open(out_path, "wt") as f:
        json.dump(
            [
                {"caption": cap_to_ascii(modern), "original": cap_to_ascii(original)}
                for modern, original in zip(
                    tqdm(modern_lines, desc="Creating shake basic"), old_lines
                )
            ],
            f,
            indent=2,
        )


# %% [markdown]
# ## To TXT

# %%


def to_txt(file_in, file_out):

    if Path(file_out).exists():
        return

    with open(file_in) as f:
        txt_caps = json.load(f)

    total = len(txt_caps)
    txt_caps = map(itemgetter("caption"), txt_caps)
    txt_caps = map(cap_to_ascii, txt_caps)
    with open(file_out, "wt") as f:
        for txt_cap in tqdm(txt_caps, total=total, desc=f"Creating {file_out}.."):
            if txt_cap:
                f.write(txt_cap + "\n")


# %% [markdown]
# # RUN OPENSESAME ON TRAIN.TXT AND SHAKE.TXT

# %%
def split_on_empty(some_list):
    res = []
    for element in some_list:
        if element:
            res.append(element)
        else:
            yield res
            res = []
    if res:
        yield res


VERB_POS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]  # TODO check MD == modal
NOUN_POS = ["NN", "NNS", "NNP", "NNPS"]


def extract_terms(all_frames):
    words = list(map(str.split, all_frames[0][1:]))
    nouns = [f"{w[3]}_NOUN" for w in words if w[3] != "UNK" and w[5] in NOUN_POS]

    all_frames = list(map(lambda x: x[1:], all_frames))
    all_frames = chain.from_iterable(all_frames)
    words = list(map(str.split, all_frames))
    verb_frames = [f"{w[13]}_FRAME" for w in words if w[13] != "_" and w[5] in VERB_POS]

    return nouns + verb_frames


def match(file_conll, file_in, file_out):
    try:
        with open(file_conll) as f:
            lines = list(f)
    except FileNotFoundError:
        raise RuntimeError(
            (
                "Please run open-sesame (https://github.com/swabhs/open-sesame)"
                f"with sesame.patch applied and place the output at {file_conll}"
            )
        )

    with open(file_in) as f:
        captions = json.load(f)

    lines = map(str.strip, lines)
    frames = list(split_on_empty(lines))  # [f1, f1, f1, f2, f5, f5]

    sentences = [f[0] for f in frames]

    sentences = Counter(sentences)  # [sent1: 3, sent2: 1, sent5: 2]
    sent_list = list(sentences)  # [sent1, sent2, sent5]

    caps_with_frames = []
    frame_idx = sentence_idx = 0

    for i, cap in enumerate(tqdm(captions, desc="Matching..")):
        current_sentence = sent_list[sentence_idx]

        if current_sentence == cap["caption"]:
            matches_len = sentences[current_sentence]
            terms = extract_terms(frames[frame_idx : frame_idx + matches_len])

            caps_with_frames.append({**cap, "terms": terms})
            sentence_idx += 1
            frame_idx += matches_len
        else:
            caps_with_frames.append({**cap, "terms": []})

    with open(file_out, "wt") as f:
        json.dump(caps_with_frames, f, indent=2)


def main():
    make_coco_basic(coco_train_conf["original"], coco_train_conf["basic"])
    make_coco_basic(coco_val_conf["original"], coco_val_conf["basic"])
    make_shake_basic(shakespare_conf["basic"])

    for conf in (coco_train_conf, shakespare_conf, coco_val_conf):
        to_txt(conf["basic"], conf["txt"])
        match(conf["conll"], conf["basic"], conf["frames"])


if __name__ == "__main__":
    main()
