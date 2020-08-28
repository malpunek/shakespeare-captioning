import json
import pickle
import re
from itertools import chain
from operator import itemgetter
from pathlib import Path

import contractions
from multiprocess import Pool
from tqdm import tqdm

from ..config import coco_train_conf, get_zipped_plays_paths, language_data_path
from .extract_tagged_lemmas import TaggerFilterLemmatizer

mls, ols = [], []

for mf_path, of_path in get_zipped_plays_paths():
    with open(mf_path) as mf, open(of_path) as of:
        mls.append(list(mf))
        ols.append(list(of))

mls = list(chain.from_iterable(mls))
ols = list(chain.from_iterable(ols))


tfl = TaggerFilterLemmatizer()

with Pool() as pool:
    mls_tagged = pool.imap(TaggerFilterLemmatizer(), mls, chunksize=8)
    mls_tagged = list(tqdm(mls_tagged, desc="Tagging modern sentences", total=len(mls)))


print(mls[50:60])
print(ols[50:60])
print(mls_tagged[50:60])


def to_words(caption):
    caption = contractions.fix(caption)
    caption = re.sub(r"\W+", " ", caption)
    return caption.lower().split()


ols = list(map(to_words, ols))
mls = list(map(to_words, mls))


with open(coco_train_conf["captions_path"]) as cf, open(
    coco_train_conf["semantic_captions_path"]
) as scf:
    coco_anns = sorted(json.load(cf)["annotations"], key=itemgetter("id"))
    semantic_anns = sorted(json.load(scf)["annotations"], key=itemgetter("id"))

assert len(coco_anns) == len(semantic_anns)


def join_annos(coco, semantic):
    tmp = [
        (to_words(c["caption"]), s["caption"].split())
        for c, s in zip(coco, semantic)
        if c["id"] == s["id"]
    ]
    assert len(tmp) == len(coco)
    return tmp


coco_merged = join_annos(coco_anns, semantic_anns)
shakesp_merged = list(zip(ols, mls, mls_tagged))


try:
    Path(language_data_path).unlink()
except FileNotFoundError:
    pass

with open(language_data_path, "wb") as f:
    pickle.dump({"coco_merged": coco_merged, "shakesp_merged": shakesp_merged,}, f)


with open(language_data_path, "rb") as f:
    data = pickle.load(f)

coco_merged = data["coco_merged"]
shakesp_merged = data["shakesp_merged"]
