from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import coco_captions_train, coco_captions_val, device
from ..model import FeatureExtractor
from ..utils import ask_overwrite


def populate_file(f, dataset):
    model = FeatureExtractor()
    model.eval()
    model.to(device)

    feature_shape = (len(dataset), model.out_features)
    features = f.create_dataset("features", feature_shape, dtype="f")

    ids = np.array(dataset.ids)
    h5py_ids = f.create_dataset("ids", (len(dataset),), dtype="i")
    h5py_ids[...] = ids

    with torch.no_grad():
        for i, (img, _) in enumerate(tqdm(dataset, desc="Computing Features")):
            img = img.unsqueeze(0).to(device)
            feats = model(img).cpu().numpy()
            features[i] = feats[0]


def extract(dataset, conf):
    if not ask_overwrite(conf["features"]):
        return

    hdf5_fname = Path(conf["features"])
    hdf5_fname.unlink(missing_ok=True)

    with h5py.File(conf["features"], "w") as f:
        populate_file(f, dataset)


def main():
    for dataset, conf in (coco_captions_train(), coco_captions_val()):
        extract(dataset, conf)


if __name__ == "__main__":
    main()
