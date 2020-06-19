from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision.datasets.coco import CocoCaptions
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from ..config import (captions_path, device, feature_ids_path, features_path,
                      mscoco_imgs_path)
from ..model import FeatureExtractor
from ..utils import ask_overwrite


def main():

    if not ask_overwrite(features_path):
        return

    hdf5_fname = Path(features_path)
    hdf5_fname.unlink(missing_ok=True)

    model = FeatureExtractor()
    model.eval()
    model.to(device)

    t = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = CocoCaptions(
        Path(mscoco_imgs_path).expanduser(),
        Path(captions_path).expanduser(),
        transform=t,
    )

    with h5py.File(features_path, "a") as f:
        feature_shape = (len(dataset), model.out_features)
        features = f.create_dataset("features", feature_shape, dtype="f")

        with torch.no_grad():
            for i, (img, _) in enumerate(tqdm(dataset, desc="Computing Features")):
                img = img.unsqueeze(0).to(device)
                feats = model(img).cpu().numpy()
                features[i] = feats[0]

    img_ids = np.array(dataset.ids)
    with open(feature_ids_path, "wb") as f:
        np.save(f, img_ids)


if __name__ == "__main__":
    main()
