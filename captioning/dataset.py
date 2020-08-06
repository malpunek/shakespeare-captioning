import os

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CaptionHdf5Dataset(Dataset):
    def __init__(self, features_file, data_file):
        super().__init__()
        self.features_file = h5py.File(features_file, "r", driver="core")
        self.data_file = h5py.File(data_file, "r", driver="core")

        self.features = self.features_file["features"]
        self.feat_ids = self.data_file["feat_ids"]
        self.encoded_caps = self.data_file["encoded_caps"]
        self.encoded_caps._local.astype = np.dtype("long")

    def __len__(self):
        return len(self.feat_ids)

    def __getitem__(self, idx):
        feat_id = self.feat_ids[idx]
        return self.features[feat_id], self.encoded_caps[idx]

    def close(self):
        self.features_file.close()
        self.data_file.close()


class FullHdf5Dataset(CaptionHdf5Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coco_ids = self.features_file["ids"]

        self.filenames = self.data_file["filenames"]
        self.coco_caps = self.data_file["coco_caps"]
        self.semantic_caps = self.data_file["semantic_caps"]

    def __getitem__(self, idx):
        feat_id = self.feat_ids[idx]
        return {
            "features": self.features[feat_id],
            "coco_id": self.coco_ids[feat_id],
            "filename": self.filenames[idx],
            "coco_cap": self.coco_caps[idx],
            "semantic_cap": self.semantic_caps[idx],
            "encoded_cap": self.encoded_caps[idx],
        }


class ValidationDataset(Dataset):
    """Slightly altered torchvision.datasets.CocoCaptions
    Provides img features instead of the image itself
    """

    def __init__(self, coco_file, features_file):
        super().__init__()
        from pycocotools.coco import COCO

        self.coco = COCO(coco_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.features_file = h5py.File(features_file, "r", driver="core")
        self.features = self.features_file["features"]
        self.coco_ids = self.features_file["feat_ids"]

    def __getitem__(self, index):
        assert self.ids[index] == self.coco_ids[index]
        feats = self.features[index]

        ann_ids = self.coco.getAnnIds(imgIds=self.ids[index])
        anns = self.coco.loadAnns(ann_ids)
        target = [ann["caption"] for ann in anns]

        return feats, target

    def __len__(self):
        return len(self.ids)

    def close(self):
        self.features_file.close()


class QualitativeDataset(ValidationDataset):
    def __init__(self, img_folder, *args):
        super().__init__(*args)
        self.img_folder = img_folder

    def __getitem__(self, index):
        feats, caps = super().__getitem__()

        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.img_folder, path)).convert("RGB")

        return feats, caps, img
