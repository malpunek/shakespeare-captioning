from pathlib import Path

import h5py
from torch.utils.data.dataset import Dataset


class CocoFeatureCaptionDataset(Dataset):
    def __init__(self, hdf5_path, caption_path, target_split=False):
        self.hdf5_file = h5py.File(Path(hdf5_path).expanduser(), "r")
        self.features = self.hdf5_file["features"]
        self.target_split = target_split

        from pycocotools.coco import COCO

        self.coco = COCO(caption_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        targets = [ann["caption"] for ann in anns]
        if self.target_split:
            targets = [t.split() for t in targets]

        return self.features[idx], targets

    def close(self):
        self.hdf5_file.close()
