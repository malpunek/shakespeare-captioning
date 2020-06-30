from pathlib import Path
from operator import itemgetter
import h5py
from torch.utils.data.dataset import Dataset


class CocoFeatureCaptionDataset(Dataset):
    def __init__(self, hdf5_path, *caption_paths):
        assert caption_paths
        self.hdf5_file = h5py.File(Path(hdf5_path).expanduser(), "r")
        self.features = self.hdf5_file["features"]

        from pycocotools.coco import COCO

        self.coco_providers = [COCO(cp) for cp in caption_paths]
        self.main_coco = self.coco_providers[0]
        self.ids = list(sorted(self.main_coco.imgs.keys()))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.main_coco.getAnnIds(imgIds=img_id)

        all_targets = (
            list(map(itemgetter("caption"), coco.loadAnns(ann_ids)))
            for coco in self.coco_providers
        )

        return self.features[idx], *all_targets

    def close(self):
        self.hdf5_file.close()
