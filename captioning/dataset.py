import h5py
from torch.utils.data import Dataset


class CaptionHdf5Dataset(Dataset):
    def __init__(self, features_file, data_file):
        super().__init__(self)
        self.features_file = h5py.File(features_file, "r", driver="core")
        self.data_file = h5py.File(data_file, "r")

        self.features = self.features_file["features"]
        self.encoded_caps = self.data_file["encoded_caps"]
        self.feat_ids = self.data_file["feat_ids"]

    def __getitem__(self, idx):
        feat_id = self.feat_ids[idx]
        return self.features[feat_id], self.encoded_caps[idx]

    def close(self):
        self.features_file.close()
        self.data_file.close()


class FullHdf5Dataset(CaptionHdf5Dataset):
    def __init__(self):
        super().__init__()
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
