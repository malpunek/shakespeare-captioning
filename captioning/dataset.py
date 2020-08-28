import os
import pickle
import sys
from collections import Counter
from functools import cached_property

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import WordIdxMap


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
        self.coco_ids = self.features_file["ids"]

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


class LanguageDataset(Dataset):
    def __init__(
        self,
        language_data_path,
        filter_coco=None,
        filter_shakespear=None,
        to_tensor=False,
    ):
        super().__init__()
        with open(language_data_path, "rb") as f:
            data = pickle.load(f)

        self.coco_merged = data["coco_merged"]  # [(caption, terms)..]
        self.shakesp_merged = data["shakesp_merged"]  # [(shakesperan, modern, terms)..]

        if filter_coco is not None:
            self.coco_merged = filter_coco(self.coco_merged)
        if filter_shakespear is not None:
            self.shakesp_merged = filter_shakespear(self.shakesp_merged)
        self.to_tensor = to_tensor

    def __getitem__(self, idx):
        """
        Returns:
            caption: The target caption
            terms: The semantic terms
        """
        if isinstance(idx, slice):
            return [
                self(i)
                for i in range(idx.start or 0, idx.stop or sys.maxsize, idx.step or 1)
            ]

        def helper(idx):
            if idx < len(self.coco_merged):
                return self.coco_merged[idx]

            idx -= len(self.coco_merged)

            if idx < len(self.shakesp_merged):
                return (
                    self.shakesp_merged[idx][0],
                    self.shakesp_merged[idx][-1] + ["<style>"],
                )

            idx -= len(self.shakesp_merged)
            return self.shakesp_merged[idx][1], self.shakesp_merged[idx][-1]

        caption, terms = helper(idx)
        if self.to_tensor:
            cm, tm = self.get_mappings
            caption = cm.prepare_for_training(caption, max_caption_len=60)
            terms = tm.prepare_for_training(terms, max_caption_len=30, terms=True)
        return caption, terms

    def __len__(self):
        return len(self.coco_merged) + 2 * len(self.shakesp_merged)

    @cached_property
    def get_mappings(self):
        terms_vocab = Counter()
        caps_vocab = Counter()
        store = self.to_tensor
        self.to_tensor = False
        for caption, terms in self:
            caps_vocab.update(caption)
            terms_vocab.update(terms)

        self.to_tensor = store

        return WordIdxMap(caps_vocab), WordIdxMap(terms_vocab)


class BalancedLanguageDataset(LanguageDataset):
    def __getitem__(self, idx):
        if idx < len(self.coco_merged):
            return super().__getitem__(idx)
        elif idx < 2 * len(self.coco_merged):
            idx -= len(self.coco_merged)
            idx %= 2 * len(self.shakesp_merged)
            return super().__getitem__(idx + len(self.coco_merged))
        raise IndexError

    def __len__(self):
        return 2 * len(self.coco_merged)
