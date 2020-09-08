import json
import sys
from collections import Counter
from functools import cached_property
from itertools import chain
from operator import itemgetter

import h5py
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import WordIdxMap


class SemStyleDataset(Dataset):
    def __init__(self, coco_final_path, shake_final_path, filter_fn=None):
        super().__init__()

        with open(coco_final_path) as cf, open(shake_final_path) as sf:
            coco = json.load(cf)
            shake = json.load(sf)

        self.coco = coco
        self.shake = shake

        if filter_fn is not None:
            self.coco = list(filter(filter_fn, self.coco))
            self.shake = list(filter(filter_fn, self.shake))

    @cached_property
    def get_cap_mapping(self):
        caps_vocab = Counter()

        total = len(self.coco) + len(self.shake)
        for cap in tqdm(
            chain(self.coco, self.shake), total=total, desc="Calculating cap mapping",
        ):
            caps_vocab.update(cap["caption_words"])
            caps_vocab.update(cap.get("original_words", []))

        return WordIdxMap(caps_vocab)

    @cached_property
    def get_term_mapping(self):
        terms_vocab = Counter()

        total = len(self.coco) + len(self.shake)
        for cap in tqdm(
            chain(self.coco, self.shake), total=total, desc="Calculating term mapping",
        ):
            terms_vocab.update(cap["terms"])

        return WordIdxMap(terms_vocab)

    def _encode_caps(self, iterable, mapping, keyword, max_len=60):
        return [
            mapping.prepare_for_training(it[keyword], max_caption_len=max_len)
            for it in iterable
        ]

    def _encode_terms(self, iterable, mapping, style, max_len=20):
        style = [style] if style else []
        return [
            mapping.prepare_for_training(
                it["terms"] + style, max_caption_len=max_len, terms=True
            )
            for it in iterable
        ]


class LanguageDataset(SemStyleDataset):
    def __init__(self, *args, encode=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode = encode

    def calculate_encoded(self):
        # Captions
        self.coco_caps_enc = self._encode_caps(
            self.coco, self.get_cap_mapping, "caption_words"
        )
        self.shake_modern_enc = self._encode_caps(
            self.shake, self.get_cap_mapping, "caption_words"
        )
        self.shake_orig_enc = self._encode_caps(
            self.shake, self.get_cap_mapping, "original_words"
        )

        # Terms:
        self.coco_terms_enc = self._encode_terms(
            self.coco, self.get_term_mapping, None, max_len=20
        )

        self.shake_modern_terms_enc = self._encode_terms(
            self.shake, self.get_term_mapping, "<shake_modern>", max_len=20
        )
        self.shake_orig_terms_enc = self._encode_terms(
            self.shake, self.get_term_mapping, "<shake_orig>", max_len=20
        )

    @property
    def encode(self):
        return self._encode

    @encode.setter
    def encode(self, value):
        self._encode = value
        if value and not hasattr(self, "coco_caps_enc"):
            self.calculate_encoded()

    def get_coco(self, idx):
        if self.encode:
            return self.coco_caps_enc[idx], self.coco_terms_enc[idx]
        item = self.coco[idx]
        return item["caption_words"], item["terms"]

    def get_shake_modern(self, idx):
        if self.encode:
            return self.shake_modern_enc[idx], self.shake_modern_terms_enc[idx]
        item = self.shake[idx]
        return item["caption_words"], item["terms"]

    def get_shake_orig(self, idx):
        if self.encode:
            return self.shake_orig_enc[idx], self.shake_orig_terms_enc[idx]
        item = self.shake[idx]
        return item["original_words"], item["terms"]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [
                self(i)
                for i in range(idx.start or 0, idx.stop or sys.maxsize, idx.step or 1)
            ]

        if idx < len(self.coco):
            return self.get_coco(idx)

        idx -= len(self.coco)
        if idx < len(self.shake):
            return self.get_shake_modern(idx)
        idx -= len(self.shake)
        return self.get_shake_orig(idx)

    def __len__(self):
        return len(self.coco) + 2 * len(self.shake)


class BalancedLanguageDataset(LanguageDataset):
    def __getitem__(self, idx):
        if idx < len(self.coco):
            return super().__getitem__(idx)
        idx -= len(self.coco)
        if idx < len(self.coco):
            return self.get_shake_modern(idx % len(self.shake))
        idx -= len(self.coco)
        if idx < len(self.coco):
            return self.get_shake_orig(idx % len(self.shake))
        raise IndexError

    def __len__(self):
        return 3 * len(self.coco)


class FeatureMixin:
    def open_feats(self, path=None):
        if path:
            self.features_path = path

        self.features_file = h5py.File(self.features_path, "r", driver="core")
        self.features = self.features_file["features"]
        self.feat_ids = self.features_file["ids"]

    def close(self):
        self.features_file.close()


class QuickCocoDataset(SemStyleDataset, FeatureMixin):
    def __init__(
        self, features_path, *args, encode=True, val_final_file=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encode = encode
        if val_final_file is not None:
            with open(val_final_file) as vf:
                self.source = json.load(vf)
        else:
            self.source = self.coco

        self.open_feats(features_path)
        self.id2idx = {img_id: idx for idx, img_id in enumerate(self.feat_ids)}

        self.coco_terms_enc = self._encode_caps(
            self.source, self.get_term_mapping, "terms", max_len=20
        )  # Treat as normal caption: we want <start> and <end>

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        feat_id = self.source[idx]["img_id"]
        feat_idx = self.id2idx[feat_id]
        terms = (
            torch.LongTensor(self.coco_terms_enc[idx])
            if self.encode
            else ["<start>"] + self.source[idx]["terms"] + ["<end>"]
        )
        return self.features[feat_idx], terms


class ValidationDataset(Dataset, FeatureMixin):
    def __init__(self, features_path, coco_val_final):
        self.open_feats(features_path)

        feat_targets = {feat_id: [] for feat_id in self.feat_ids}

        with open(coco_val_final) as vf:
            self.coco = json.load(vf)

        for ann in self.coco:
            key = ann["img_id"]
            if key not in feat_targets:
                feat_targets[key] = list()
            feat_targets[key].append(ann["terms"])

        self.feat_targets = feat_targets

    def __len__(self):
        return len(self.feat_ids)

    def __getitem__(self, idx):
        feat_id = self.feat_ids[idx]
        return self.features[idx], self.feat_targets[feat_id]


class AllTermsDataset(SemStyleDataset, FeatureMixin):
    def __init__(self, features_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.open_feats(features_path)

        coco_img_ids = list(set(map(itemgetter("img_id"), self.coco)))
        id_map = {img_id: [] for img_id in coco_img_ids}

        for i, c in enumerate(tqdm(self.coco, desc="Merging terms")):
            id_map[c["img_id"]].append(c["terms"])

        tmap = self.get_term_mapping

        self.merged_terms = {
            img_id: torch.LongTensor(
                tmap.prepare_for_training(
                    list(set(chain.from_iterable(terms))), max_caption_len=20
                )
            )
            for img_id, terms in tqdm(
                id_map.items(), desc="Chaining", total=len(id_map)
            )
        }

        assert all(
            idx in self.merged_terms for idx in tqdm(self.feat_ids, desc="Asserting")
        )

    def __getitem__(self, idx):
        feat_id = self.feat_ids[idx]
        return self.features[idx], self.merged_terms[feat_id]

    def __len__(self):
        return len(self.feat_ids)
