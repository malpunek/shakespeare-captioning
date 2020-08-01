from operator import itemgetter
from pathlib import Path

import h5py
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..config import coco_train_conf
from ..utils import ask_overwrite


class CocoFeatureCaptionDataset(Dataset):
    def __init__(self, hdf5_path, *caption_paths):
        assert caption_paths
        self.hdf5_file = h5py.File(Path(hdf5_path).expanduser(), "r")
        self.features = self.hdf5_file["features"]
        self.hdf5_ids = self.hdf5_file["ids"]

        from pycocotools.coco import COCO

        self.coco_providers = [COCO(cp) for cp in caption_paths]
        self.main_coco = self.coco_providers[0]
        self.ids = list(sorted(self.main_coco.imgs.keys()))

    def __len__(self):
        return len(self.features)

    def count_caps(self):
        result = 0
        for _, *caps in tqdm(self, desc="Computing number of captions"):
            result += min(map(lambda c: len(c), caps))
        return result

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        assert img_id == self.hdf5_ids[idx], "Something is wrong"

        ann_ids = self.main_coco.getAnnIds(imgIds=img_id)

        all_targets = (
            list(map(itemgetter("caption"), coco.loadAnns(ann_ids)))
            for coco in self.coco_providers
        )

        file_name = self.main_coco.loadImgs(img_id)[0]["file_name"]

        return (self.features[idx], file_name, *all_targets)

    def close(self):
        self.hdf5_file.close()


def main():
    def make_transformation(f: h5py.File, dataset):

        data_len = dataset.count_caps()

        feat_ids = f.create_dataset("feat_ids", (data_len,), dtype="i")
        filenames = f.create_dataset(
            "filenames", (data_len,), dtype=h5py.string_dtype()
        )
        coco_caps = f.create_dataset(
            "coco_caps", (data_len,), dtype=h5py.string_dtype()
        )
        semantic_caps = f.create_dataset(
            "semantic_caps", (data_len,), dtype=h5py.string_dtype()
        )
        encoded_caps = f.create_dataset(
            "encoded_caps", (data_len,), dtype=h5py.string_dtype()
        )

        idx = 0
        for feat_idx, (_, filename, *caps) in enumerate(
            tqdm(dataset, desc="Constructing HDF5 file")
        ):
            for cap in zip(*caps):
                feat_ids[idx] = feat_idx
                coco_caps[idx] = cap[0]
                semantic_caps[idx] = cap[1]
                encoded_caps[idx] = " ".join(map(lambda x: str(x), cap[2]))
                filenames[idx] = filename
                idx += 1

    ctc = coco_train_conf
    if not ask_overwrite(ctc["transformed_data_path"]):
        return

    hdf5_fname = Path(ctc["transformed_data_path"])
    hdf5_fname.unlink(missing_ok=True)

    dataset = CocoFeatureCaptionDataset(
        ctc["features_path"],
        ctc["captions_path"],
        ctc["semantic_captions_path"],
        ctc["encoded_captions_path"],
    )
    with h5py.File(ctc["transformed_data_path"], "w") as final:
        make_transformation(final, dataset)


# %%
if __name__ == "__main__":
    main()
