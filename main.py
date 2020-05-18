import json
from itertools import islice
from pathlib import Path

import fire
import h5py
import torch
from torchvision import transforms
from torchvision.datasets.coco import CocoCaptions
from tqdm.auto import tqdm

from model import FeatureExtractor


class Commands:
    """
    Args:
        cpu: Use cpu even if GPU is available
        gpu_num: Use the GPU number GPU_NUM. Use `main.py lgpu` to \
list available gpus
    """

    def __init__(self, cpu: bool = False, gpu_num: int = 0):

        device = "cpu" if cpu or not torch.cuda.is_available() else f"cuda:{gpu_num}"
        if device == "cpu":
            print("Using cpu!")
        else:
            print(f"Using {torch.cuda.get_device_name(gpu_num)}")
        self.device = torch.device(device)

    def lgpu(self):
        """Lists the GPUs found by PyTorch"""
        if not torch.cuda.is_available():
            print("Torch didn't find any available GPUs.")

        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            high, low = torch.cuda.get_device_capability(i)
            message = (
                f"GPU {i}: Device: {torch.cuda.get_device_name(i)}, "
                f"Compute Capability: {high}.{low}"
            )
            print(message)

    def extract_features(
        self,
        root_path: str = "data",
        annotations_path: str = "data/annotations/captions_train2014.json",
        save_base_path: str = "extracted_data",
    ):
        hdf5_fname = Path(f"{save_base_path}.hdf5")
        hdf5_fname.unlink(missing_ok=True)

        json_fname = Path(f"{save_base_path}.json")
        json_fname.unlink(missing_ok=True)

        model = FeatureExtractor()
        model.eval()
        model.to(self.device)

        t = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset = CocoCaptions(root_path, annotations_path, transform=t)

        with h5py.File(hdf5_fname, "a") as f:
            feature_shape = (len(dataset), model.out_features)
            features = f.create_dataset("features", feature_shape, dtype="f")

            with torch.no_grad():
                for i, (img, _) in enumerate(tqdm(dataset, desc="Computing Features")):
                    img = img.unsqueeze(0).to(self.device)
                    feats = model(img).cpu().numpy()
                    features[i] = feats[0]

        with open(json_fname, "w") as f:
            captions = [caps for _, caps in tqdm(dataset, desc="Extracting captions")]
            json.dump(captions, f)


if __name__ == "__main__":
    fire.Fire(Commands)
