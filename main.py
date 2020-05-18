import json
from itertools import islice

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
        model = FeatureExtractor()
        model.eval()
        model.to(self.device)

        t = transforms.Compose(
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        dataset = CocoCaptions(root_path, annotations_path, transform=t)

        captions = []
        with h5py.File(f"{save_base_path}.hdf5", "a") as f:
            feature_shape = (len(dataset), model.out_features)
            features = f.create_dataset("features", feature_shape, dtype="f")

            for i, (img, caption) in enumerate(tqdm(islice(dataset, 10))):
                img.unsqueeze(0)
                img = img.to(self.device)
                feats = model(img).cpu().numpy()
                features[i] = feats[0]
                captions.append(caption)

        with open(f"{save_base_path}.json", "w") as f:
            json.dump(captions, f)



if __name__ == "__main__":
    fire.Fire(Commands)