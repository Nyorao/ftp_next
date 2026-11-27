"""SPIDER frequency inputs using the original MONAI augmentation operations.

Unlabeled masks are never opened. Fixed text comes from model buffers, never
from a sample's ground truth. Image axes follow the original MONAI PILReader.
"""

import itertools
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose, NormalizeIntensityd, RandAffined, RandGaussianNoised,
    RandGaussianSmoothd, RandGridDistortiond, Resized, ToTensord,
)
from PIL import Image
from torch.utils.data import Dataset, Sampler


def read_image(path):
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale image: {path}, got {array.shape}")
    # MONAI's PILReader uses reverse_indexing=True by default (W,H).
    return array.T.copy()[None].astype(np.float32)


class SpiderDataset(Dataset):
    def __init__(self, root, split, image_size=224, labeled_count=None, max_samples=None):
        self.root = Path(root) / split
        self.split = split
        self.names = sorted(path.name for path in (self.root / "image").glob("*.png"))
        if not self.names:
            raise ValueError(f"No PNG images in {self.root / 'image'}")
        if max_samples is not None:
            if max_samples > len(self.names):
                raise ValueError(f"max_samples={max_samples} exceeds {len(self.names)} images")
            self.names = self.names[:max_samples]
        self.labeled_count = len(self.names) if labeled_count is None else labeled_count
        if not 0 <= self.labeled_count <= len(self.names):
            raise ValueError("labeled_count is outside the dataset")
        keys = ["image", "lf", "hf", "label"]
        transforms = []
        if split == "train":
            transforms += [
                RandGaussianNoised(["image"], prob=0.4),
                RandGaussianSmoothd(["image"], prob=0.4),
                RandAffined(keys, prob=0.4, translate_range=(0.1, 0.1),
                            rotate_range=(0.1, 0.1), shear_range=(0.1, 0.1),
                            scale_range=(0.1, 0.1),
                            mode=("bilinear", "bilinear", "bilinear", "nearest")),
                RandGridDistortiond(keys, num_cells=20, distort_limit=(-0.2, 0.2),
                                    padding_mode="zeros", prob=0.7,
                                    mode=["bilinear", "bilinear", "bilinear", "nearest"]),
            ]
        for key in ("image", "lf", "hf"):
            transforms.append(Resized([key], spatial_size=(image_size, image_size), mode="bicubic"))
        if split == "train":
            transforms.append(Resized(["label"], spatial_size=(image_size, image_size), mode="nearest"))
        transforms += [NormalizeIntensityd(["image", "lf", "hf"], channel_wise=True), ToTensord(keys)]
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        labeled = self.split != "train" or index < self.labeled_count
        raw = read_image(self.root / "image" / name)
        data = {
            "image": raw,
            "lf": read_image(self.root / "low_frequency" / name),
            "hf": read_image(self.root / "high_frequency" / name),
            "label": read_image(self.root / "label" / name) if labeled else np.zeros_like(raw),
        }
        data = self.transform(data)
        data = {key: value.as_tensor() if hasattr(value, "as_tensor") else value
                for key, value in data.items()}
        data["label"] = data["label"].long().squeeze(0)
        if not labeled:
            data["label"].fill_(-100)
        data.update(is_labeled=labeled, file_name=name)
        return data


class TwoStreamBatchSampler(Sampler):
    """Original labeled-first sampling, with deterministic per-epoch shuffling."""

    def __init__(self, labeled, unlabeled, batch_size, labeled_bs, seed=42):
        self.labeled = list(labeled)
        self.unlabeled = list(unlabeled)
        self.labeled_bs = labeled_bs
        self.unlabeled_bs = batch_size - labeled_bs
        self.seed = seed
        self.epoch = 0
        if not 0 < labeled_bs <= len(self.labeled):
            raise ValueError("Not enough labeled images for a batch")
        if not 0 < self.unlabeled_bs <= len(self.unlabeled):
            raise ValueError("Not enough unlabeled images for a batch")

    def __len__(self):
        return len(self.labeled) // self.labeled_bs

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        primary = rng.permutation(self.labeled)

        def repeat_unlabeled():
            while True:
                yield from rng.permutation(self.unlabeled)

        secondary = repeat_unlabeled()
        for start in range(0, len(primary) - self.labeled_bs + 1, self.labeled_bs):
            yield list(primary[start:start + self.labeled_bs]) + list(
                itertools.islice(secondary, self.unlabeled_bs)
            )


def seed_worker(worker_id):
    # MONAI transforms carry their own RNG; seed that as well as numpy/random.
    import random
    from torch.utils.data import get_worker_info

    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    worker = get_worker_info()
    if worker is not None:
        worker.dataset.transform.set_random_state(seed=seed)
