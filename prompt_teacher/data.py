"""Image-only cache generation and explicitly authorized tissue supervision."""
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from freqssl.anatomy import family_labels
from freqssl.data import SpiderSliceDataset, _read_transposed
from freqssl.prompt_supervision import file_sha256


class TissueDataset(SpiderSliceDataset):
    def __init__(self, root, split, names, mean, std, *, augment=False, seed=42, image_only=False):
        super().__init__(root, split, 224, names=names,
            labeled_names=[] if image_only else names, augment=augment, seed=seed)
        self.image_only = image_only
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]

    def __getitem__(self, index):
        name = self.names[index]
        image = _read_transposed(self.root / "image" / name, torch.float32)[None]
        if image.min() < 0 or image.max() > 255:
            raise ValueError("Expected the existing quantized gray8 SPIDER inputs")
        native_shape = tuple(image.shape[-2:])
        label = None
        if not self.image_only:
            if name not in self.labeled_names:
                raise ValueError("Teacher attempted to access an unauthorized annotation")
            label = _read_transposed(self.root / "label" / name, torch.long)
            if tuple(label.shape) != native_shape:
                raise ValueError("Teacher image/label shapes differ")
            label = family_labels(label)
            label = F.interpolate(label[None, None].float(), (224, 224), mode="nearest")[0, 0].long()
        image = F.interpolate(image[None], (224, 224), mode="bilinear", align_corners=False)[0]
        if self.augment:
            image, label = self._geometry(image, label, index)
        image = (image.expand(3, -1, -1) / 255. - self.mean) / self.std
        result = {"image": image, "file_name": name, "native_shape": native_shape}
        if label is not None:
            result["label"] = label
        return result


def image_seed(name, seed=42):
    return (int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little") + seed) % (2**31)


def supervision_fingerprint(root, split):
    """Hashes only authorized training labels; never enumerates unlabeled masks."""
    base = Path(root) / "train"
    return {"labeled_annotation_sha256": {n: file_sha256(base / "label" / n) for n in split["labeled_files"]},
            "labeled_image_sha256": {n: file_sha256(base / "image" / n) for n in split["labeled_files"]}}
