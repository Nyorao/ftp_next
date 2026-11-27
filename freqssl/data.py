"""Raw SPIDER slices and a reproducible, labeled-first sampling protocol.

The original PILReader convention is retained: a PNG with axes (H, W) is
returned with spatial axes (W, H). Frequency views are constructed by the
model from these raw inputs, so this module never opens LF/HF files.
"""

from itertools import islice
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler


_NAME = re.compile(r"^(SPIDER_\d+)_(\d+)_(\d+)\.png$")


def _case_id(name):
    match = _NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid SPIDER slice name: {name}")
    index, count = int(match.group(2)), int(match.group(3))
    if count <= 0 or not 0 <= index < count:
        raise ValueError(f"Invalid slice index/count: {name}")
    return match.group(1)


def _slice_names(root, split):
    return sorted(path.name for path in (Path(root) / split / "image").glob("*.png"))


def _image_shape(image_size):
    shape = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("image_size must be a positive integer or two positive integers")
    return shape


def _read_transposed(path, dtype):
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale PNG: {path}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite pixels in {path}")
    # SPIDER annotations can be 16-bit PNG. PyTorch 2.0 cannot construct a
    # tensor directly from numpy.uint16, so convert before from_numpy.
    array = array.T.astype(np.float32 if dtype == torch.float32 else np.int64, copy=True)
    return torch.from_numpy(array).to(dtype=dtype)


class SpiderSliceDataset(Dataset):
    """One raw input and one spatially aligned target per sample.

    Unlabeled training samples return an ignore target filled with -100 and
    never open their annotation. Validation/test targets retain native PNG
    resolution in transposed axes, for exact 3D reconstruction after inference.
    Call ``set_epoch`` before constructing each DataLoader epoch iterator;
    use the default ``persistent_workers=False`` when workers are enabled.
    """

    def __init__(self, root, split, image_size, names=None, labeled_names=None,
                 augment=False, seed=42, *, horizontal_flip_probability=0.0, slice_context="none",
                 case_reference="none", context_position="none", context_geometry_manifest=""):
        if split not in ("train", "valid", "test"):
            raise ValueError("split must be train, valid or test")
        if augment and split != "train":
            raise ValueError("Random geometry is only supported for training")
        if seed < 0 or not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("Invalid augmentation seed or flip probability")
        self.root = Path(root) / split
        self.split = split
        self.image_size = _image_shape(image_size)
        self.names = list(names) if names is not None else _slice_names(root, split)
        if not self.names:
            raise ValueError(f"No PNG images in {self.root / 'image'}")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Dataset filenames must be unique")
        for name in self.names:
            _case_id(name)
        self.labeled_names = set(self.names if labeled_names is None else labeled_names)
        if not self.labeled_names.issubset(self.names):
            raise ValueError("labeled_names contains filenames outside this dataset")
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        if slice_context not in ("none", "mean", "adaptive"):
            raise ValueError("Unknown slice_context")
        self.slice_context = slice_context
        self.case_reference, self.context_position = case_reference, context_position
        if (case_reference,context_position) not in (('none','none'),('middle','none'),('none','relative')):
            raise ValueError('Choose one case-context arm')
        if (case_reference != 'none' or context_position != 'none') and slice_context != 'adaptive':
            raise ValueError('Case context requires adaptive input')
        self.references, self.positions, self.geometry_receipt = {}, {}, None
        self.prompt_cache = None
        self.neighbors = {}
        if slice_context != "none":
            # Only the selected image pool is eligible, including at budget edges.
            groups = {}
            for name in self.names:
                match = _NAME.fullmatch(name)
                case, index, count = match[1], int(match[2]), int(match[3])
                group = groups.setdefault(case, {})
                if index in group:
                    raise ValueError(f"Duplicate numeric slice index in {case}")
                group[index] = (name, count)
            for case, group in groups.items():
                counts = {count for _, count in group.values()}
                if len(counts) != 1 or set(group) != set(range(next(iter(counts)))):
                    raise ValueError(f"Incomplete context slice sequence: {case}")
                count = len(group)
                for index, (name, _) in group.items():
                    self.neighbors[name] = tuple(group[min(count - 1, max(0, index + d))][0]
                                                 for d in (-1, 0, 1))
                    self.references[name] = group[(count-1)//2][0]
        if context_position == 'relative':
            from .case_context import verify_dataset_geometry
            self.positions,self.geometry_receipt = verify_dataset_geometry(self,context_geometry_manifest)

    def __len__(self):
        return len(self.names)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def _geometry(self, image, label, index, auxiliary=None, auxiliary_mode="nearest"):
        # Local RNG makes geometry independent of model count, global RNG use,
        # worker scheduling and how often validation is run.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        if rng.random() < 0.7:
            angle = rng.uniform(-np.pi / 18, np.pi / 18)
            scale = rng.uniform(0.9, 1.1)
            cosine, sine = np.cos(angle) * scale, np.sin(angle) * scale
            theta = image.new_tensor([[cosine, -sine, rng.uniform(-0.08, 0.08)],
                                      [sine, cosine, rng.uniform(-0.08, 0.08)]])[None]
            grid = F.affine_grid(theta, (1, 1, *self.image_size), align_corners=False)
            image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="zeros",
                                  align_corners=False)[0]
            if label is not None:
                label = F.grid_sample(label[None, None].float(), grid, mode="nearest",
                                      padding_mode="zeros", align_corners=False)[0, 0].long()
            if auxiliary is not None:
                auxiliary = F.grid_sample(auxiliary[None], grid, mode=auxiliary_mode,
                                          padding_mode="zeros", align_corners=False)[0]
        if rng.random() < self.horizontal_flip_probability:
            # PNG left/right is axis -2 after the preserved W,H transpose.
            # Never invert PNG rows: cranial/caudal class identity must survive.
            image = image.flip(-2)
            if label is not None:
                label = label.flip(-2)
            if auxiliary is not None:
                auxiliary = auxiliary.flip(-2)
        return (image, label) if auxiliary is None else (image, label, auxiliary)

    def read_image(self, index):
        """Image-only center input for cached inference; never opens labels."""
        image = _read_transposed(self.root / "image" / self.names[index], torch.float32)[None]
        image = F.interpolate(image[None], size=self.image_size, mode="bilinear", align_corners=False)[0]
        return (image - image.mean()) / image.std(unbiased=False).clamp_min(1e-6)

    def __getitem__(self, index):
        index = int(index)
        name = self.names[index]
        labeled = self.split != "train" or name in self.labeled_names
        image = _read_transposed(self.root / "image" / name, torch.float32)[None]
        if self.slice_context != "none":
            neighbors = self.neighbors[name]
            image = torch.cat((_read_transposed(self.root / "image" / neighbors[0], torch.float32)[None],
                               image,
                               _read_transposed(self.root / "image" / neighbors[2], torch.float32)[None]))
            if self.case_reference == 'middle':
                image = torch.cat((image,_read_transposed(self.root/'image'/self.references[name],torch.float32)[None]))
        label = None
        if labeled:
            label = _read_transposed(self.root / "label" / name, torch.long)
            if tuple(label.shape) != tuple(image.shape[-2:]):
                raise ValueError(f"Image/label shape mismatch for {name}")
            if torch.any((label < 0) | (label >= 20)):
                raise ValueError(f"SPIDER labels must be class IDs 0..19: {name}")
        # Resizing happens before any common geometry. There is no independent
        # transformation of an input view and its supervision/pseudo-target.
        image = F.interpolate(image[None], size=self.image_size, mode="bilinear",
                              align_corners=False)[0]
        if self.split == "train" and label is not None:
            label = F.interpolate(label[None, None].float(), size=self.image_size,
                                  mode="nearest")[0, 0].long()
        auxiliary = None
        if self.prompt_cache is not None:
            auxiliary = (torch.zeros((self.prompt_cache.channels, *self.image_size))
                         if labeled and not getattr(self.prompt_cache, "for_labeled", False)
                         else self.prompt_cache.read(name))
        if self.augment:
            if auxiliary is None:
                image, label = self._geometry(image, label, index)
            else:
                image, label, auxiliary = self._geometry(image, label, index, auxiliary,
                                                        self.prompt_cache.geometry_mode)
        if auxiliary is not None and self.prompt_cache.version == 2:
            from .prompt_revision import unpack_template_probabilities
            auxiliary = unpack_template_probabilities(auxiliary[None])[0]
        if self.slice_context == "none":
            image = (image - image.mean()) / image.std(unbiased=False).clamp_min(1e-6)
        else:
            # Same per-slice reduction as the old one-channel path.
            image = torch.cat([(part - part.mean()) / part.std(unbiased=False).clamp_min(1e-6)
                               for part in image.split(1)])
        if label is None:
            label = torch.full(self.image_size, -100, dtype=torch.long)
        sample = {"image": image, "label": label, "is_labeled": labeled, "file_name": name}
        if self.context_position == 'relative':
            sample['position'] = torch.full((1,*self.image_size),self.positions[name],dtype=torch.float32)
        if auxiliary is not None:
            sample["prompt_targets"] = auxiliary
        return sample


class TwoStreamBatchSampler(Sampler):
    """Shuffle labeled samples once per epoch and cycle shuffled unlabeled ones.

    Labeled entries always precede unlabeled entries. Incomplete labeled batches
    are dropped; the manifest records this budget rather than silently rounding
    it up. The sampler's RNG is independent of all augmentation/model RNGs.
    """

    def __init__(self, labeled, unlabeled, batch_size, labeled_bs, seed=42):
        self.labeled = list(labeled)
        self.unlabeled = list(unlabeled)
        self.labeled_bs = int(labeled_bs)
        self.unlabeled_bs = int(batch_size) - self.labeled_bs
        self.seed = int(seed)
        self.epoch = 0
        if not 0 < self.labeled_bs <= len(self.labeled):
            raise ValueError("Not enough labeled images for a batch")
        if not 0 < self.unlabeled_bs <= len(self.unlabeled):
            raise ValueError("Not enough unlabeled images for a batch")
        if (len(set(self.labeled)) != len(self.labeled)
                or len(set(self.unlabeled)) != len(self.unlabeled)
                or set(self.labeled) & set(self.unlabeled)):
            raise ValueError("Sampler pools must be disjoint and contain unique indices")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.labeled) // self.labeled_bs

    def __iter__(self):
        rng = np.random.RandomState((self.seed + self.epoch) % (2 ** 32))
        primary = rng.permutation(self.labeled)

        def repeat_unlabeled():
            while True:
                yield from rng.permutation(self.unlabeled)

        secondary = repeat_unlabeled()
        for start in range(0, len(primary) - self.labeled_bs + 1, self.labeled_bs):
            yield [int(value) for value in primary[start:start + self.labeled_bs]] + [
                int(value) for value in islice(secondary, self.unlabeled_bs)]


def build_training_data(cfg):
    """Build the unchanged sorted-slice supervision budget and its audit trail.

    Split isolation is checked using filenames only, including test filenames.
    A case shared across labeled/unlabeled training pools is reported explicitly;
    this API does not silently reinterpret the historical budget as case based.
    """
    inventories = {split: _slice_names(cfg.root_path, split) for split in ("train", "valid", "test")}
    all_names = inventories["train"]
    maximum = len(all_names) if cfg.max_samples is None else cfg.max_samples
    if not 0 < maximum <= len(all_names):
        raise ValueError(f"max_samples={maximum} is outside the {len(all_names)} training images")
    if not 0 < cfg.labelnum < maximum:
        raise ValueError("Require 0 < labelnum < selected training samples")
    names = all_names[:maximum]
    labeled, unlabeled = names[:cfg.labelnum], names[cfg.labelnum:]
    partition = None
    if cfg.training_split:
        from .prompt_revision import read_partition
        partition = read_partition(cfg.training_split, labeled, unlabeled)
        labeled, unlabeled = partition["labeled_files"], partition["unlabeled_files"]
        names = labeled + unlabeled
    split_cases = {split: sorted({_case_id(name) for name in values})
                   for split, values in inventories.items()}
    overlaps = {f"{first}/{second}": sorted(set(split_cases[first]) & set(split_cases[second]))
                for first, second in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    if any(overlaps.values()):
        raise ValueError(f"Overlapping SPIDER case identifiers across data splits: {overlaps}")
    labeled_cases = sorted({_case_id(name) for name in labeled})
    unlabeled_cases = sorted({_case_id(name) for name in unlabeled})
    shared = sorted(set(labeled_cases) & set(unlabeled_cases))
    dataset = SpiderSliceDataset(cfg.root_path, "train", cfg.image_size, names=names,
                                 labeled_names=labeled, augment=True, seed=cfg.seed,
                                 slice_context=cfg.slice_context, case_reference=cfg.case_reference,
                                 context_position=cfg.context_position, context_geometry_manifest=cfg.context_geometry_manifest)
    sampler = TwoStreamBatchSampler(range(len(labeled)), range(len(labeled), len(names)),
                                    cfg.batch_size, cfg.labeled_bs, cfg.seed)
    manifest = {
        "root_path": str(Path(cfg.root_path).resolve()),
        "label_budget_policy": "first labelnum filenames in lexicographically sorted training PNGs",
        "training_partition_unit": "slice", "labeled_files": labeled, "unlabeled_files": unlabeled,
        "labeled_cases": labeled_cases, "unlabeled_cases": unlabeled_cases,
        "labeled_unlabeled_shared_cases": shared,
        "case_disjoint_labeled_unlabeled": not bool(shared),
        "labeled_boundary_note": ("Some cases straddle the slice-label budget; this is not a case-disjoint "
                                "semi-supervised split." if shared else "No training case crosses the label budget."),
        "available_training_slices": len(all_names), "selected_training_slices": len(names),
        "labeled_slices": len(labeled), "unlabeled_slices": len(unlabeled),
        "split_cases": split_cases, "split_case_overlaps": overlaps,
        "split_case_audit_complete": all(bool(values) for values in inventories.values()),
        "split_slice_counts": {split: len(values) for split, values in inventories.items()},
        "validation_cases": split_cases["valid"], "test_case_count": len(split_cases["test"]),
        "test_filenames_inspected_only": True, "test_labels_or_images_used": False,
        "offline_frequency_files_used": False, "unlabeled_annotations_opened": False,
        "steps_per_epoch": len(sampler),
        "labeled_samples_dropped_per_epoch": len(labeled) % cfg.labeled_bs,
        "augmentation": {"source": "raw PNG", "resize_before_geometry": True,
                         "shared_image_target_affine": True, "rotation_degrees": [-10, 10],
                         "scale": [0.9, 1.1], "translation_normalized": [-0.08, 0.08],
                         "affine_probability": 0.7, "horizontal_flip_probability": 0.0,
                         "vertical_flip_probability": 0.0,
                         "seed_policy": "SeedSequence(seed, epoch, sample_index)",
                         "persistent_workers": False},
    }
    if cfg.prompt_source != "none":
        from .prompt_supervision import PromptCache
        dataset.prompt_cache = PromptCache(cfg.prompt_cache, cfg.prompt_source, cfg.image_size, manifest)
        manifest["prompt_guidance"] = {"source": cfg.prompt_source,
            "cache_manifest_sha256": dataset.prompt_cache.manifest_sha256,
            "cache": str(dataset.prompt_cache.root)}
        if cfg.prompt_route == "competence":
            from .prompt_revision import CompetenceGate
            gate = CompetenceGate(cfg.prompt_gate, cfg.prompt_source)
            if dataset.prompt_cache.version != 2 or gate.variant != dataset.prompt_cache.manifest.get("teacher_variant"):
                raise ValueError("Gate and cache teacher variants differ")
            manifest["prompt_guidance"]["gate_sha256"] = gate.sha256
    if partition is not None:
        from .prompt_supervision import file_sha256
        manifest["calibration_partition"] = {"sha256": file_sha256(cfg.training_split),
            "heldout_cases": partition["calibration_cases"], "heldout_slices": len(partition["calibration_files"])}
    if cfg.case_reference != 'none' or cfg.context_position != 'none':
        manifest['case_context'] = dict(reference=cfg.case_reference,position=cfg.context_position,
            reference_uses_labels=False,geometry=dataset.geometry_receipt)
    if cfg.reference_condition != 'none':
        from .reference_data import ReferencePredictionCache, ReferenceTrainingDataset, file_hash
        from .case_context import verify_dataset_geometry
        if not cfg.reference_cache:
            raise ValueError('Generate the frozen reference cache before training')
        teacher_hash = file_hash(cfg.reference_teacher)
        if cfg.reference_teacher_sha256 and teacher_hash != cfg.reference_teacher_sha256:
            raise ValueError('Frozen reference teacher changed')
        _, receipt = verify_dataset_geometry(dataset, cfg.context_geometry_manifest)
        cache = ReferencePredictionCache(Path(cfg.reference_cache) / 'train', dataset, teacher_hash)
        manifest['reference_conditioning'] = dict(condition=cfg.reference_condition,
            teacher_sha256=teacher_hash, cache_manifest_sha256=cache.manifest_hash,
            sampling=cfg.reference_sampling, gt_curriculum=cfg.reference_gt,
            reference_labels_restricted_to_labeled_files=True, geometry=receipt)
        dataset = ReferenceTrainingDataset(dataset, cfg, cache)
    return dataset, sampler, manifest
