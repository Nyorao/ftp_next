"""Complete native-volume validation without offline frequency assets.

Checkpoint selection uses per-case foreground overlap only. Physical spacing
corrections do not alter these metrics; no HD95/ASD in millimetres is reported.
The legacy volume utilities are reused without importing the legacy trainer.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch
from torch.nn import functional as F

from spider_volume import (
    aggregate_cases, geometry, group_complete_slices, layout_from_reference,
    reconstruct, same_geometry, volume_metrics,
)
from .data import SpiderSliceDataset
from .context_cache import iter_cached_logits


METRIC_POLICY = {
    "primary": "foreground GT-present class mean within each case, then equal-weight case mean",
    "supplementary": "foreground GT-or-prediction-present class mean within each case, then case mean",
    "background_excluded": True,
    "empty_gt_pred_class_pair": "excluded from both means",
    "false_positive_absent_gt_class": "excluded from primary, zero in union-present supplement",
    "surface_metrics_requested": False,
    "spacing_note": "Dice and IoU are voxel-overlap metrics; physical-spacing corrections do not affect them.",
}


def _audit_raw_case(png_root, volume_root, split, case, names):
    """Check PNG/native label identity and raw-image geometry, without LF/HF."""
    reference_path = Path(volume_root) / split / "label" / f"{case}.mha"
    reference = sitk.ReadImage(str(reference_path))
    target = sitk.GetArrayFromImage(reference)
    if not set(np.unique(target)).issubset(range(20)):
        raise ValueError(f"Unexpected reference labels in {reference_path}")
    axis, flip = layout_from_reference(target, len(names))
    slices = []
    for name in names:
        image_path = Path(png_root) / split / "image" / name
        label_path = Path(png_root) / split / "label" / name
        with Image.open(image_path) as image:
            # Only the shape is needed here. Pixels are read by the dataset.
            image_size = image.size
            if image.mode not in ("1", "L", "I", "I;16", "I;16B", "I;16L", "F"):
                raise ValueError(f"Expected grayscale image: {image_path}")
        with Image.open(label_path) as image:
            label = np.array(image)
        if label.ndim != 2 or label.shape != (image_size[1], image_size[0]):
            raise ValueError(f"PNG image/label shape mismatch: {name}")
        slices.append(label)
    restored = reconstruct(slices, axis, flip, target.shape)
    mismatches = int(np.count_nonzero(restored != target))
    if mismatches:
        raise ValueError(f"{case}: PNG label round-trip has {mismatches} mismatched voxels")
    raw_path = Path(volume_root) / split / "image" / f"{case}.mha"
    # Reading metadata suffices for image/label geometry consistency and avoids
    # allocating an additional full-resolution intensity volume during audit.
    raw = sitk.ImageFileReader()
    raw.SetFileName(str(raw_path))
    raw.ReadImageInformation()
    if not same_geometry(raw, reference):
        raise ValueError(f"{case}: MHA image and label geometry differ")
    if min(reference.GetSpacing()) <= 0:
        raise ValueError(f"{case}: invalid voxel spacing")
    return {"case": case, "split": split, "slice_count": len(names), "slice_names": list(names),
            "reference": str(reference_path), "array_shape_zyx": list(target.shape),
            "slice_axis": axis, "undo_vertical_flip": flip, "geometry": geometry(reference),
            "label_roundtrip_mismatched_voxels": mismatches, "image_label_geometry_equal": True,
            "gt_class_voxels": np.bincount(target.ravel(), minlength=20).tolist(),
            "offline_frequency_files_required": False, "metric_policy": dict(METRIC_POLICY)}


def prepare_validation(cfg, split="valid"):
    """Audit complete validation cases, or a test split only when explicitly selected."""
    if split not in ("valid", "test"):
        raise ValueError("Volume evaluation split must be valid or explicitly requested test")
    dataset = SpiderSliceDataset(cfg.root_path, split, cfg.image_size, augment=False, seed=cfg.seed,
                                 slice_context=cfg.slice_context,case_reference=cfg.case_reference,
                                 context_position=cfg.context_position,context_geometry_manifest=cfg.context_geometry_manifest)
    groups = group_complete_slices(dataset.names)
    volume_root = getattr(cfg, "volume_root", None) or Path(cfg.root_path).parent / "mha"
    records = [_audit_raw_case(cfg.root_path, volume_root, split, case, names)
               for case, names in groups.items()]
    if cfg.reference_condition != 'none':
        from .reference_data import ReferencePredictionCache, file_hash
        from .case_context import verify_dataset_geometry
        teacher_hash = file_hash(cfg.reference_teacher)
        if cfg.reference_teacher_sha256 and teacher_hash != cfg.reference_teacher_sha256:
            raise ValueError('Frozen reference teacher changed')
        _, dataset.reference_geometry = verify_dataset_geometry(dataset, cfg.context_geometry_manifest)
        dataset.reference_cache = ReferencePredictionCache(Path(cfg.reference_cache) / split, dataset, teacher_hash)
    return dataset, records


def anatomical_diagnostics(prediction, target):
    """Native-grid counts; tissue Dice and level errors never replace class Dice."""
    family = np.array([0] + [1] * 9 + [2] + [3] * 9)
    confusion = np.bincount(target.astype(np.int64).ravel() * 20 + prediction.ravel(),
                            minlength=400).reshape(20, 20)
    true_tissue, pred_tissue = family[target], family[prediction]
    tissue_dice = {}
    for value, name in ((1, "vertebra"), (2, "canal"), (3, "disc")):
        gt, pred = true_tissue == value, pred_tissue == value
        denominator = int(gt.sum() + pred.sum())
        tissue_dice[name] = float(2 * (gt & pred).sum() / denominator) if gt.any() else None
    numbered = (true_tissue == 1) | (true_tissue == 3)
    wrong_level = numbered & (true_tissue == pred_tissue) & (prediction != target)
    return {"tissue_dice": tissue_dice, "confusion_gt_rows_pred_columns": confusion.tolist(),
            "numbered_gt_pixels": int(numbered.sum()), "same_tissue_wrong_level_pixels": int(wrong_level.sum())}


def _context_case_slices(model, dataset, indices, record, device, network, batch_size):
    if network in ("teacher1", "teachers_mean"):
        backbone = model.teachers[0]
    elif network in ("student1", "students_mean"):
        backbone = model.students[0]
    else:
        raise ValueError(f"Unavailable single-network context prediction: {network}")
    names = record["slice_names"]
    def read(index):
        return dataset.read_image(indices[names[index]])
    slices = []
    options = {'read_position':lambda i:dataset.positions[names[i]]} if dataset.context_position=='relative' else {}
    if getattr(backbone, 'reference_condition', 'none') != 'none':
        from .reference_inference import iter_reference_logits
        cache = dataset.reference_cache
        iterator = iter_reference_logits(backbone, len(names), read,
            lambda i: cache.read(names[i]), device, batch_size,
            reference_index=cache.selected[record['case']])
    else:
        iterator = iter_cached_logits(backbone, len(names), read, device, batch_size, **options)
    for start, logits in iterator:
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite context logits")
        predictions = logits.argmax(1).float()[:, None]
        for offset, prediction in enumerate(predictions):
            # Shape comes only from the input image, never from a reference mask.
            with Image.open(dataset.root / "image" / names[start + offset]) as image:
                shape = image.size
            restored = F.interpolate(prediction[None], size=shape, mode="nearest")[0, 0]
            slices.append(restored.cpu().numpy().astype(np.uint8).T.copy())
    return slices


@torch.inference_mode()
def evaluate_volumes(model, dataset, records, device, networks=("teacher1",), batch_size=8):
    """Score named networks from the same in-memory checkpoint, independently.

    Each probability map is reduced to labels, resized with nearest-neighbour
    interpolation to the original transposed PNG shape, then transposed back
    before native 3D reconstruction. Only one case and one input minibatch are
    held at a time; no full-dataset probability maps are retained.
    """
    if batch_size < 1 or isinstance(networks, str):
        raise ValueError("batch_size must be positive and networks must be a sequence of names")
    networks = tuple(networks)
    if not networks or len(set(networks)) != len(networks):
        raise ValueError("networks must be nonempty and unique")
    if dataset.split == "train" or dataset.augment:
        raise ValueError("Native-volume evaluation requires unaugmented validation/test data")
    records = list(records)
    groups = group_complete_slices(dataset.names)
    if len({record["case"] for record in records}) != len(records):
        raise ValueError("Duplicate evaluation case records")
    if {record["case"]: record["slice_names"] for record in records} != groups:
        raise ValueError("Evaluation records must cover every complete dataset case in numeric slice order")
    indices = {name: index for index, name in enumerate(dataset.names)}
    cases = {name: [] for name in networks}
    was_training = model.training
    model.eval()
    try:
        for record in records:
            slices = {network: [] for network in networks}
            case_names = record["slice_names"]
            context = dataset.slice_context != "none"
            if context:
                slices = {network: _context_case_slices(model, dataset, indices, record, device,
                                                        network, batch_size) for network in networks}
            for start in ([] if context else range(0, len(case_names), batch_size)):
                samples = [dataset[indices[name]] for name in case_names[start:start + batch_size]]
                images = torch.stack([sample["image"] for sample in samples]).to(device)
                required = set()
                for network in networks:
                    if network in ("students_mean", "teachers_mean"):
                        prefix = "student" if network == "students_mean" else "teacher"
                        required.update(f"{prefix}{i+1}" for i in range(len(model.students)))
                    else:
                        required.add(network)
                cached = {network: model.predict_proba(images, network) for network in sorted(required)}
                for network in networks:
                    if network in ("students_mean", "teachers_mean"):
                        prefix = "student" if network == "students_mean" else "teacher"
                        probabilities = torch.stack([cached[f"{prefix}{i+1}"]
                                                     for i in range(len(model.students))]).mean(0)
                    else:
                        probabilities = cached[network]
                    if (probabilities.ndim != 4 or probabilities.shape[0] != len(samples)
                            or not 0 < probabilities.shape[1] <= 20
                            or not torch.isfinite(probabilities).all()):
                        raise ValueError(f"Invalid class probabilities from {network}")
                    predictions = probabilities.argmax(1).float()[:, None]
                    for prediction, sample in zip(predictions, samples):
                        restored = F.interpolate(prediction[None], size=sample["label"].shape,
                                                 mode="nearest")[0, 0]
                        slices[network].append(restored.cpu().numpy().astype(np.uint8).T.copy())
                    del probabilities, predictions
                del images, samples, cached
            reference = sitk.ReadImage(record["reference"])
            target = sitk.GetArrayFromImage(reference)
            if list(target.shape) != list(record["array_shape_zyx"]):
                raise ValueError(f"Reference changed since audit: {record['case']}")
            for network in networks:
                prediction = reconstruct(slices.pop(network), record["slice_axis"],
                                         record["undo_vertical_flip"], record["array_shape_zyx"])
                classes = volume_metrics(prediction, target, tuple(reversed(reference.GetSpacing())),
                                         surfaces=False)
                cases[network].append({"case": record["case"], "classes": classes,
                                       "anatomy": anatomical_diagnostics(prediction, target)})
                del prediction
            del target, reference
    finally:
        model.train(was_training)
    return {network: dict(aggregate_cases(rows), cases=rows, metric_policy=dict(METRIC_POLICY))
            for network, rows in cases.items()}
