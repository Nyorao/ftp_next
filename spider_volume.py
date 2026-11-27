"""Native-volume SPIDER reconstruction and explicit foreground metric policies."""

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, find_objects
import SimpleITK as sitk


# Matches the actual label converter and class_embedding/bert_embedding_spider.py.
CLASS_NAMES = (
    "background", "L5 vertebra", "L4 vertebra", "L3 vertebra", "L2 vertebra",
    "L1 vertebra", "T12 vertebra", "T11 vertebra", "T10 vertebra", "T9 vertebra",
    "spinal canal", "L5/S1 disc", "L4/L5 disc", "L3/L4 disc", "L2/L3 disc",
    "L1/L2 disc", "T12/L1 disc", "T11/T12 disc", "T10/T11 disc", "T9/T10 disc",
)
METRICS = ("dice", "iou", "hd95_mm", "asd_mm")
PATTERN = re.compile(r"^(SPIDER_\d+)_(\d+)_(\d+)\.png$")


def parse_name(name):
    match = PATTERN.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"Invalid SPIDER slice name: {name}")
    case, index, count = match.group(1), int(match.group(2)), int(match.group(3))
    if count <= 0 or not 0 <= index < count:
        raise ValueError(f"Invalid slice index/count: {name}")
    return case, index, count


def group_complete_slices(names):
    groups = defaultdict(list)
    for name in names:
        case, index, count = parse_name(name)
        groups[case].append((index, count, name))
    output = {}
    for case, items in sorted(groups.items()):
        counts = {item[1] for item in items}
        if len(counts) != 1:
            raise ValueError(f"{case}: inconsistent declared slice counts")
        count = counts.pop()
        items.sort()
        if [item[0] for item in items] != list(range(count)):
            raise ValueError(f"{case}: duplicate or missing slices")
        output[case] = [item[2] for item in items]
    return output


def layout_from_reference(reference_array, count):
    shape = reference_array.shape
    candidates = [axis for axis, size in enumerate(shape) if size == count and size == min(shape)]
    if len(shape) != 3 or len(candidates) != 1:
        raise ValueError(f"Ambiguous/incompatible volume shape {shape}, count={count}")
    # Invert the fixed original mha2png.py rule, before inspecting any prediction.
    one = np.where(reference_array == 1)[0]
    three = np.where(reference_array == 3)[0]
    keep = bool(len(one) and len(three) and one.mean() > three.mean())
    return candidates[0], not keep


def reconstruct(slices, axis, flip_rows, expected_shape):
    arrays = [np.flip(value, axis=0) if flip_rows else value for value in slices]
    volume = np.stack(arrays, axis=axis)
    if volume.shape != tuple(expected_shape):
        raise ValueError(f"Reconstruction shape {volume.shape} != {expected_shape}")
    return volume


def geometry(image):
    return {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_xyz": list(image.GetOrigin()), "direction": list(image.GetDirection())}


def same_geometry(first, second):
    return first.GetSize() == second.GetSize() and all(
        np.allclose(getattr(first, method)(), getattr(second, method)(), atol=1e-6, rtol=0)
        for method in ("GetSpacing", "GetOrigin", "GetDirection"))


def audit_case(png_root, volume_root, split, case, names):
    reference_path = Path(volume_root) / split / "label" / f"{case}.mha"
    reference = sitk.ReadImage(str(reference_path))
    target = sitk.GetArrayFromImage(reference)
    if not set(np.unique(target)).issubset(range(20)):
        raise ValueError(f"Unexpected reference labels in {reference_path}")
    axis, flip = layout_from_reference(target, len(names))
    slices = []
    for name in names:
        for folder in ("image", "low_frequency", "high_frequency", "label"):
            if not (Path(png_root) / split / folder / name).is_file():
                raise FileNotFoundError(f"Missing {split}/{folder}/{name}")
        with Image.open(Path(png_root) / split / "label" / name) as image:
            slices.append(np.asarray(image).copy())
    restored = reconstruct(slices, axis, flip, target.shape)
    mismatches = int(np.count_nonzero(restored != target))
    if mismatches:
        raise ValueError(f"{case}: PNG label round-trip has {mismatches} mismatched voxels")
    image = sitk.ReadImage(str(Path(volume_root) / split / "image" / f"{case}.mha"))
    if not same_geometry(image, reference):
        raise ValueError(f"{case}: MHA image and label geometry differ")
    direction = np.asarray(reference.GetDirection()).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError(f"{case}: non-orthonormal direction requires physical-coordinate distances")
    if min(reference.GetSpacing()) <= 0:
        raise ValueError(f"{case}: invalid voxel spacing")
    return {"case": case, "slice_count": len(names), "slice_names": names,
            "reference": str(reference_path), "array_shape_zyx": list(target.shape),
            "slice_axis": axis, "undo_vertical_flip": flip, "geometry": geometry(reference),
            "label_roundtrip_mismatched_voxels": mismatches, "image_label_geometry_equal": True,
            "gt_class_voxels": np.bincount(target.ravel(), minlength=20).tolist()}


def save_volume(path, array, reference):
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.CopyInformation(reference)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.SetUseCompression(True)
    writer.SetCompressionLevel(1)
    writer.Execute(image)
    restored = sitk.ReadImage(str(path))
    if not same_geometry(restored, reference) or not np.array_equal(sitk.GetArrayFromImage(restored), array):
        raise RuntimeError(f"Saved volume failed geometry/voxel round-trip: {path}")


def surface_distances(prediction, target, spacing):
    # Default connectivity=1 matches the original MedPy evaluator (six neighbours).
    pred_surface = prediction ^ binary_erosion(prediction)
    gt_surface = target ^ binary_erosion(target)
    to_gt = distance_transform_edt(~gt_surface, sampling=spacing)[pred_surface]
    to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[gt_surface]
    return float(np.percentile(np.concatenate((to_gt, to_pred)), 95)), float(to_gt.mean())


def volume_metrics(prediction, target, spacing_zyx, surfaces=True):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Metrics require matching 3D arrays")
    if not set(np.unique(prediction)).issubset(range(20)):
        raise ValueError("Predictions must contain only class IDs 0..19")
    code = target.astype(np.int32).ravel() * 20 + prediction.ravel()
    confusion = np.bincount(code, minlength=400).reshape(20, 20)
    gt_counts, pred_counts = confusion.sum(1), confusion.sum(0)
    gt_boxes = find_objects(target, max_label=19) if surfaces else None
    pred_boxes = find_objects(prediction, max_label=19) if surfaces else None
    rows = []
    for label in range(1, 20):
        g, p, intersection = int(gt_counts[label]), int(pred_counts[label]), int(confusion[label, label])
        dice = 2 * intersection / (g + p) if g + p else None
        iou = intersection / (g + p - intersection) if g + p else None
        hd95 = asd = None
        if surfaces and g and p:
            # Cropping the union of both masks preserves their surface distances.
            boxes = (gt_boxes[label - 1], pred_boxes[label - 1])
            region = tuple(slice(max(0, min(b[i].start for b in boxes) - 1),
                                 min(target.shape[i], max(b[i].stop for b in boxes) + 1)) for i in range(3))
            hd95, asd = surface_distances(prediction[region] == label, target[region] == label, spacing_zyx)
        rows.append({"class_id": label, "class_name": CLASS_NAMES[label],
                     "gt_voxels": g, "pred_voxels": p, "intersection_voxels": intersection,
                     "dice": dice if g else None, "iou": iou if g else None,
                     "dice_union_present": dice, "iou_union_present": iou,
                     "hd95_mm": hd95, "asd_mm": asd,
                     "missed_gt_class": bool(g and not p),
                     "false_positive_absent_class": bool(p and not g),
                     "surface_metrics_requested": surfaces})
    return rows


def mean_defined(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def aggregate_cases(cases):
    metrics = (*METRICS, "dice_union_present", "iou_union_present")
    per_class = []
    for label in range(1, 20):
        rows = [case["classes"][label - 1] for case in cases]
        per_class.append({"class_id": label, "class_name": CLASS_NAMES[label],
                          **{metric: mean_defined(row[metric] for row in rows) for metric in metrics},
                          "gt_present_cases": sum(row["gt_voxels"] > 0 for row in rows),
                          "pred_present_cases": sum(row["pred_voxels"] > 0 for row in rows),
                          "missed_gt_cases": sum(row["missed_gt_class"] for row in rows),
                          "fp_absent_gt_cases": sum(row["false_positive_absent_class"] for row in rows),
                          "valid_surface_cases": sum(row["hd95_mm"] is not None for row in rows)})
    per_case = [{"case": case["case"], **{metric: mean_defined(row[metric] for row in case["classes"])
                 for metric in metrics}} for case in cases]
    overall = {metric: mean_defined(case[metric] for case in per_case) for metric in metrics}
    overall.update(cases=len(cases),
                   gt_present_class_case_pairs=sum(row["gt_present_cases"] for row in per_class),
                   missed_gt_class_case_pairs=sum(row["missed_gt_cases"] for row in per_class),
                   fp_absent_gt_class_case_pairs=sum(row["fp_absent_gt_cases"] for row in per_class),
                   valid_surface_class_case_pairs=sum(row["valid_surface_cases"] for row in per_class))
    return {"overall": overall, "per_class": per_class, "per_case": per_case}
