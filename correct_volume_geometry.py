"""Rescore verified metadata corrections without changing data or checkpoints.

Every correction must match an original SPIDER mask exactly after the recorded
class mapping/axis transform. Original evaluations are retained separately.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import SimpleITK as sitk

from evaluate_volumes import export_csv, file_hash, write_json
from spider_volume import aggregate_cases, geometry, save_volume, volume_metrics


def map_raw_labels(array):
    mapped = array.copy()
    mapped[array == 100] = 10
    for label in range(201, 210):
        mapped[array == label] = label - 190
    return mapped


def verified_geometry(record, target):
    references = []
    evidence = []
    for match in record["matches"]:
        source = sitk.ReadImage(match["raw_mask"])
        # NumPy target ZYX = raw.transpose(perm); SimpleITK needs the XYZ order.
        order_xyz = [2 - axis for axis in reversed(match["transpose"])]
        transformed = sitk.PermuteAxes(source, order_xyz)
        flips_xyz = [2 - axis in match["flip_axes"] for axis in range(3)]
        if any(flips_xyz):
            transformed = sitk.Flip(transformed, flips_xyz)
        if not np.array_equal(map_raw_labels(sitk.GetArrayFromImage(transformed)), target):
            raise ValueError(f"Source mask does not reproduce the full reference: {match['raw_mask']}")
        references.append(transformed)
        evidence.append({"raw_mask": match["raw_mask"], "sha256": file_hash(match["raw_mask"]),
                         "sitk_axis_order_xyz": order_xyz, "flip_xyz": flips_xyz,
                         "corrected_geometry": geometry(transformed), "exact_label_match": True})
    if not references:
        raise ValueError("No verified source geometry")
    first = references[0]
    if any(not np.allclose(first.GetSpacing(), item.GetSpacing(), rtol=0, atol=1e-6) for item in references[1:]):
        raise ValueError("Source candidates disagree on spacing; cannot resolve without user input")
    # Different origins/rotations do not change within-volume Euclidean distances.
    # Record all candidates, and explicitly identify the chosen frame for export.
    return first, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation_dir.resolve()
    result_path = root / "results_geometry_corrected.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    original = json.loads((root / "results.json").read_text())
    if original["status"] != "completed":
        raise ValueError("Wait for complete inference/evaluation before correcting geometry")
    results = copy.deepcopy(original)
    trace = json.loads(args.trace.read_text())
    data_audit = json.loads((root / "data_audit.json").read_text())
    trace = {case: item for case, item in trace.items() if item["split"] in data_audit["splits"]}
    records = {(split, record["case"]): record for split, entry in data_audit["splits"].items()
               for record in entry["case_records"]}
    corrected_references = {}
    correction_audit = {}
    for case, item in trace.items():
        record = records[item["split"], case]
        old_reference = sitk.ReadImage(record["reference"])
        target = sitk.GetArrayFromImage(old_reference)
        corrected, evidence = verified_geometry(item, target)
        corrected_references[case] = corrected
        correction_audit[case] = {
            "split": item["split"], "provided_geometry": geometry(old_reference),
            "corrected_geometry": geometry(corrected), "evidence": evidence,
            "export_frame_source": item["matches"][0]["raw_mask"],
            "provided_dataset_modified": False,
        }
    cached = {}
    for split, entries in results["splits"].items():
        for identity, payload in entries.items():
            surfaces = payload["network"] == "teacher"
            for case_result in payload["cases"]:
                case = case_result["case"]
                if case not in corrected_references or trace[case]["split"] != split:
                    continue
                key = (case_result["prediction"], surfaces)
                if key not in cached:
                    prediction = sitk.GetArrayFromImage(sitk.ReadImage(case_result["prediction"]))
                    target = sitk.GetArrayFromImage(sitk.ReadImage(records[split, case]["reference"]))
                    reference = corrected_references[case]
                    path = Path(case_result["prediction"]).parent.parent / "mha_geometry_corrected" / f"{case}.mha"
                    path.parent.mkdir(exist_ok=True)
                    save_volume(path, prediction, reference)
                    classes = volume_metrics(prediction, target, tuple(reversed(reference.GetSpacing())), surfaces)
                    for old, new in zip(case_result["classes"], classes):
                        if old["dice"] != new["dice"] or old["iou"] != new["iou"]:
                            raise AssertionError("Spacing correction must not change overlap metrics")
                    cached[key] = {"classes": classes, "geometry_corrected_prediction": str(path),
                                   "geometry_corrected_from_raw_mask": True}
                case_result.update(cached[key])
            payload.update(aggregate_cases(payload["cases"]))
            directory = root / split / identity
            write_json(directory / "result_geometry_corrected.json", payload)
            export_csv(directory / "per_class_geometry_corrected.csv", payload["per_class"])
            export_csv(directory / "per_case_geometry_corrected.csv", payload["per_case"])
            print(f"CORRECTED {split}/{identity}", flush=True)
    results["geometry_correction"] = {
        "original_results": str(root / "results.json"), "trace": str(args.trace.resolve()),
        "trace_sha256": file_hash(args.trace), "cases": sorted(corrected_references),
        "source": "exact full-volume label matches to local original SPIDER masks after recorded axis permutation",
        "overlap_metrics_unchanged": True, "dataset_and_checkpoints_modified": False,
        "correction_code_sha256": file_hash(__file__),
    }
    shutil.copy2(__file__, root / "sources" / Path(__file__).name)
    write_json(root / "geometry_correction_audit.json", correction_audit)
    write_json(result_path, results)


if __name__ == "__main__":
    main()
