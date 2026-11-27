"""Image-only context deployment check on two train cases, including LR reversal."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from freqssl.data import SpiderSliceDataset
from freqssl.engine import seed_everything
from freqssl.geometry import canonicalize, restore_native
from freqssl.inference import load_single_network, predict_volume, prepare_slice, read_volume
from spider_volume import group_complete_slices


@torch.inference_mode()
def check(checkpoint, device):
    model, cfg, _ = load_single_network(checkpoint, device)
    seed_everything(cfg)
    # Match predict_spine.py's documented deployment precision.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    data = SpiderSliceDataset(cfg.root_path, "train", cfg.image_size,
                              augment=False, slice_context=cfg.slice_context, labeled_names=[])
    groups = group_complete_slices(data.names)
    indices = {name: i for i, name in enumerate(data.names)}
    manifest = json.loads((ROOT / "data/spider_image_geometry_manifest.json").read_text())
    records = [r for r in manifest["records"] if r["split"] == "train" and r["case"] in ("SPIDER_001", "SPIDER_006")]
    rows = []
    for record in records:
        array, plan, _ = read_volume(record["converted_image"], record)
        canonical = canonicalize(array, plan)
        reverse = 0 in plan["flip_canonical_axes"]
        ordered = canonical[::-1] if reverse else canonical
        images = torch.stack([data.read_image(indices[n]) for n in groups[record["case"]]])
        raw_inputs = torch.stack([prepare_slice(value, cfg.image_size) for value in ordered])
        torch.testing.assert_close(images, raw_inputs, rtol=0, atol=0)
        triplets = torch.stack([images[[max(0, i-1), i, min(len(images)-1, i+1)], 0]
                                for i in range(len(images))])
        outputs = []
        for values in triplets.split(8):
            labels = model(values.to(device)).argmax(1).float()[:, None]
            outputs.append(F.interpolate(labels, size=canonical.shape[1:], mode="nearest")[:, 0].cpu().numpy().astype(np.uint8))
        expected = np.concatenate(outputs)
        expected = restore_native(expected[::-1] if reverse else expected, plan)
        actual = predict_volume(model, array, plan, cfg.image_size, device, 8)
        raw, raw_plan, _ = read_volume(record["source_image"])
        raw_labels = predict_volume(model, raw, raw_plan, cfg.image_size, device, 8)
        transform = record["verified_transform"]
        raw_labels = raw_labels.transpose(transform["transpose_zyx"])
        if transform["flip_axes_zyx"]:
            raw_labels = np.flip(raw_labels, tuple(transform["flip_axes_zyx"]))
        row = {"case": record["case"], "slices": len(images), "lr_reversed": reverse,
               "png_and_volume_inputs_exact": True,
               "png_vs_converted_changed_voxels": int(np.count_nonzero(expected != actual)),
               "original_vs_converted_changed_voxels": int(np.count_nonzero(raw_labels != actual))}
        print(json.dumps(row), flush=True)
        rows.append(row)
        assert row["png_vs_converted_changed_voxels"] == row["original_vs_converted_changed_voxels"] == 0, row
    assert len(rows) == 2
    return {"status": "passed", "device": device, "checkpoint": str(checkpoint),
            "annotation_files_read": 0, "test_evaluated": False, "tf32": False, "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(check(args.checkpoint, args.device), indent=2) + "\n")
