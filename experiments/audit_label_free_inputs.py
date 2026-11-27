"""Compare image-only geometric preprocessing with every legacy PNG input.

This is an input-equivalence audit, not a segmentation evaluation. It never
opens labels or predictions, including when valid/test image inputs are read.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freqssl.geometry import canonicalize
from freqssl.inference import legacy_gray8, prepare_slice, read_volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--png-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    rows = []
    for record in manifest["records"]:
        raw, plan, _ = read_volume(record["converted_image"], record)
        canonical = canonicalize(raw, plan)
        # Legacy PNG IDs follow the converted array's left/right order.
        # The manifest must actually support this correspondence, not guess it.
        if plan["transpose_zyx_to_slice_ap_si"] != [2, 1, 0]:
            raise ValueError("Audit needs an explicit PNG slice-index mapping for this layout")
        mismatched_pixels = mismatched_slices = mismatched_tensors = 0
        max_input_difference = 0.0
        png_digest = hashlib.sha256()
        for index, raw_slice in enumerate(canonical):
            source_index = len(canonical) - 1 - index if 0 in plan["flip_canonical_axes"] else index
            name = f"{record['case']}_{source_index:02}_{len(canonical)}.png"
            path = args.png_root / record["split"] / "image" / name
            with Image.open(path) as image:
                expected = np.array(image).T
            png_digest.update(name.encode())
            png_digest.update(expected.tobytes())
            actual = legacy_gray8(raw_slice)
            if actual.shape != expected.shape:
                raise ValueError(f"Shape mismatch: {name}")
            different = int(np.count_nonzero(actual != expected))
            mismatched_pixels += different
            mismatched_slices += int(different > 0)
            # Independent legacy preprocessing, using only the PNG image.
            old = torch.from_numpy(expected.astype(np.float32))[None, None]
            old = F.interpolate(old, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)[0]
            old = (old - old.mean()) / old.std(unbiased=False).clamp_min(1e-6)
            new = prepare_slice(raw_slice, args.image_size)
            mismatched_tensors += int(not torch.equal(old, new))
            max_input_difference = max(max_input_difference, float((old - new).abs().max()))
        rows.append({"case": record["case"], "split": record["split"], "slices": len(canonical),
                     "png_input_sha256": png_digest.hexdigest(),
                     "mismatched_pixels": mismatched_pixels, "mismatched_slices": mismatched_slices,
                     "mismatched_normalized_tensors": mismatched_tensors,
                     "max_normalized_input_difference": max_input_difference})
        if len(rows) % 25 == 0:
            print(f"Audited {len(rows)}/{len(manifest['records'])} volumes", flush=True)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest": str(args.manifest.resolve()),
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "image_size": args.image_size, "label_files_read": 0, "model_inference_performed": False,
              "purpose": "Image geometry/intensity/input equivalence only; no test-set model development",
              "volumes": len(rows), "slices": sum(r["slices"] for r in rows),
              "mismatched_pixels": sum(r["mismatched_pixels"] for r in rows),
              "mismatched_slices": sum(r["mismatched_slices"] for r in rows),
              "mismatched_normalized_tensors": sum(r["mismatched_normalized_tensors"] for r in rows),
              "seconds": time.monotonic() - start, "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if result["mismatched_slices"] or result["mismatched_normalized_tensors"]:
        raise SystemExit("Input mismatch; retain the audit and investigate before claiming equivalence")


if __name__ == "__main__":
    main()
