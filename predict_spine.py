"""Segment one scalar MHA/NIfTI image without loading any annotation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time

import matplotlib
import numpy as np
import PIL
import SimpleITK as sitk
import torch

from freqssl.engine import seed_everything
from freqssl.inference import load_single_network, predict_volume, prediction_image, read_volume


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .mha prediction; never overwrites an existing file")
    parser.add_argument("--geometry-manifest", type=Path, help="Required for legacy converted volumes with incorrect headers")
    parser.add_argument("--network", choices=("teacher1", "teacher2", "student1", "student2"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reference-teacher", type=Path, help="Relocated frozen A checkpoint for reference-conditioned models")
    args = parser.parse_args(argv)
    receipt_path = args.output.with_suffix(".json")
    if args.output.suffix != ".mha":
        parser.error("Output must be .mha to preserve physical geometry in one file")
    if args.output.exists() or receipt_path.exists():
        raise FileExistsError("Prediction or its JSON receipt already exists")
    start = time.monotonic()
    record = None
    if args.geometry_manifest:
        manifest = json.loads(args.geometry_manifest.read_text())
        matches = [r for r in manifest["records"]
                   if Path(r["converted_image"]).resolve() == args.input.resolve()]
        if len(matches) != 1:
            raise ValueError("Need one manifest record for this exact converted image path")
        record = matches[0]
    else:
        known_manifest = Path(__file__).resolve().parent / "data/spider_image_geometry_manifest.json"
        if known_manifest.exists():
            known = json.loads(known_manifest.read_text())
            needs_repair = any(Path(r["converted_image"]).resolve() == args.input.resolve()
                               and r["metadata_fields_requiring_correction"] for r in known["records"])
            if needs_repair:
                raise ValueError(f"Known converted image has incorrect geometry; use --geometry-manifest {known_manifest}")
    array, plan, receipt = read_volume(args.input, record)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model, cfg, checkpoint_info = load_single_network(args.checkpoint, device, args.network)
    seed_everything(cfg)
    # Physical reorientation can change which slices enter the short final
    # batch. TF32 introduced near-tie label changes across CUDA batch shapes
    # in a real-image check. Use full float32 for this new deployment path.
    # Frozen training/evaluation protocols are intentionally not modified.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    labels = predict_volume(model, array, plan, cfg.image_size, device, args.batch_size,
                            str(args.reference_teacher) if args.reference_teacher else None)
    result = prediction_image(labels, receipt["output_geometry"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".mha", delete=False) as stream:
            temporary = Path(stream.name)
        sitk.WriteImage(result, str(temporary), useCompression=True)
        # Hard-link creation fails if another process has produced the output.
        args.output.hardlink_to(temporary)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    receipt.update(checkpoint_info)
    if hasattr(model, 'reference_receipt'):
        receipt['reference_pipeline'] = model.reference_receipt
    code_root = Path(__file__).resolve().parent
    sources = [Path(__file__).resolve(), *sorted((code_root / "freqssl").glob("*.py"))]
    receipt.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": file_hash(args.checkpoint),
                    "output": str(args.output.resolve()), "output_sha256": file_hash(args.output),
                    "geometry_manifest": str(args.geometry_manifest.resolve()) if args.geometry_manifest else None,
                    "geometry_manifest_sha256": file_hash(args.geometry_manifest) if args.geometry_manifest else None,
                    "preprocessing": "PIL/Matplotlib gray8 per slice; bilinear resize; mean/std normalization",
                    "image_size": cfg.image_size, "device": str(device),
                    "batch_size": args.batch_size, "postprocessing": "none",
                    "numeric_settings": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                         "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                         "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
                    "source_sha256": {str(path.relative_to(code_root)): file_hash(path) for path in sources},
                    "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                                 "Pillow": PIL.__version__, "matplotlib": matplotlib.__version__,
                                 "SimpleITK": sitk.Version_VersionString()},
                    "label_semantics": "0 background; 1..9 relative vertebra IDs; 10 canal; 11..19 relative disc IDs",
                    "seconds": time.monotonic() - start})
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "receipt": str(receipt_path),
                      "network": receipt["network"], "annotation_files_read": 0,
                      "seconds": receipt["seconds"]}, indent=2))
    return receipt


if __name__ == "__main__":
    main()
