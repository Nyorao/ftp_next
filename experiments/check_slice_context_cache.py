"""Compare cached/ordinary inference on real validation images without labels."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from freqssl.data import SpiderSliceDataset
from freqssl.engine import seed_everything
from freqssl.inference import load_single_network
from freqssl.slice_context import iter_cached_logits
from spider_volume import group_complete_slices


@torch.inference_mode()
def check(checkpoint, device, cases):
    model, cfg, _ = load_single_network(checkpoint, device)
    seed_everything(cfg)
    data = SpiderSliceDataset(cfg.root_path, "valid", cfg.image_size, augment=False,
                              seed=cfg.seed, slice_context=cfg.slice_context)
    indices = {name: i for i, name in enumerate(data.names)}
    rows = []
    for case, names in list(group_complete_slices(data.names).items())[:cases]:
        images = torch.stack([data.read_image(indices[name]) for name in names])
        triplets = torch.stack([images[[max(0, i-1), i, min(len(names)-1, i+1)], 0]
                                for i in range(len(names))])
        expected = torch.cat([model.backbone(values.to(device)).cpu()
                              for values in triplets.split(cfg.eval_batch_size)])
        for batch in (1, cfg.eval_batch_size):
            reads = []
            def read(i):
                reads.append(i)
                return images[i]
            actual = torch.cat([value.cpu() for _, value in iter_cached_logits(
                model.backbone, len(names), read, device, batch)])
            row = {"case": case, "slices": len(names), "output_batch": batch,
                   "max_logit_error": float((actual - expected).abs().max()),
                   "max_probability_error": float((actual.softmax(1)-expected.softmax(1)).abs().max()),
                   "changed_pixels": int((actual.argmax(1) != expected.argmax(1)).sum()),
                   "every_real_slice_read_once": reads == list(range(len(names)))}
            print(json.dumps(row), flush=True)
            rows.append(row)
            if row["changed_pixels"] or row["max_logit_error"] > 1e-5 or not row["every_real_slice_read_once"]:
                raise AssertionError(row)
    return {"status": "passed", "device": device, "checkpoint": str(checkpoint),
            "annotation_files_read": 0, "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = check(args.checkpoint, args.device, args.cases)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
