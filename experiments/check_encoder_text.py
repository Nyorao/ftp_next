"""Three full-size real-training-batch updates per arm; no efficacy evaluation."""
import argparse
import gc
import hashlib
import itertools
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data._utils.collate import default_collate

from experiments.run_encoder_text import ARMS, arm_config
from experiments.run_hierarchical_text import initialize_device
from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.engine import make_optimizer, seed_everything, train_step, write_json
from freqssl.models import StudentTeacherSystem
from freqssl.augment import strong_view


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for value in tensors:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(config, bank, output, device, *, arms=ARMS, make_config=arm_config, center_pairing=False):
    output.mkdir(parents=True, exist_ok=False)
    arm = None
    try:
        base = load_config(config)
        device = initialize_device(device)
        seed_everything(base)
        dataset, sampler, split = build_training_data(base)
        indices = list(itertools.islice(iter(sampler), 3))
        batches = [default_collate([dataset[index] for index in batch]) for batch in indices]
        before = tensor_digest([b[k] for b in batches for k in ("image", "label", "is_labeled")])
        write_json(output / "inputs.json", dict(indices=indices, batch_sha256=before, split=split,
            real_training_data_only=True, steps_per_arm=3, formal_budget_unchanged=True,
            inference_or_efficacy_claim=False))
        rows, augmentation_reference = [], None
        for arm in arms:
            cfg = make_config(base, arm, bank, output / arm)
            arm_batches = ([dict(batch, image=batch["image"][:, 1:2]) for batch in batches]
                           if center_pairing and cfg.slice_context == "none" else batches)
            seed_everything(cfg)
            model = StudentTeacherSystem(cfg).to(device)
            optimizer = make_optimizer(model, cfg)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            frozen = {k:v.detach().cpu().clone() for k,v in model.students[0].named_buffers()
                      if k.startswith("encoder_text.")}
            write_json(output / "status.json", dict(status="running", arm=arm, pid=os.getpid(), device=str(device)))
            started = time.monotonic()
            augmented_images = [strong_view(batch["image"], cfg.seed + step * 104729,
                cfg.noise_std, cfg.gain_std) for step,batch in enumerate(arm_batches)]
            augmented = tensor_digest([value[:, 1:2] if center_pairing and value.shape[1] == 3 else value
                                       for value in augmented_images])
            if augmentation_reference is None:
                augmentation_reference = augmented
            if augmented != augmentation_reference:
                raise AssertionError("Paired strong augmentation changed")
            metrics = [train_step(model, optimizer, batch, cfg, device, step) for step,batch in enumerate(arm_batches)]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for collection in (model.students, model.teachers):
                buffers = dict(collection[0].named_buffers())
                if any(not torch.equal(v, buffers[k].detach().cpu()) for k,v in frozen.items()):
                    raise AssertionError("Frozen text assets changed")
            if any(p.grad is not None for p in model.teachers.parameters()):
                raise AssertionError("Teacher received gradients")
            active_stages = cfg.encoder_text_stages if cfg.text_fusion_layout == "encoder_multi" else (
                () if cfg.encoder_text_stage is None else (cfg.encoder_text_stage,))
            if cfg.internal_text != "none":
                module = model.students[0].encoder_text
                for name,p in module.named_parameters():
                    if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum() > 0):
                        raise AssertionError(f"No finite active gradient: {arm}:{name}")
                if any(module.gates[stage-2].detach() == 0 for stage in active_stages):
                    raise AssertionError("Selected gate stayed zero")
            rows.append(dict(arm=arm, updates=3, seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                metrics=metrics, strong_view_sha256=augmented,
                trainable_parameters=sum(p.numel() for p in model.students[0].parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.students[0].parameters())))
            write_json(output / "checks.json", dict(status="partial", arms=rows, efficacy_verified=False))
            print(f"SMOKE {arm}: 3 updates, loss={metrics[-1]['loss']:.6f}, peak={rows[-1]['peak_allocated_bytes']}", flush=True)
            del model, optimizer, frozen
            # Loop locals can retain parameters until the next iteration.
            if cfg.internal_text != "none":
                del module, p
            del collection, buffers
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        after = tensor_digest([b[k] for b in batches for k in ("image", "label", "is_labeled")])
        if before != after:
            raise AssertionError("Training mutated shared input batches")
        write_json(output / "checks.json", dict(status="passed", arms=rows,
            identical_data_and_strong_augmentation=True, comparison_scope="center_slice" if center_pairing else "all_slices",
            efficacy_verified=False, test_evaluated=False))
        write_json(output / "status.json", dict(status="passed", arms=list(arms), total_updates=3*len(arms)))
    except BaseException as error:
        write_json(output / "status.json", dict(status="failed", arm=arm, error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/v3/context_efficiency_reference.yaml")
    parser.add_argument("--bank", type=Path, default=ROOT/"outputs/hierarchical_text_assets_20261108/text.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    run(args.config, args.bank, args.output.resolve(), args.device)


if __name__ == "__main__":
    main()
