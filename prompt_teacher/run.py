"""Fit a frozen-backbone teacher or export image-only training guidance."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional as F

from freqssl.anatomy import FAMILY_IDS
from freqssl.config import load_config
from freqssl.data import build_training_data
from freqssl.engine import seed_everything, state_hash, write_json, recover_log
from freqssl.evaluation import prepare_validation
from freqssl.objectives import segmentation_loss
from freqssl.prompt_supervision import file_sha256
from spider_volume import reconstruct
from .data import TissueDataset, image_seed, supervision_fingerprint
from .model import build_teacher

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = [*sorted((ROOT / "prompt_teacher").rglob("*.py")), *sorted((ROOT / "freqssl").glob("*.py")),
             ROOT / "spider_volume.py", ROOT / "experiments/run_prompt_context.py",
             ROOT / "experiments/run_prompt_revision.py"]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in paths if p.exists()}


def protocol(cfg, source):
    return {"source": source, "seed": cfg.seed, "root_path": cfg.root_path,
        "labelnum": cfg.labelnum, "max_samples": cfg.max_samples,
        "steps": 6000, "batch_size": 4, "image_size": 224, "eval_every": 500,
        "learning_rate": .0003, "weight_decay": .0001, "poly_exponent": .9,
        "gradient_clip": 12., "mc_samples": 8, "evaluation_batch_size": 4,
        "selection": "mean of per-tissue, case-mean native valid Dice; ties keep earlier step",
        "supervision": "0.5 four-class CE + 0.5 squared-denominator soft Dice including background",
        "amp": False, "gradient_checkpointing": True}


def save_teacher(path, model, optimizer, recipe, step, best, provenance):
    payload = {"format_version": 1, "recipe": recipe, "step": step, "best_score": best,
        "adaptation": model.adaptation_state(), "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "provenance": provenance}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate_teacher(model, dataset, records, device, seed):
    indices = {name: i for i, name in enumerate(dataset.names)}
    rows = []
    was_training = model.training
    model.eval()
    try:
        for record in records:
            slices = []
            for start in range(0, len(record["slice_names"]), 4):
                names = record["slice_names"][start:start + 4]
                samples = [dataset[indices[name]] for name in names]
                images = torch.stack([s["image"] for s in samples]).to(device)
                if len(samples) < 4:
                    images = torch.cat((images, torch.zeros((4 - len(samples), *images.shape[1:]), device=device)))
                _, tissue, _, _ = model.predict_mc(images, image_seed("|".join(names), seed))
                for pred, sample in zip(tissue, samples):
                    native = F.interpolate(pred[None, None].float(), sample["native_shape"], mode="nearest")[0, 0]
                    slices.append(native.cpu().numpy().astype(np.uint8).T.copy())
            prediction = reconstruct(slices, record["slice_axis"], record["undo_vertical_flip"], record["array_shape_zyx"])
            target = np.array(FAMILY_IDS)[sitk.GetArrayFromImage(sitk.ReadImage(record["reference"]))]
            scores = {}
            for value, name in ((1, "vertebra"), (2, "canal"), (3, "disc")):
                g, p = target == value, prediction == value
                scores[name] = float(2 * (g & p).sum() / (g.sum() + p.sum())) if g.any() else None
            rows.append({"case": record["case"], "tissue_dice": scores})
    finally:
        model.train(was_training)
    means = {name: float(np.mean([r["tissue_dice"][name] for r in rows if r["tissue_dice"][name] is not None]))
             for name in ("vertebra", "canal", "disc")}
    return {"dice": float(np.mean(list(means.values()))), "tissue_dice": means, "cases": rows,
            "split": "valid", "mc_samples": 8, "native_volume_evaluation": True}


def fit(cfg, source, output, device, resume=None, stop_after=None):
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite teacher run {output}")
    cfg = replace(cfg, prompt_source="none", prompt_cache="", slice_context="none")
    seed_everything(cfg)
    recipe = protocol(cfg, source)
    _, _, split = build_training_data(cfg)
    validation, records = prepare_validation(cfg)
    model, metadata = build_teacher(source, device, cfg.seed)
    base_hash = state_hash(model.base)
    provenance = {**metadata, "foundation_sha256": file_sha256(metadata["checkpoint"]),
        "initial_frozen_state_sha256": base_hash,
        "initial_adaptation_sha256": {k: state_hash(getattr(model, k)) for k in ("adapters", "mask_head", "upscale")},
        "prompts": list(model.prompts), "source_hashes": source_hashes(),
        "torch_version": str(torch.__version__), "gpu": torch.cuda.get_device_name(device),
        "unlabeled_annotations_opened": False, "test_evaluated": False,
        **supervision_fingerprint(cfg.root_path, split)}
    training = TissueDataset(cfg.root_path, "train", split["labeled_files"], metadata["mean"], metadata["std"],
                             augment=True, seed=cfg.seed)
    valid = TissueDataset(cfg.root_path, "valid", validation.names, metadata["mean"], metadata["std"], image_only=True)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=.0003, weight_decay=.0001)
    step, best = 0, -1.
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["recipe"] != recipe or saved["provenance"] != provenance:
            raise ValueError("Teacher resume protocol, weights, labels or source changed")
        if json.loads((output / "split.json").read_text()) != split:
            raise ValueError("Teacher resume split changed")
        model.load_adaptation(saved["adaptation"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, best = saved["step"], saved["best_score"]
        recover_log(output, step)
    else:
        write_json(output / "recipe.json", recipe)
        write_json(output / "split.json", split)
        write_json(output / "provenance.json", provenance)
        write_json(output / "validation_audit.json", records)
    model.train()
    steps_per_epoch = len(training) // 4
    end = min(6000, step + stop_after) if stop_after else 6000
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with (output / "metrics.jsonl").open("a", buffering=1) as log:
            while step < end:
                epoch, offset = divmod(step, steps_per_epoch)
                training.set_epoch(epoch)
                indices = np.random.RandomState(cfg.seed + epoch).permutation(len(training))
                for batch_index in range(offset, steps_per_epoch):
                    samples = [training[int(i)] for i in indices[batch_index * 4:batch_index * 4 + 4]]
                    images = torch.stack([s["image"] for s in samples]).to(device)
                    targets = torch.stack([s["label"] for s in samples]).to(device)
                    lr = .0003 * (1 - step / 6000) ** .9
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = segmentation_loss(logits, targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite teacher loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 12., error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    entry = {"kind": "train", "step": step, "loss": float(loss.detach()), "lr": lr,
                             "grad_norm": float(norm), "epoch": step / steps_per_epoch}
                    log.write(json.dumps(entry, allow_nan=False) + "\n")
                    if step == 1 or step % 25 == 0:
                        status = {"status": "running", "stage": "fit", "source": source, "step": step,
                            "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started,
                            "best_dice": None if best < 0 else best}
                        write_json(output / "status.json", status)
                        print(f"TEACHER {source} step={step}/6000 loss={entry['loss']:.5f}", flush=True)
                    if step % 500 == 0:
                        scores = evaluate_teacher(model, valid, records, device, cfg.seed)
                        write_json(output / f"valid_step_{step}.json", scores)
                        log.write(json.dumps({"kind": "validation", "step": step, "dice": scores["dice"]}) + "\n")
                        if scores["dice"] > best:
                            best = scores["dice"]
                            save_teacher(output / "best.pt", model, optimizer, recipe, step, best, provenance)
                        print(f"TEACHER VALID {source} step={step} dice={scores['dice']:.6f}", flush=True)
                    if step % 100 == 0 or step == end:
                        save_teacher(output / "last.pt", model, optimizer, recipe, step, best, provenance)
                    if step >= end:
                        break
        if state_hash(model.base) != base_hash:
            raise AssertionError("Foundation weights changed during adaptation")
        result = {"status": "completed" if step == 6000 else "paused", "step": step,
            "best_valid_tissue_dice": None if best < 0 else best,
            "seconds_this_invocation": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "trainable_parameters": sum(p.numel() for p in model.trainable_parameters()),
            "frozen_weights_unchanged": True, "test_evaluated": False}
        write_json(output / "result.json", result)
        write_json(output / "status.json", result)
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "step": step, "error": repr(error),
                   "last_checkpoint": str(output / "last.pt") if (output / "last.pt").exists() else None})
        raise


@torch.inference_mode()
def export_cache(cfg, source, checkpoint_path, output, device):
    cfg = replace(cfg, prompt_source="none", prompt_cache="", slice_context="none")
    seed_everything(cfg)
    _, _, split = build_training_data(cfg)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["recipe"] != protocol(cfg, source):
        raise ValueError("Teacher checkpoint recipe differs")
    if json.loads((checkpoint_path.parent / "split.json").read_text()) != split:
        raise ValueError("Teacher checkpoint used a different label budget")
    fingerprints = supervision_fingerprint(cfg.root_path, split)
    if any(saved["provenance"].get(key) != value for key, value in fingerprints.items()):
        raise ValueError("Authorized teacher supervision changed")
    model, metadata = build_teacher(source, device, cfg.seed)
    if file_sha256(metadata["checkpoint"]) != saved["provenance"]["foundation_sha256"]:
        raise ValueError("Foundation checkpoint changed")
    model.load_adaptation(saved["adaptation"])
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite cache {output}")
    dataset = TissueDataset(cfg.root_path, "train", split["unlabeled_files"], metadata["mean"], metadata["std"], image_only=True)
    entries = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    accepted = predicted = np.zeros(4, dtype=np.int64)
    accepted, predicted = accepted.copy(), predicted.copy()
    for start in range(0, len(dataset), 4):
        samples = [dataset[i] for i in range(start, min(start + 4, len(dataset)))]
        names = [s["file_name"] for s in samples]
        images = torch.stack([s["image"] for s in samples]).to(device)
        if len(samples) < 4:
            images = torch.cat((images, torch.zeros((4 - len(samples), *images.shape[1:]), device=device)))
        q, tissue, trusted, votes = model.predict_mc(images, image_seed("|".join(names), cfg.seed))
        for i, name in enumerate(names):
            file = Path(name).stem + ".npz"
            labels, mask = tissue[i].cpu().numpy().astype(np.uint8), trusted[i].cpu().numpy()
            temporary = output / (file + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, probabilities=q[i].cpu().numpy().astype(np.float16), tissue=labels,
                    trusted=mask, valid=np.ones_like(mask), votes=votes[i].cpu().numpy().astype(np.uint8))
            temporary.replace(output / file)
            entries[name] = {"file": file, "sha256": file_sha256(output / file),
                "image_sha256": file_sha256(Path(cfg.root_path) / "train/image" / name)}
            accepted += np.bincount(labels[mask], minlength=4)
            predicted += np.bincount(labels.ravel(), minlength=4)
        if start % 100 == 0:
            write_json(output / "status.json", {"status": "running", "slices": len(entries), "total": len(dataset)})
            print(f"CACHE {source} {len(entries)}/{len(dataset)}", flush=True)
    manifest = {"format_version": 1, "status": "completed", "source": source, "prompts": list(model.prompts),
        "root_path": split["root_path"], "labeled_files": split["labeled_files"], "unlabeled_files": split["unlabeled_files"],
        "image_size": 224, "family_ids": list(FAMILY_IDS), "mc_samples": 8,
        "mc_seed_policy": "SHA256 of ordered batch filenames plus seed; fixed batch four, zero-padded tail",
        "thresholds": {"foreground": .8, "background": .95, "minimum_votes": 7},
        "teacher_checkpoint": str(checkpoint_path.resolve()), "teacher_checkpoint_sha256": file_sha256(checkpoint_path),
        "teacher_step": saved["step"], "foundation": metadata, "entries": entries,
        "unlabeled_annotations_opened": False, "test_evaluated": False,
        "accepted_pixels_per_tissue": accepted.tolist(), "predicted_pixels_per_tissue": predicted.tolist(),
        "coverage": float(accepted.sum() / predicted.sum()), "seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), **fingerprints}
    write_json(output / "manifest.json", manifest)
    write_json(output / "status.json", {"status": "completed", "slices": len(entries)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "cache"))
    parser.add_argument("--config", default="configs/v3/prompt_context.yaml")
    parser.add_argument("--source", choices=("text", "neutral"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    if args.stage == "fit":
        fit(cfg, args.source, args.output.resolve(), device, args.resume, args.stop_after)
    else:
        if not args.checkpoint:
            parser.error("cache requires --checkpoint")
        export_cache(cfg, args.source, args.checkpoint.resolve(), args.output.resolve(), device)


if __name__ == "__main__":
    main()
