"""Paired, bounded SPIDER pilot for the single-teacher two-student design.

This is a short validation experiment, not a convergence/benchmark claim.
Run from any directory; all outputs are isolated under --output-dir.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from config import load_config
from dataset import SpiderDataset, TwoStreamBatchSampler
from trainer import (build_model, evaluate, make_views, save_checkpoint,
                     seed_everything, student_parameters, train_batch, model_features)


ARMS = ("anchor", "dual_no_feedback", "dual_feedback")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def patient_id(name):
    match = re.match(r"SPIDER_(\d+)_", name)
    if match is None:
        raise ValueError(f"Cannot determine SPIDER patient from {name}")
    return match.group(1)


def uniform_indices(indices, count):
    indices = list(indices)
    if not 0 < count <= len(indices):
        raise ValueError(f"Cannot select {count} samples from {len(indices)}")
    return [indices[i] for i in np.linspace(0, len(indices) - 1, count, dtype=int)]


def validation_indices(names, count):
    groups = {}
    for index, name in enumerate(names):
        groups.setdefault(patient_id(name), []).append(index)
    patients = sorted(groups)
    selected = uniform_indices(range(len(patients)), min(count, len(patients)))
    return [groups[patients[i]][len(groups[patients[i]]) // 2] for i in selected]


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def snapshot_sources(output):
    sources = output / "sources"
    manifest = {}
    paths = [*ROOT.glob("*.py"), *ROOT.glob("model/*.py"), *ROOT.glob("configs/*.yaml"),
             *ROOT.glob("tests/*.py"), Path(__file__), ROOT / "docs/dual_student_design.md"]
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT)
        destination = sources / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "source_hashes.json", manifest)


@torch.no_grad()
def student_diagnostics(model, loader, cfg, device):
    if model.student2 is None:
        return None
    was_training = model.training
    model.eval()
    counts = dict(pixels=0, foreground_pixels=0, disagreement=0, foreground_disagreement=0,
                  s1_wrong=0, s2_correct_s1_wrong=0, s1_correct_s2_wrong=0, both_wrong=0)
    for batch in loader:
        low, high = make_views(batch, device, cfg.use_frequency)
        if cfg.secondary_view == "lf":
            high = low
        outputs = model(low, high, return_plain=False)
        label = batch["label"].to(device)
        predictions = []
        for route in ("lf_prompt", "hf_prompt"):
            small = outputs[route].argmax(1).float().unsqueeze(1)
            predictions.append(F.interpolate(small, size=label.shape[-2:], mode="nearest")[:, 0].long())
        p1, p2 = predictions
        foreground = label != 0
        wrong1, wrong2 = p1 != label, p2 != label
        counts["pixels"] += label.numel()
        counts["foreground_pixels"] += int(foreground.sum())
        counts["disagreement"] += int((p1 != p2).sum())
        counts["foreground_disagreement"] += int(((p1 != p2) & foreground).sum())
        counts["s1_wrong"] += int((wrong1 & foreground).sum())
        counts["s2_correct_s1_wrong"] += int((wrong1 & ~wrong2 & foreground).sum())
        counts["s1_correct_s2_wrong"] += int((~wrong1 & wrong2 & foreground).sum())
        counts["both_wrong"] += int((wrong1 & wrong2 & foreground).sum())
    model.train(was_training)
    return dict(counts,
                disagreement_rate=counts["disagreement"] / max(counts["pixels"], 1),
                foreground_disagreement_rate=counts["foreground_disagreement"] / max(counts["foreground_pixels"], 1),
                s2_correction_opportunity=counts["s2_correct_s1_wrong"] / max(counts["s1_wrong"], 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ftp_dual.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled-samples", type=int, default=48)
    parser.add_argument("--unlabeled-samples", type=int, default=72)
    parser.add_argument("--validation-patients", type=int, default=16)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--no-checkpoints", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.validation_patients, args.timeout_seconds) <= 0:
        parser.error("steps, validation-patients and timeout-seconds must be positive")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {output}")
    cfg = replace(load_config(args.config), seed=args.seed, batch_size=2, labeled_bs=1,
                  num_workers=0, max_iterations=args.steps, consistency_rampup_epochs=1,
                  feedback_start_epoch=1, feedback_rampup_epochs=2,
                  feedback_confidence=0.5, feedback_margin=0.05, feedback_weight=0.25)
    cfg.validate()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is not accessible")
    seed_everything(cfg)
    training = SpiderDataset(cfg.root_path, "train", cfg.image_size, cfg.labelnum, cfg.max_samples)
    validation = SpiderDataset(cfg.root_path, "valid", cfg.image_size)
    labeled = uniform_indices(range(cfg.labelnum), args.labeled_samples)
    unlabeled = uniform_indices(range(cfg.labelnum, len(training)), args.unlabeled_samples)
    valid = validation_indices(validation.names, args.validation_patients)
    train_patients = {patient_id(name) for name in training.names}
    valid_patients = {patient_id(name) for name in validation.names}
    overlap = sorted(train_patients & valid_patients)
    if overlap:
        raise RuntimeError(f"Train/validation patient overlap: {overlap}")
    manifest = {
        "protocol": "paired_short_pilot_v2_strict", "arguments": vars(args), "config": cfg.to_dict(),
        "labeled_files": [training.names[i] for i in labeled],
        "unlabeled_files": [training.names[i] for i in unlabeled],
        "validation_files": [validation.names[i] for i in valid],
        "train_validation_patient_overlap": overlap, "test_split_used": False,
        "torch_version": str(torch.__version__), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "strict_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "training_interpolation": "bilinear on CPU with autograd when CUDA strict mode is enabled",
        "scope": "one-seed short subset pilot; does not establish converged efficacy",
    }
    write_json(output / "manifest.json", manifest)
    snapshot_sources(output)
    text = model_features(cfg)
    valid_loader = DataLoader(Subset(validation, valid), batch_size=1, num_workers=0)
    started = time.monotonic()
    completed = {}
    reference = None
    write_json(output / "results.json", {"status": "running", "arms": completed})
    try:
        for arm in args.arms:
            arm_dir = output / arm
            arm_dir.mkdir()
            arm_cfg = replace(cfg, architecture="anchor" if arm == "anchor" else "dual",
                              feedback_weight=0.25 if arm == "dual_feedback" else 0.0,
                              output_dir=str(arm_dir))
            write_json(arm_dir / "config.json", arm_cfg.to_dict())
            seed_everything(arm_cfg)
            training.transform.set_random_state(seed=arm_cfg.seed)
            net = build_model(arm_cfg, text).to(device)
            initial_hash = state_hash(net.student)
            optimizer = torch.optim.AdamW(student_parameters(net), lr=cfg.base_lr,
                                          weight_decay=cfg.weight_decay, eps=1e-5, betas=(0.9, 0.999))
            sampler = TwoStreamBatchSampler(labeled, unlabeled, 2, 1, cfg.seed)
            loader = DataLoader(training, batch_sampler=sampler, num_workers=0,
                                generator=torch.Generator().manual_seed(cfg.seed))
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            initial = evaluate(net, valid_loader, arm_cfg, device)
            step = 0
            epoch = 0
            coverage = []
            active_steps = 0
            positive_steps = 0
            arm_started = time.monotonic()
            print(f"START {arm} S1={initial_hash[:12]} initial_dice={initial['dice']}", flush=True)
            with (arm_dir / "metrics.jsonl").open("w") as log:
                while step < args.steps:
                    sampler.epoch = epoch
                    for batch in loader:
                        if time.monotonic() - started > args.timeout_seconds:
                            raise TimeoutError("Pilot wall-time budget exhausted; incomplete arms are not compared")
                        for group in optimizer.param_groups:
                            group["lr"] = cfg.base_lr * (1 - step / args.steps) ** 0.9
                        metrics = train_batch(net, optimizer, batch, arm_cfg, device, epoch)
                        step += 1
                        metrics.update(kind="train", step=step, epoch=epoch)
                        log.write(json.dumps(metrics, allow_nan=False) + "\n")
                        log.flush()
                        if metrics["feedback_weight"] > 0:
                            active_steps += 1
                            coverage.append(metrics["feedback_coverage"])
                            positive_steps += int(metrics["feedback_pixels"] > 0)
                        if step == 1 or step % 20 == 0:
                            print(f"{arm} step={step}/{args.steps} sup_lf={metrics['supervised_lf']:.4f} "
                                  f"feedback={metrics['feedback']:.5f} "
                                  f"coverage={metrics.get('feedback_coverage', 0):.5f}", flush=True)
                        if step >= args.steps:
                            break
                    teacher_metrics = evaluate(net, valid_loader, arm_cfg, device)
                    log.write(json.dumps(dict(teacher_metrics, kind="validation", step=step, epoch=epoch)) + "\n")
                    log.flush()
                    print(f"{arm} epoch={epoch} step={step} teacher_dice={teacher_metrics['dice']}", flush=True)
                    epoch += 1
            networks = ["teacher", "student1"]
            if net.student2 is not None:
                networks += ["student2", "students_mean"]
            results = {name: evaluate(net, valid_loader, arm_cfg, device, network=name) for name in networks}
            final_hash = state_hash(net.student)
            teacher_hash = state_hash(net.teacher)
            difference = None
            if arm == "anchor":
                reference = {member: {key: value.detach().cpu().clone() for key, value in getattr(net, member).state_dict().items()}
                             for member in ("student", "teacher")}
            elif arm == "dual_no_feedback" and reference is not None:
                difference = {}
                for member in ("student", "teacher"):
                    difference[member] = max(float((value.detach().cpu().double() - reference[member][key].double()).abs().max())
                                             for key, value in getattr(net, member).state_dict().items())
            result = {
                "arm": arm, "steps": step, "initial_s1_hash": initial_hash,
                "final_s1_hash": final_hash, "final_teacher_hash": teacher_hash,
                "initial_teacher": initial, "final": results,
                "negative_control_max_abs_difference": difference,
                "feedback_active_steps": active_steps, "feedback_nonempty_steps": positive_steps,
                "mean_active_feedback_coverage": float(np.mean(coverage)) if coverage else 0.0,
                "diagnostics": student_diagnostics(net, valid_loader, arm_cfg, device),
                "trainable_parameters": sum(p.numel() for p in student_parameters(net)),
                "seconds": time.monotonic() - arm_started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20 if device.type == "cuda" else None,
            }
            if not args.no_checkpoints:
                dice = results["teacher"]["dice"]
                save_checkpoint(arm_dir / "last.pt", net, optimizer, arm_cfg, epoch - 1, step,
                                dice if dice is not None else -1)
            write_json(arm_dir / "result.json", result)
            completed[arm] = result
            write_json(output / "results.json", {"status": "running", "arms": completed})
            if difference is not None and any(value != 0 for value in difference.values()):
                raise RuntimeError(f"Negative control failed: {difference}; remaining arms were not started")
            print(f"DONE {arm} teacher_dice={results['teacher']['dice']} seconds={result['seconds']:.1f}", flush=True)
            del net, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_json(output / "results.json", {"status": "completed", "arms": completed,
                                               "seconds": time.monotonic() - started,
                                               "scope": manifest["scope"]})
    except Exception as error:
        write_json(output / "results.json", {"status": "incomplete", "arms": completed,
                                               "error": f"{type(error).__name__}: {error}",
                                               "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
